# Copyright 2021 The QHBM Library Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tools for inference on quantum circuits represented by QuantumCircuit."""

from typing import Union

import cirq
import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib.model import circuit
from qhbmlib.model import energy
from qhbmlib.model import hamiltonian
from qhbmlib import utils


class QuantumInference(tf.keras.layers.Layer):
  """Methods for inference on QuantumCircuit objects."""

  def __init__(self,
               input_circuit: circuit.QuantumCircuit,
               backend: Union[str, cirq.Sampler] = "noiseless",
               differentiator: Union[None,
                                     tfq.differentiators.Differentiator] = None,
               name: Union[None, str] = None):
    """Initialize a QuantumInference layer.

    Args:
      input_circuit: The parameterized quantum circuit on which to do inference.
      backend: Specifies what backend TFQ will use to compute expectation
        values. `str` options are {"noisy", "noiseless"}; users may also specify
        a preconfigured cirq execution object to use instead.
      differentiator: Specifies how to take the derivative of a quantum circuit.
      name: Identifier for this inference engine.
    """
    input_circuit.build([])
    self._circuit = input_circuit
    self._differentiator = differentiator
    self._backend = backend
    self._sample_layer = tfq.layers.Sample(backend=backend)
    if backend == "noiseless":
      self._expectation_layer = tfq.layers.Expectation(
          backend=backend, differentiator=differentiator)

      def _expectation_function(circuits, symbol_names, symbol_values,
                                operators, *args):
        del args
        return self._expectation_layer(
            circuits,
            symbol_names=symbol_names,
            symbol_values=symbol_values,
            operators=operators)
    else:
      self._expectation_layer = tfq.layers.SampledExpectation(
          backend=backend, differentiator=differentiator)

      def _expectation_function(circuits, symbol_names, symbol_values,
                                operators, repetitions):
        return self._expectation_layer(
            circuits,
            symbol_names=symbol_names,
            symbol_values=symbol_values,
            operators=operators,
            repetitions=repetitions,
        )

    self._expectation_function = _expectation_function
    super().__init__(name=name)

  @property
  def backend(self):
    return self._backend

  @property
  def circuit(self):
    return self._circuit

  @property
  def differentiator(self):
    return self._differentiator

  def expectation(self, initial_states: tf.Tensor,
                  observables: Union[tf.Tensor, hamiltonian.Hamiltonian]):
    """Returns the expectation values of the observables against the QNN.

    Args:
      initial_states: Shape [batch_size, num_qubits] of dtype `tf.int8`.
        Each entry is an initial state for the set of qubits.  For each state,
        `qnn` is applied and the pure state expectation value is calculated.
      observables: Hermitian operators to measure.  If `tf.Tensor`, strings with
        shape [n_ops], result of calling `tfq.convert_to_tensor` on a list of
        cirq.PauliSum, `[op1, op2, ...]`.  Otherwise, a Hamiltonian.  Will be
        tiled to measure `<op_j>_((qnn)|initial_states[i]>)` for each i and j.

    Returns:
      `tf.Tensor` with shape [batch_size, n_ops] whose entries are the
      unaveraged expectation values of each `operator` against each
      transformed initial state.
    """
    if isinstance(observables, tf.Tensor):
      u = self.circuit
      ops = observables
      post_process = lambda x: x
    elif isinstance(observables.energy, energy.PauliMixin):
      u = self.circuit + observables.circuit_dagger
      ops = observables.operator_shards
      post_process = lambda y: tf.map_fn(
          lambda x: tf.expand_dims(
              observables.energy.operator_expectation(x), 0), y)
    else:
      raise NotImplementedError(
          "General `BitstringEnergy` models not yet supported.")

    unique_states, idx, counts = utils.unique_bitstrings_with_counts(
        initial_states)
    circuits = u(unique_states)
    num_circuits = tf.shape(circuits)[0]
    num_ops = tf.shape(ops)[0]
    tiled_values = tf.tile(
        tf.expand_dims(u.symbol_values, 0), [num_circuits, 1])
    tiled_ops = tf.tile(tf.expand_dims(ops, 0), [num_circuits, 1])
    expectations = self._expectation_function(
        circuits,
        u.symbol_names,
        tiled_values,
        tiled_ops,
        tf.tile(tf.expand_dims(counts, 1), [1, num_ops]),
    )
    return utils.expand_unique_results(post_process(expectations), idx)

  def sample(self, initial_states: tf.Tensor, counts: tf.Tensor):
    """Returns bitstring samples from the QNN.

      Args:
        initial_states: Shape [batch_size, num_qubits] of dtype `tf.int8`.
          These are the initial states of each qubit in the circuit.
        counts: Shape [batch_size] of dtype `tf.int32` such that `counts[i]` is
          the number of samples to draw from `(qnn)|initial_states[i]>`.

      Returns:
        ragged_samples: `tf.RaggedTensor` of DType `tf.int8` structured such
          that `ragged_samples[i]` contains `counts[i]` bitstrings drawn from
          `(qnn)|initial_states[i]>`.
    """
    circuits = self.circuit(initial_states)
    num_circuits = tf.shape(circuits)[0]
    tiled_values = tf.tile(
        tf.expand_dims(self.circuit.symbol_values, 0), [num_circuits, 1])
    num_samples_mask = tf.cast((tf.ragged.range(counts) + 1).to_tensor(),
                               tf.bool)
    num_samples_mask = tf.map_fn(tf.random.shuffle, num_samples_mask)
    samples = self._sample_layer(
        circuits,
        symbol_names=self.circuit.symbol_names,
        symbol_values=tiled_values,
        repetitions=tf.expand_dims(tf.math.reduce_max(counts), 0))
    return tf.ragged.boolean_mask(samples, num_samples_mask)