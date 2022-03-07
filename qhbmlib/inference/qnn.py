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

from qhbmlib.models import circuit  # pylint: disable=unused-import
from qhbmlib.models import energy
from qhbmlib.models import hamiltonian
from qhbmlib import utils


class QuantumInference(tf.keras.layers.Layer):
  """Methods for inference on QuantumCircuit objects."""

  def __init__(self,
               input_circuit: circuit.QuantumCircuit,
               expectation_samples: Union[None, int] = None,
               backend: Union[str, cirq.Sampler] = "noiseless",
               differentiator: Union[None,
                                     tfq.differentiators.Differentiator] = None,
               name: Union[None, str] = None):
    """Initialize a QuantumInference layer.

    Args:
      input_circuit: The parameterized quantum circuit on which to do inference.
      expectation_samples: Number of samples to use when estimating the
        expectation value of a Hamiltonian with a general BitstringEnergy.
        If None, can only use Hamiltonians whose energy inherits PauliMixin.
      backend: Specifies what backend TFQ will use to compute expectation
        values. `str` options are {"noisy", "noiseless"}; users may also specify
        a preconfigured cirq execution object to use instead.
      differentiator: Specifies how to take the derivative of a quantum circuit.
        Note that derivatives of expectation values of general Hamiltonian
        observables are only supported if this value is not None.
      name: Identifier for this inference engine.
    """
    input_circuit.build([])
    self._circuit = input_circuit
    if expectation_samples is None:
      self._expectation_samples = None
    else:
      # Expand for compatibility with sample layer
      self._expectation_samples = tf.constant([expectation_samples],
                                              dtype=tf.int32)
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

  # TODO(#201): consider Hamiltonian type renaming
  def _sampled_expectation(self, initial_states: tf.Tensor,
                           observable: hamiltonian.Hamiltonian):
    """Returns the expectation values of the observables against the QNN.

    Args:
      initial_states: Shape [batch_size, num_qubits] of dtype `tf.int8`.
        Each entry is an initial state for the set of qubits.  For each state,
        `qnn` is applied and the pure state expectation value is calculated.
      observable: Hermitian operator to measure.  Will be tiled to measure
        the expectation value of the observable in the state
        `qnn|initial_states[i]>` batched over `i`.

    Returns:
      `tf.Tensor` with shape [batch_size, 1] whose entries are the
      unaveraged expectation values of `observable` against each transformed
      initial state.
    """

    @tf.custom_gradient
    def _inner_expectation():
      """Enables derivatives."""
      # control the order of variables handed to the later gradient function.
      # see the implementation of `tf.custom_gradient` at the following link:
      # https://github.com/tensorflow/tensorflow/blob/3f878cff5b698b82eea85db2b60d65a2e320850e/tensorflow/python/ops/custom_gradient.py#L530
      _ = [tf.identity(v) for v in observable.energy.trainable_variables]
      _ = [tf.identity(v) for v in observable.circuit.trainable_variables]

      unique_states, idx, _ = utils.unique_bitstrings_with_counts(initial_states)
      u = self.circuit + observable.circuit_dagger
      circuits = u(unique_states)
      num_circuits = tf.shape(circuits)[0]
      tiled_values = tf.tile(tf.expand_dims(u.symbol_values, 0), [num_circuits, 1])
      samples = self._sample_layer(
          circuits,
          symbol_names=u.symbol_names,
          symbol_values=tiled_values,
          repetitions=self._expectation_samples).to_tensor()
      with tf.GradientTape() as thetas_tape:
        unique_expectations = tf.map_fn(
            lambda x: tf.math.reduce_mean(observable.energy(x)),
            samples,
            fn_output_signature=tf.float32)
      forward_pass = tf.expand_dims(
        utils.expand_unique_results(unique_expectations, idx), 1)

      def grad_fn(*upstream, variables):
        """See equation A5 in the QHBM paper appendix for details.

        # TODO(#119): confirm equation number.
        """
        unique_thetas_grads = thetas_tape.jacobian(
            unique_expectations,
            observable.energy.trainable_variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
        thetas_grads = [
            utils.expand_unique_results(g, idx) for g in unique_thetas_grads]

        (
          batch_programs, new_symbol_names, batch_symbol_values,
          batch_weights, batch_mapper
        ) = self.differentiator.get_gradient_circuits(
            circuits, u.symbol_names, tiled_values)
        n_batch_programs = tf.reduce_prod(tf.shape(batch_programs))
        n_symbols = tf.shape(new_symbol_names)[0]

        gradient_samples = self._sample_layer(
            circuits,
            symbol_names=u.symbol_names,
            symbol_values=tf.reshape(
              batch_symbol_values, [n_batch_programs, n_symbols]),
            repetitions=self._expectation_samples).to_tensor()
        gradient_expectations = tf.map_fn(
            lambda x: tf.math.reduce_mean(observable.energy(x)),
            gradient_samples,
            fn_output_signature=tf.float32)

        batch_jacobian = tf.map_fn(
            lambda x: tf.einsum('km,kmp->kp', x[0], tf.gather(x[1], x[2])),
            (batch_weights, batch_expectations, batch_mapper),
            fn_output_signature=tf.float32)
        unique_phis_gradients = tf.reduce_sum(batch_jacobian, -1)
        phis_gradients = [
            utils.expand_unique_results(g, idx) for g in unique_phis_gradients]

        return tuple(), [upstream * g for g in thetas_gradients + phis_gradients]
      return forward_pass, grad_fn
    return _inner_expectation()

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
      return self._sampled_expectation(initial_states, observables)

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
