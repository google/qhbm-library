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

import abc
from typing import Union

import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib.models import circuit  # pylint: disable=unused-import
from qhbmlib.models import energy
from qhbmlib.models import hamiltonian
from qhbmlib import utils


class QuantumInference(tf.keras.layers.Layer, abc.ABC):
  """Interface for inference on quantum circuits."""

  def __init__(self,
               input_circuit: circuit.QuantumCircuit,
               name: Union[None, str] = None):
    """Initializes a generic QuantumInference layer.

    Args:
      input_circuit: The parameterized quantum circuit on which to do inference.
      name: Identifier for this inference engine.
    """
    super().__init__(name=name)
    input_circuit.build([])
    self._circuit = input_circuit

  @property
  def circuit(self):
    return self._circuit

  # TODO(#201): consider Hamiltonian type renaming
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
    unique_states, idx, _ = utils.unique_bitstrings_with_counts(initial_states)
    if isinstance(observables, tf.Tensor):
      total_circuit = self.circuit
    else:
      total_circuit = self.circuit + observables.circuit_dagger
    circuits = total_circuit(unique_states)
    num_circuits = tf.shape(circuits)[0]
    tiled_values = tf.tile(
        tf.expand_dims(total_circuit.symbol_values, 0), [num_circuits, 1])
    unique_expectations = self._expectation(circuits,
                                            total_circuit.symbol_names,
                                            tiled_values, observables)
    return utils.expand_unique_results(unique_expectations, idx)

  @abc.abstractmethod
  def _expectation(self, circuits, symbol_names, symbol_values, observables):
    raise NotImplementedError()


class AnalyticQuantumInference(QuantumInference):
  """Analytic methods for inference on QuantumCircuit objects.

  This class uses the TensorFlow Quantum `Expectation` layer to compute
  expectation values of observables.  It uses adjoint gradients to compute
  the derivatives of those expectation values.

  Why shouldn't we use the `ParameterShift` differentiator with this class?
  First, in this class expectation values of observables are exact irrespective
  of the chosen differentiator.  Second, parameter shift derivatives are exactly
  equal to the true derivative in the noiseless, exact-expectation regime.  Thus
  using it would just increase computational overhead without gaining additional
  accuracy.
  """

  def __init__(self,
               input_circuit: circuit.QuantumCircuit,
               name: Union[None, str] = None):
    """Initialize an AnalyticQuantumInference layer.

    Args:
      input_circuit: The parameterized quantum circuit on which to do inference.
      name: Identifier for this inference engine.
    """
    super().__init__(input_circuit, name)
    self._expectation_layer = tfq.layers.Expectation()

  def _expectation(self, circuits, symbol_names, symbol_values, observables):
    """See base class docstring.

    Note that a `hamiltonian.Hamiltonian` object is only accepted if its
    energy function inherits from `energy.PauliMixin`.
    """
    if isinstance(observables, tf.Tensor):
      ops = observables
      post_process = lambda x: x
    elif isinstance(observables.energy, energy.PauliMixin):
      ops = observables.operator_shards
      post_process = lambda y: tf.map_fn(
          lambda x: tf.expand_dims(
              observables.energy.operator_expectation(x), 0), y)
    else:
      raise TypeError("General Hamiltonians not accepted.  "
                      "Please use `SampledQuantumInference` instead.")

    num_circuits = tf.shape(circuits)[0]
    tiled_ops = tf.tile(tf.expand_dims(ops, 0), [num_circuits, 1])
    expectations = self._expectation_layer(
        circuits,
        symbol_names=symbol_names,
        symbol_values=symbol_values,
        operators=tiled_ops)
    return post_process(expectations)


class SampledQuantumInference(QuantumInference):
  """Sampling methods for inference on QuantumCircuit objects.

  This class uses the TensorFlow Quantum `SampledExpectation` and `Sample`
  layers to compute expectation values of observables.  It uses parameter
  shift gradients to compute derivatives of those expectation values.
  """

  def __init__(self,
               input_circuit: circuit.QuantumCircuit,
               expectation_samples: int,
               name: Union[None, str] = None):
    """Initialize an SampledQuantumInference layer.

    Args:
      input_circuit: The parameterized quantum circuit on which to do inference.
      expectation_samples: Number of samples to use when estimating the
        expectation value of each observable on each input circuit.
      name: Identifier for this inference engine.
    """
    super().__init__(input_circuit, name)
    # Expand for compatibility with sample layer
    self._expectation_samples = tf.constant([expectation_samples],
                                            dtype=tf.int32)
    self._sample_layer = tfq.layers.Sample()
    self._expectation_layer = tfq.layers.SampledExpectation()
    self._differentiator = tfq.differentiators.ParameterShift()

  def _sampled_expectation(self, circuits, symbol_names, symbol_values,
                           observable):

    @tf.custom_gradient
    def _inner_expectation(circuits, symbol_names, symbol_values):
      """Enables derivatives."""
      num_circuits = tf.shape(circuits)[0]
      unique_samples = self._sample_layer(
          circuits,
          symbol_names=symbol_names,
          symbol_values=symbol_values,
          repetitions=self._expectation_samples).to_tensor()
      with tf.GradientTape() as thetas_tape:
        unique_expectations = tf.map_fn(
            lambda x: tf.math.reduce_mean(observable.energy(x)),
            unique_samples,
            fn_output_signature=tf.float32)
        forward_pass = tf.expand_dims(unique_expectations, 1)

      def grad_fn(*upstream, variables):
        """Use `get_gradient_circuits` method to get QNN variable derivatives"""
        # This block adapted from my `differentiate_sampled` in TFQ.
        (batch_programs, new_symbol_names, batch_symbol_values, batch_weights,
         batch_mapper) = self._differentiator.get_gradient_circuits(
             circuits, symbol_names, symbol_values)
        m_i = tf.shape(batch_programs)[1]
        # shape is [num_circuits, m_i, n_ops]
        n_batch_programs = tf.size(batch_programs)
        n_symbols = tf.shape(new_symbol_names)[0]
        gradient_samples = self._sample_layer(
            tf.reshape(batch_programs, [n_batch_programs]),
            symbol_names=new_symbol_names,
            symbol_values=tf.reshape(batch_symbol_values,
                                     [n_batch_programs, n_symbols]),
            repetitions=self._expectation_samples).to_tensor()
        gradient_expectations = tf.map_fn(
            lambda x: tf.math.reduce_mean(observable.energy(x)),
            gradient_samples,
            fn_output_signature=tf.float32)
        # last dimension is number of observables.
        # TODO(#207): parameterize it if more than one observable is accepted.
        batch_expectations = tf.reshape(gradient_expectations,
                                        [num_circuits, m_i, 1])

        # In the einsum equation, s is the symbols index, m is the
        # differentiator tiling index, o is the observables index.
        # `batch_jacobian` has shape [num_unique_programs, n_symbols, n_ops]
        unique_batch_jacobian = tf.map_fn(
            lambda x: tf.einsum("sm,smo->so", x[0], tf.gather(
                x[1], x[2], axis=0)),
            (batch_weights, batch_expectations, batch_mapper),
            fn_output_signature=tf.float32)

        # Connect upstream to symbol_values gradient
        symbol_values_gradients = tf.einsum("pso,po->ps", unique_batch_jacobian,
                                            upstream[0])

        thetas_gradients = thetas_tape.gradient(
            forward_pass,
            variables,
            output_gradients=upstream[0],
            unconnected_gradients=tf.UnconnectedGradients.ZERO)

        # Note: upstream gradient is already a coefficient below.
        return (None, None, symbol_values_gradients), thetas_gradients

      return forward_pass, grad_fn

    return _inner_expectation(circuits, symbol_names, symbol_values)

  def _expectation(self, circuits, symbol_names, symbol_values, observables):
    if isinstance(observables, tf.Tensor):
      ops = observables
      post_process = lambda x: x
    elif isinstance(observables.energy, energy.PauliMixin):
      ops = observables.operator_shards
      post_process = lambda y: tf.map_fn(
          lambda x: tf.expand_dims(
              observables.energy.operator_expectation(x), 0), y)
    else:
      return self._sampled_expectation(circuits, symbol_names, symbol_values,
                                       observables)

    num_circuits = tf.shape(circuits)[0]
    num_ops = tf.shape(ops)[0]
    tiled_ops = tf.tile(tf.expand_dims(ops, 0), [num_circuits, 1])
    repetitions = tf.tile(
        tf.expand_dims(self._expectation_samples, 1), [num_circuits, num_ops])
    expectations = self._expectation_layer(
        circuits,
        symbol_names=symbol_names,
        symbol_values=symbol_values,
        operators=tiled_ops,
        repetitions=repetitions)
    return post_process(expectations)

  def _sample(self, initial_states: tf.Tensor, counts: tf.Tensor):
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
