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
"""Tools for inference on quantum circuits."""

from typing import List

import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib import circuit_model


class BitstringInjector(tf.keras.Model):
  """Class for combining bitstrings and quantum circuits."""

  def __init__(self, qubits, name="bit_circuit"):
    """Initializes a BitstringInjector."""
    super().__init__(name=name)
    self.qubits = qubits
    circuit = cirq.Circuit()
    symbols = []
    for n, q in enumerate(qubits):
      bit = sympy.Symbol("{0}_bit_{1}".format(name, n))
      circuit += cirq.X(q)**bit
      symbols.append(bit)
    self.bit_circuit = tfq.convert_to_tensor([circuit])
    self.bit_symbols = tf.constant([str(x) for x in symbols])

  def inject_bitstrings(self, bitstrings, qnn: circuit_model.QuantumCircuit):
    """Returns circuits which are the qnn appended to bitstring injections."""
    num_bitstrings = tf.shape(bitstrings)[0]
    bit_circuits = tfq.resolve_parameters(
        tf.tile(self.bit_circuit, [num_bitstrings]), self.bit_symbols,
        tf.cast(bitstrings, tf.float32))
    pqcs = tf.tile(qnn.pqc, [num_bitstrings])
    return tfq.append_circuit(bit_circuits, pqcs)

  def call(self, bitstrings, qnn):
    return self.inject_bitstrings(bitstrings, qnn)

  
class Expectation(tf.keras.Model):
  """Class for taking expectation values of quantum circuits."""
  
  def __init__(self,
               qubits: List[cirq.GridQubit],
               backend="noiseless",
               differentiator=None,
               name=None):
    """Initialize an instance of Expectation.

    Args:
      qubits: The qubits on which this class makes inferences.
      backend: Optional Python `object` that specifies what backend TFQ will use
        to compute expectation values. Options are {"noisy", "noiseless"}.
        Users may also specify a preconfigured cirq execution
        object to use instead, which must inherit `cirq.Sampler`.
      differentiator: Either None or a `tfq.differentiators.Differentiator`,
        which specifies how to take the derivative of a quantum circuit.
      name: Identifier for this inference engine.
    """
    super().__init__(name=name)

    self._qubits = qubits
    self._bitstring_injector = BitstringInjector(qubits, name)
    
    self._differentiator = differentiator
    if backend == "noiseless" or backend is None:
      self._backend = "noiseless"
      self._expectation_layer = tfq.layers.Expectation(
          backend=backend, differentiator=differentiator)
      def _expectation_function(circuits, symbol_names, symbol_values, operators, _):
        return self._expectation_layer(
          circuits,
          symbol_names=symbol_names,
          symbol_values=symbol_values,
          operators=operators
        )
    else:
      self._backend = backend
      self._expectation_layer = tfq.layers.SampledExpectation(
          backend=backend, differentiator=differentiator)
      def _expectation_function(circuits, symbol_names, symbol_values, operators, repetitions):
        return self._expectation_layer(
          circuits,
          symbol_names=symbol_names,
          symbol_values=symbol_values,
          operators=operators,
          repetitions=repetitions,
        )
    self._expectation_function = _expectation_function

  @property
  def qubits(self):
    return self._qubits

  @property
  def backend(self):
    return self._backend

  @property
  def differentiator(self):
    return self._differentiator

  def expectation(self,
                  qnn: circuit_model.QuantumCircuit,
                  initial_states: tf.Tensor,
                  counts: tf.Tensor,
                  operators: tf.Tensor,
                  reduce: bool=True):
    """Returns the expectation values of the operators against the QNN.

      Args:
        qnn: Unitary to feed forward.
        operators: `tf.Tensor` of strings with shape [n_ops], result of calling
          `tfq.convert_to_tensor` on a list of cirq.PauliSum, `[op1, op2, ...]`.
          Will be tiled to measure `<op_j>_(qnn)|initial_states[i]>`
          for each i and j.
        initial_states: Shape [batch_size, num_qubits] of dtype `tf.int8`.
          These are the initial states of each qubit in the circuit.
        counts: Shape [batch_size] of dtype `tf.int32` such that `counts[i]` is
          the weight of `initial_states[i]` when computing expectations.
          Additionally, if `self.backend != "noiseless", `counts[i]` samples
          are drawn from `(qnn)|initial_states[i]>` and used to compute
          the the corresponding expectation.
        reduce: bool flag for whether or not to average over i.

      Returns:
        If `reduce` is true, a `tf.Tensor` with shape [n_ops] whose entries are
        are the batch-averaged expectation values of `operators`.
        Else, a `tf.Tensor` with shape [batch_size, n_ops] whose entries are the
        unaveraged expectation values of each `operator` against each `circuit`.
      """
    circuits = self._bitstring_injector(initial_states, qnn)
    symbol_names = qnn.symbols
    symbol_values = qnn.values
    num_circuits = tf.shape(circuits)[0]
    num_operators = tf.shape(operators)[0]
    tiled_values = tf.tile(tf.expand_dims(symbol_values, 0), [num_circuits, 1])
    tiled_operators = tf.tile(tf.expand_dims(operators, 0), [num_circuits, 1])
    expectations = self._expectation_function(
          circuits,
          symbol_names=symbol_names,
          symbol_values=tiled_values,
          operators=tiled_operators,
          repetitions=tf.tile(tf.expand_dims(counts, 1), [1, num_operators]),
    )
    if reduce:
      probs = tf.cast(counts, tf.float32) / tf.cast(
          tf.reduce_sum(counts), tf.float32)
      return tf.reduce_sum(tf.transpose(probs * tf.transpose(expectations)), 0)
    return expectations

  def call(self,
           qnn,
           initial_states,
           counts,
           operators,
           reduce=True):
    return expectation(qnn,
                       initial_states,
                       counts,
                       operators,
                       reduce=reduce)


