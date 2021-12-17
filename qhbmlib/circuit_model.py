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
"""Tools for defining quantum circuit models."""

from typing import List, Union

import cirq
import numpy as np
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib import circuit_model_utils


class QuantumCircuit(tf.keras.layers.Layer):
  """Class for representing a quantum circuit."""

  def __init__(self,
               pqc: cirq.Circuit,
               symbols: tf.Tensor,
               value_layers_inputs: Union[tf.Variable, List[tf.Variable]],
               value_layers: List[tf.keras.layers.Layer],
               name: Union[None, str] = None):
    super().__init__(name=name)

    if set(tfq.util.get_circuit_symbols(pqc)) != {
        s.decode("utf-8") for s in symbols.numpy()
    }:
      raise ValueError(
          "`pqc` must contain exactly the parameters in `symbols`.")
    self._qubits = sorted(pqc.all_qubits())
    self._symbols = symbols
    self._value_layers = value_layers
    self._value_layers_inputs = value_layers_inputs

    self._pqc = tfq.convert_to_tensor([pqc])
    self._inverse_pqc = tfq.convert_to_tensor([pqc**-1])

    raw_bit_circuit = circuit_model_utils.bit_circuit(self.qubits)
    bit_symbols = list(sorted(tfq.util.get_circuit_symbols(raw_bit_circuit)))
    self._bit_symbols = tf.constant([str(x) for x in bit_symbols])
    self._bit_circuit = tfq.convert_to_tensor([raw_bit_circuit])

  @property
  def qubits(self):
    return self._qubits

  @property
  def symbols(self):
    """1D `tf.Tensor` of strings which are the parameters of circuit."""
    return self._symbols

  @property
  def value_layers(self):
    """List of Keras layers which calculate the current parameter values.

    This list of layers is where the caller would access model weights to be
    updated from a secondary model or hypernetwork.
    """
  return self._value_layers
  
  @property
  def values(self):
    """1D `tf.Tensor` of floats specifying the current values of the parameters.

    This should be structured such that `self.values[i]` is the current value of
    `self.symbols[i]` in `self.pqc` and `self.inverse_pqc`.
    """
    x = self._value_layers_inputs
    for layer in self._value_layers:
      x = layer(x)
    return x

  @property
  def pqc(self):
    return self._pqc

  @property
  def inverse_pqc(self):
    return self._inverse_pqc

  def build(self, unused):
    """Builds the layers which calculate the values.

    The input shape is unused because it is known to be the shape of
    `self._value_layers_inputs`.
    """
    x = [tf.shape(t) for t in self._value_layers_inputs]
    for layer in self._energy_layers:
      x = layer.compute_output_shape(x)
  
  def call(self, inputs):
    """Inputs are bitstrings prepended as initial states to `self.pqc`."""
    num_bitstrings = tf.shape(inputs)[0]
    bit_circuits = tfq.resolve_parameters(
        tf.tile(self._bit_circuit, [num_bitstrings]), self._bit_symbols,
        tf.cast(inputs, tf.float32))
    pqcs = tf.tile(self.pqc, [num_bitstrings])
    return tfq.append_circuit(bit_circuits, pqcs)


class DirectQuantumCircuit(QuantumCircuit):
  """QuantumCircuit with direct map from model variables to circuit params."""

  def __init__(
      self,
      pqc: cirq.Circuit,
      initializer: tf.keras.initializers.Initializer = tf.keras.initializers
      .RandomUniform(0, 2 * np.pi),
      name: Union[None, str] = None,
  ):
    """Initializes a DirectQuantumCircuit.

    Args:
      pqc: Representation of a parameterized quantum circuit.
      initializer: A `tf.keras.initializers.Initializer` which specifies how to
        initialize the values of the parameters in `circuit`.  This argument is
        ignored if `values` is not None.
      name: Identifier for this DirectQuantumCircuit.
    """
    raw_symbols = list(sorted(tfq.util.get_circuit_symbols(pqc)))
    symbols = tf.constant([str(x) for x in raw_symbols], dtype=tf.string)
    values = tf.Variable(initializer(shape=[len(raw_symbols)]))
    super().__init__(pqc, symbols, values, [])


class QAIA(QuantumCircuit):
  """Quantum circuit defined by a classical energy and a Hamiltonian."""

  def __init__(self,
               quantum_h_terms: List[cirq.PauliSum],
               classical_h_terms: List[cirq.PauliSum],
               num_layers: int,
               initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi),
               name=None):
    """Initializes a QAIA."""

    quantum_symbols = []
    classical_symbols = []
    for j in range(num_layers):
      quantum_symbols.append([])
      classical_symbols.append([])
      for k, _ in enumerate(quantum_h_terms):
        quantum_symbols[-1].append(f"gamma_{j}_{k}")
      for k, _ in enumerate(classical_h_terms):
        classical_symbols[-1].append(f"eta_{j}_{k}")

    pqc = cirq.Circuit()
    flat_symbols = []
    for q_symb, c_symb in zip(quantum_symbols, classical_symbols):
      pqc += tfq.util.exponential(quantum_h_terms, coefficients=q_symb)
      pqc += tfq.util.exponential(classical_h_terms, coefficients=c_symb)
      flat_symbols.extend(q_symb + c_symb)
    symbols = tf.constant(flat_symbols)

    value_layers_inputs = [
        tf.Variable(initializer(shape=[num_layers])),  # true etas
        tf.Variable(initializer(shape=[len(classical_h_terms)])),  # thetas
        tf.Variable(
            initializer(shape=[num_layers, len(quantum_h_terms)])),  # gammas
    ]

    def embed_params(inputs):
      """Tiles up the variables to properly tie QAIA parameters."""
      exp_etas = tf.expand_dims(inputs[0], -1)
      tiled_thetas = tf.tile(inputs[1], [tf.shape(inputs[0])[0], 1])
      classical_params = exp_etas * tiled_thetas
      return tf.concat([classical_params, inputs[2]], 1)

    value_layers = [tf.keras.layers.Lambda(embed_params)]

    super().__init__(pqc, symbols, value_layers_inputs, value_layers)
