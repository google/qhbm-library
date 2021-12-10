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

import abc
from typing import List

import cirq
import numpy as np
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq


def check_layers(layers):
  """Confirms the input is a valid list of keras Layers."""
  if not isinstance(layers, list) or not all(
      [isinstance(e, tf.keras.layers.Layer) for e in layers]):
    raise TypeError("must be a list of keras layers.")
  return layers


def bit_circuit(qubits, name="bit_circuit"):
  """Returns exponentiated X gate on each qubit and the exponent symbols."""
  circuit = cirq.Circuit()
  for n, q in enumerate(qubits):
    bit = sympy.Symbol("{0}_bit_{1}".format(name, n))
    circuit += cirq.X(q)**bit
  return circuit


class QuantumCircuit(tf.keras.layers.Layer):
  """Class for representing a quantum circuit."""

  def __init__(self, pqc, symbols, value_layers_inputs, value_layers, name=None):
    super().__init__(name=name)

    if not isinstance(pqc, cirq.Circuit):
      raise TypeError(f"pqc must be a cirq.Circuit object."
                      " Given: {type(pqc)}")

    if set(tfq.util.get_circuit_symbols(pqc)) != {
        s.decode("utf-8") for s in symbols.numpy()
    }:
      raise ValueError(
        "`pqc` must contain exactly the parameters in `symbols`."
      )
    self._qubits = sorted(pqc.all_qubits())
    self._symbols = symbols
    self._value_layers = check_layers(value_layers)
    self._value_layers_inputs = value_layers_inputs

    test_values = self.values
    values_shape = tf.shape(test_values)
    symbols_shape = tf.shape(self.symbols)
    if not all(values_shape == symbols_shape):
      raise ValueError(
        "value calculation must yield one value for every symbol."
        f" Got {values_shape} and {symbols_shape}.")

    self._pqc = tfq.convert_to_tensor([pqc])
    self._inverse_pqc = tfq.convert_to_tensor([pqc**-1])

    _bit_circuit = bit_circuit(self.qubits)
    bit_symbols = list(sorted(tfq.util.get_circuit_symbols(_bit_circuit)))
    self._bit_symbols = tf.constant([str(x) for x in bit_symbols])
    self._bit_circuit = tfq.convert_to_tensor([_bit_circuit])
    
  @property
  def qubits(self):
    return self._qubits

  @property
  def symbols(self):
    """1D `tf.Tensor` of strings which are the parameters of circuit."""
    return self._symbols

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
      pqc,
      initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi),
      name=None,
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


class Squeeze(tf.keras.layers.Layer):
  """Wraps tf.squeeze in a Keras Layer."""

  def __init__(self, axis=None):
    """Initializes a Squeeze layer.
    Args:
      axis: An optional list of ints. Defaults to []. If specified, only
        squeezes the dimensions listed. The dimension index starts at 0. It is
        an error to squeeze a dimension that is not 1. Must be in the range
        [-rank(input), rank(input)). Must be specified if input is
        a RaggedTensor.
    """
    super().__init__()
    if axis is None:
      axis = []
    self._axis = axis

  def call(self, inputs):
    """Applies tf.squeeze to the inputs."""
    return tf.squeeze(inputs, axis=self._axis)


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

    num_true_etas = num_layers
    num_thetas = len(classical_h_terms)
    num_gammas = len(quantum_h_terms) * num_layers

    value_layers_inputs = [
      tf.Variable(initializer(shape=[num_layers])),  # true etas
      tf.Variable(initializer(shape=[len(classical_h_terms)])),  # thetas
      tf.Variable(initializer(shape=[num_layers, len(quantum_h_terms)])),  # gammas
    ]

    def embed_params(inputs):
      """Tiles up the variables to properly tie QAIA parameters."""
      exp_etas = tf.expand_dims(inputs[0], -1)
      tiled_thetas = tf.tile(inputs[1], [tf.shape(inputs[0])[0], 1])
      classical_params = exp_etas * tiled_thetas
      return tf.concat([classical_params, inputs[2]], 1)

    value_layers = [
      tf.keras.layers.Lambda(embed_params)
    ]

    super().__init__(pqc, symbols, value_layers_inputs, value_layers)
