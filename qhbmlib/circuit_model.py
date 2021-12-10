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

  def __init__(self, pqc, symbols, value_layers_init, value_layers, name=None):
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
    self._value_layers_init = value_layers_init

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
    x = self._value_layers_init
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
      initializer: A "tf.keras.initializers.Initializer" which specifies how to
        initialize the values of the parameters in `circuit`.  This argument is
        ignored if `values` is not None.
      name: Identifier for this DirectQuantumCircuit.
    """
    if symbols is None:
      raw_symbols = list(sorted(tfq.util.get_circuit_symbols(pqc)))
      symbols = tf.constant([str(x) for x in raw_symbols], dtype=tf.string)
    if values is None:
      values = initializer(shape=[tf.shape(self._symbols)[0]])
    values = tf.Variable(
        initial_value=values, name=f"{self.name}_pqc_values")
    super().__init__(pqc, symbols, values, [])


class QAIA(QuantumCircuit):
  """Quantum circuit defined by a classical energy and a Hamiltonian."""

  def __init__(self,
               classical_h_terms: List[cirq.PauliSum],
               quantum_h_terms: List[cirq.PauliSum]):
    """Initializes a QAIA."""
    pass
