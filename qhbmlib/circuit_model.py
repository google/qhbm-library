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
import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib import circuit_model_utils


class QuantumCircuit(tf.keras.layers.Layer):
  """Class for representing a quantum circuit."""

  def __init__(self,
               pqc: tf.Tensor,
               qubits: List[cirq.GridQubit],
               symbol_names: tf.Tensor,
               value_layers_inputs: List[Union[tf.Variable, List[tf.Variable]]],
               value_layers: List[List[tf.keras.layers.Layer]],
               name: Union[None, str] = None):
    """Initializes a QuantumCircuit.

    Args:
      pqc: TFQ string representation of a parameterized quantum circuit.
      qubits: The qubits on which `pqc` acts.
      symbol_names: Strings which are used to specify the order in which the
        values in `self.symbol_values` should be placed inside of the circuit.
      value_layers_inputs: Inputs to the `value_layers` argument.
      value_layers: The concatenation of the layers in entry `i` yields a
        trainable map from `value_layers_inputs[i]` to the `i` entry in the list
        of intermediate values.  The list of intermediate values is concatenated
        to yield the values to substitute into the circuit.
      name: Optional name for the model.
    """
    super().__init__(name=name)

    self._pqc = pqc
    self._qubits = sorted(qubits)
    self._symbol_names = symbol_names
    self._value_layers = value_layers
    self._value_layers_inputs = value_layers_inputs

    raw_bit_circuit = circuit_model_utils.bit_circuit(self.qubits)
    bit_symbol_names = list(
        sorted(tfq.util.get_circuit_symbols(raw_bit_circuit)))
    self._bit_symbol_names = tf.constant([str(x) for x in bit_symbol_names])
    self._bit_circuit = tfq.convert_to_tensor([raw_bit_circuit])

  @property
  def qubits(self):
    """Sorted list of the qubits on which this circuit acts."""
    return self._qubits

  @property
  def symbol_names(self):
    """1D tensor of strings which are the free parameters of the circuit."""
    return self._symbol_names

  @property
  def value_layers_inputs(self):
    """List of lists of variables which are inputs to `value_layers`.

    This property (and `value_layers`) is where the caller would access model
    weights to be updated from a secondary model or hypernetwork.
    """
    return self._value_layers_inputs

  @property
  def value_layers(self):
    """List of lists of Keras layers which calculate current parameter values.

    This property (and `value_layers_inputs`) is where the caller would access
    model weights to be updated from a secondary model or hypernetwork.
    """
    return self._value_layers

  @property
  def symbol_values(self):
    """1D `tf.Tensor` of floats specifying the current values of the parameters.

    This should be structured such that `self.symbol_values[i]` is the current
    value of `self.symbol_names[i]` in `self.pqc` and `self.inverse_pqc`.
    """
    # TODO(#123): empty value because concat requires at least two entries.
    intermediate_values = [[]]
    for inputs, layers in zip(self.value_layers_inputs, self.value_layers):
      x = inputs
      for layer in layers:
        x = layer(x)
      intermediate_values.append(x)
    return tf.concat(intermediate_values, 0)

  @property
  def pqc(self):
    """TFQ tensor representation of the parameterized unitary circuit."""
    return self._pqc

  def build(self, input_shape):
    """Builds the layers which calculate the values.

    `input_shape` is unused because it is known to be the shape of
    `self._value_layers_inputs`.
    """
    del input_shape
    for inputs, layers in zip(self.value_layers_inputs, self.value_layers):
      if isinstance(inputs, tf.Variable):
        x = inputs.get_shape()
      else:
        x = [v.get_shape() for v in inputs]
      for layer in layers:
        x = layer.compute_output_shape(x)

  def call(self, inputs):
    """Inputs are bitstrings prepended as initial states to `self.pqc`."""
    num_bitstrings = tf.shape(inputs)[0]
    bit_circuits = tfq.resolve_parameters(
        tf.tile(self._bit_circuit, [num_bitstrings]), self._bit_symbol_names,
        tf.cast(inputs, tf.float32))
    pqcs = tf.tile(self.pqc, [num_bitstrings])
    return tfq.append_circuit(bit_circuits, pqcs)

  def __add__(self, other: "QuantumCircuit"):
    """Returns a QuantumCircuit with `self.pqc` appended to `other.pqc`.

    Note that no new `tf.Variable`s are created, the new QuantumCircuit contains
    the variables in both `self` and `other`.
    """
    if isinstance(other, QuantumCircuit):
      intersection = tf.sets.intersection(
          tf.expand_dims(self.symbol_names, 0),
          tf.expand_dims(other.symbol_names, 0))
      tf.debugging.assert_equal(
          tf.size(intersection.values),
          0,
          message="Circuits to be summed must not have symbols in common.")
      new_pqc = tfq.append_circuit(self.pqc, other.pqc)
      new_qubits = list(set(self.qubits + other.qubits))
      new_symbol_names = tf.concat([self.symbol_names, other.symbol_names], 0)
      new_value_layers_inputs = (
          self.value_layers_inputs + other.value_layers_inputs)
      new_value_layers = self.value_layers + other.value_layers
      new_name = self.name + "_" + other.name
      return QuantumCircuit(new_pqc, new_qubits, new_symbol_names,
                            new_value_layers_inputs, new_value_layers, new_name)
    else:
      raise TypeError

  def __pow__(self, exponent):
    """Returns a QuantumCircuit with inverted `self.pqc`.

    Note that no new `tf.Variable`s are created, the new QuantumCircuit contains
    the same variables as `self`.
    """
    if exponent == -1:
      new_pqc = tfq.from_tensor(self.pqc)[0]**-1
      new_name = self.name + "_inverse"
      return QuantumCircuit(
          tfq.convert_to_tensor([new_pqc]), new_pqc.all_qubits(),
          self.symbol_names, self.value_layers_inputs, self.value_layers,
          new_name)
    else:
      raise ValueError("Only the inverse (exponent == -1) is supported.")


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
        initialize the values of the parameters in `circuit`.
      name: Optional name for the model.
    """
    raw_symbol_names = list(sorted(tfq.util.get_circuit_symbols(pqc)))
    symbol_names = tf.constant([str(x) for x in raw_symbol_names],
                               dtype=tf.string)
    values = [tf.Variable(initializer(shape=[len(raw_symbol_names)]))]
    value_layers = [[]]
    super().__init__(
        tfq.convert_to_tensor([pqc]), pqc.all_qubits(), symbol_names, values,
        value_layers)


class QAIA(QuantumCircuit):
  """Quantum circuit defined by a classical energy and a Hamiltonian.

  This circuit model is intended for use with VQT.
  """

  def __init__(self,
               quantum_h_terms: List[cirq.PauliSum],
               classical_h_terms: List[cirq.PauliSum],
               num_layers: int,
               initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi),
               name=None):
    r"""Initializes a QAIA.

    The ansatz is QAOA-like, with the exponential of the EBM ansatz in place
    of the usual "problem Hamiltonian". Mathematically, it is represented as:

    $$\prod_{\ell=1}^P \left[
        \left(
          \prod_{\bm{b} \in \mathcal{B}_K}
            e^{i\eta_\ell \theta_{\bm{b}}\bm{\hat{Z}}^{\bm{b}}}
        \right)\left(
          \prod_{r\in \mathcal{I}}
            e^{i\gamma_{r\ell}\hat{H}_r}
        \right)
      \right],$$

    where $\hat{H}_r$ is `quantum_h_terms`, $\bm{\hat{Z}}^{\bm{b}}$ is
    `classical_h_terms`, and $P$ is `num_layers`.

    # TODO(#119): add link to new version of the paper.
    For further discussion, see the section "Physics-Inspired Architecture:
    Quantum Adiabatic-Inspired Ansatz" in the QHBM paper.

    Args:
      quantum_h_terms: Non-commuting terms of the target thermal state
        assumed in the QAIA ansatz.
      classical_h_terms: Hamiltonian representation of the EBM chosen to model
        the target thermal state.
      num_layers: How many layers of the ansatz to apply.
      initializer: A `tf.keras.initializers.Initializer` which specifies how to
        initialize the values of the parameters in `circuit`.
      name: Optional name for the model.
    """
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
    symbol_names = tf.constant(flat_symbols)

    value_layers_inputs = [[
        tf.Variable(initializer(shape=[num_layers])),  # true etas
        tf.Variable(initializer(shape=[len(classical_h_terms)])),  # thetas
        tf.Variable(
            initializer(shape=[num_layers, len(quantum_h_terms)])),  # gammas
    ]]

    def embed_params(inputs):
      """Tiles up the variables to properly tie QAIA parameters."""
      exp_etas = tf.expand_dims(inputs[0], 1)
      tiled_thetas = tf.tile(
          tf.expand_dims(inputs[1], 0), [tf.shape(inputs[0])[0], 1])
      classical_params = exp_etas * tiled_thetas
      return tf.reshape(tf.concat([classical_params, inputs[2]], 1), [-1])

    value_layers = [[tf.keras.layers.Lambda(embed_params)]]

    super().__init__(
        tfq.convert_to_tensor([pqc]), pqc.all_qubits(), symbol_names,
        value_layers_inputs, value_layers)
