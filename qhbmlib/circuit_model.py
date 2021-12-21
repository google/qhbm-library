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

import absl
from typing import List, Union

import cirq
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib import circuit_model_utils


class QuantumCircuit(tf.keras.layers.Layer):
  """Class for representing a quantum circuit."""

  def __init__(self,
               pqc: cirq.Circuit,
               symbol_names: tf.Tensor,
               value_layers_inputs: Union[tf.Variable, List[tf.Variable]],
               value_layers: List[tf.keras.layers.Layer],
               name: Union[None, str] = None):
    """Initializes a QuantumCircuit.

    Args:
      pqc: Representation of a parameterized quantum circuit.
      symbol_names: Strings which are used to specify the order in which the
        values in `self.symbol_values` should be placed inside of the circuit.
      value_layers_inputs: Inputs to the `value_layers` argument.
      value_layers: Concatenation of these layers yields trainable map from
        `value_layers_inputs` to the values to substitute into the circuit.
      name: Optional name for the model.
    """
    super().__init__(name=name)

    if set(tfq.util.get_circuit_symbols(pqc)) != {
        s.decode("utf-8") for s in symbol_names.numpy()
    }:
      absl.logging.warning(
          "Argument `pqc` does not have exactly the same parameters as"
          "`symbol_names`, indicating unused `self.symbol_values` outputs.")
    self._qubits = sorted(pqc.all_qubits())
    self._symbol_names = symbol_names
    self._value_layers = value_layers
    self._value_layers_inputs = value_layers_inputs

    self._pqc = tfq.convert_to_tensor([pqc])
    self._inverse_pqc = tfq.convert_to_tensor([pqc**-1])

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
    """Variable or list of variables which are inputs to `value_layers`.

    This property (and `value_layers`) is where the caller would access model
    weights to be updated from a secondary model or hypernetwork.
    """
    return self._value_layers_inputs

  @property
  def value_layers(self):
    """List of Keras layers which calculate the current parameter values.

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
    x = self._value_layers_inputs
    for layer in self._value_layers:
      x = layer(x)
    return x

  @property
  def pqc(self):
    """TFQ tensor representation of the parameterized unitary circuit."""
    return self._pqc

  @property
  def inverse_pqc(self):
    """Inverse of `self.pqc`."""
    return self._inverse_pqc

  def build(self, input_shape):
    """Builds the layers which calculate the values.

    `input_shape` is unused because it is known to be the shape of
    `self._value_layers_inputs`.
    """
    del input_shape
    if isinstance(self.value_layers_inputs, list):
      x = [tf.shape(t) for t in self._value_layers_inputs]
    else:
      x = tf.shape(self.value_layers_inputs)
    for layer in self._value_layers:
      x = layer.compute_output_shape(x)

  def call(self, inputs):
    """Inputs are bitstrings prepended as initial states to `self.pqc`."""
    num_bitstrings = tf.shape(inputs)[0]
    bit_circuits = tfq.resolve_parameters(
        tf.tile(self._bit_circuit, [num_bitstrings]), self._bit_symbol_names,
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
        initialize the values of the parameters in `circuit`.
      name: Optional name for the model.
    """
    raw_symbol_names = list(sorted(tfq.util.get_circuit_symbols(pqc)))
    symbol_names = tf.constant([str(x) for x in raw_symbol_names],
                               dtype=tf.string)
    values = tf.Variable(initializer(shape=[len(raw_symbol_names)]))
    super().__init__(pqc, symbol_names, values, [])


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

    value_layers_inputs = [
        tf.Variable(initializer(shape=[num_layers])),  # true etas
        tf.Variable(initializer(shape=[len(classical_h_terms)])),  # thetas
        tf.Variable(
            initializer(shape=[num_layers, len(quantum_h_terms)])),  # gammas
    ]

    def embed_params(inputs):
      """Tiles up the variables to properly tie QAIA parameters."""
      exp_etas = tf.expand_dims(inputs[0], 1)
      tiled_thetas = tf.tile(
          tf.expand_dims(inputs[1], 0), [tf.shape(inputs[0])[0], 1])
      classical_params = exp_etas * tiled_thetas
      return tf.concat([classical_params, inputs[2]], 1)

    value_layers = [tf.keras.layers.Lambda(embed_params)]

    super().__init__(pqc, symbol_names, value_layers_inputs, value_layers)
