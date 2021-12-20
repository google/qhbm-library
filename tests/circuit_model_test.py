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
"""Tests for the circuit_model module."""

import absl
import itertools
import random

import cirq
import numpy as np
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib import circuit_model


class QuantumCircuitTest(tf.test.TestCase):
  """Tests the QuantumCircuit class."""

  def test_init_and_call(self):
    """Tests initialization and correct outputs on call."""
    num_qubits = 5
    raw_symbols = [sympy.Symbol("s0"), sympy.Symbol("s1")]
    expected_symbol_names = tf.constant([str(s) for s in raw_symbols])
    expected_qubits = cirq.GridQubit.rect(1, num_qubits)
    pqc = cirq.Circuit()
    for s in raw_symbols:
      for q in expected_qubits:
        pqc += cirq.X(q)**s
        pqc += cirq.Z(q)**s
    inverse_pqc = pqc**-1
    expected_pqc = tfq.convert_to_tensor([pqc])
    expected_inverse_pqc = tfq.convert_to_tensor([inverse_pqc])
    expected_name = "TestOE"
    init_val_const = tf.random.uniform([1, 42], dtype=tf.float32)
    initial_values = tf.Variable(init_val_const)
    value_layer_0 = tf.keras.layers.Dense(5)
    value_layer_1 = tf.keras.layers.Dense(len(raw_symbols))
    value_layer_2 = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 0))
    expected_symbol_values = value_layer_2(
        value_layer_1(value_layer_0(initial_values)))
    actual_layer = circuit_model.QuantumCircuit(
        pqc, expected_symbol_names, initial_values,
        [value_layer_0, value_layer_1, value_layer_2], expected_name)
    self.assertAllEqual(actual_layer.qubits, expected_qubits)
    self.assertAllEqual(actual_layer.symbol_names, expected_symbol_names)
    self.assertAllEqual(actual_layer.symbol_values, expected_symbol_values)
    self.assertAllEqual(
        tfq.from_tensor(actual_layer.pqc), tfq.from_tensor(expected_pqc))
    self.assertAllEqual(
        tfq.from_tensor(actual_layer.inverse_pqc),
        tfq.from_tensor(expected_inverse_pqc))
    self.assertEqual(actual_layer.name, expected_name)

    # test call
    bitstrings = 2 * list(itertools.product([0, 1], repeat=num_qubits))
    inputs = tf.constant(bitstrings, dtype=tf.int8)
    bit_injectors = []
    for b in bitstrings:
      bit_injectors.append(
          cirq.Circuit(cirq.X(q)**b_i for q, b_i in zip(expected_qubits, b)))
    combined = [b + pqc for b in bit_injectors]
    expected_outputs = tfq.convert_to_tensor(combined)
    actual_outputs = actual_layer(inputs)
    self.assertAllEqual(
        tfq.from_tensor(actual_outputs), tfq.from_tensor(expected_outputs))


class DirectQuantumCircuitTest(tf.test.TestCase):
  """Tests the DirectQuantumCircuit class."""

  def test_init(self):
    """Tests initialization."""
    raw_symbol_names = ["test_symbol", "s_2222", "where"]
    expected_symbol_names = tf.constant(sorted(raw_symbol_names))
    symbols = [sympy.Symbol(s) for s in raw_symbol_names]
    num_qubits = 5
    expected_qubits = cirq.GridQubit.rect(num_qubits, 1)
    expected_pqc = cirq.Circuit()
    for q in expected_qubits:
      for s in symbols:
        random_gate = random.choice([cirq.X, cirq.Y, cirq.Z])
        expected_pqc += random_gate(q)**s
    init_const = random.uniform(-1, 1)
    expected_value_layers_inputs = [init_const] * len(raw_symbol_names)
    actual_qnn = circuit_model.DirectQuantumCircuit(
        expected_pqc, initializer=tf.keras.initializers.Constant(init_const))
    self.assertAllEqual(actual_qnn.qubits, expected_qubits)
    self.assertAllEqual(actual_qnn.symbol_names, expected_symbol_names)
    self.assertAllClose(actual_qnn.value_layers_inputs,
                        expected_value_layers_inputs)
    self.assertEqual(actual_qnn.value_layers, [])
    self.assertAllClose(actual_qnn.symbol_values, expected_value_layers_inputs)
    self.assertAllEqual(
        tfq.from_tensor(actual_qnn.pqc),
        tfq.from_tensor(tfq.convert_to_tensor([expected_pqc])))
    self.assertAllEqual(
        tfq.from_tensor(actual_qnn.inverse_pqc),
        tfq.from_tensor(tfq.convert_to_tensor([expected_pqc**-1])))


class QAIATest(tf.test.TestCase):
  """Tests the DirectQuantumCircuit class."""

  def test_init(self):
    """Tests initialization."""
    num_qubits = 3
    expected_qubits = cirq.GridQubit.rect(1, num_qubits)
    classical_h_terms = [
        cirq.Z(q0) * cirq.Z(q1) for q0, q1 in zip(expected_qubits, expected_qubits[1:])
    ]
    x_terms = cirq.PauliSum()
    y_terms = cirq.PauliSum()
    for q in expected_qubits:
      x_terms += cirq.X(q)
      y_terms += cirq.Y(q)
    quantum_h_terms = [x_terms, y_terms]

    num_layers = 2
    expected_symbol_names = []
    expected_pqc = cirq.Circuit()
    for p in range(num_layers):
      for k, q in enumerate(quantum_h_terms):
        symbol_name = f"gamma_{p}_{k}"
        expected_symbol_names.append(symbol_name)
        expected_pqc += tfq.util.exponential([q], [symbol_name])
      for k, c in enumerate(classical_h_terms):
        symbol_name = f"eta_{p}_{k}"
        expected_symbol_names.append(symbol_name)
        expected_pqc += tfq.util.exponential([c], [symbol_name])

    eta_const = random.uniform(-1, 1)
    theta_const = random.uniform(-1, 1)
    gamma_const = random.uniform(-1, 1)
    expected_value_layers_inputs = [
        tf.Variable([eta_const] * num_layers),
        tf.Variable([theta_const] * len(classical_h_terms)),
        tf.Variable([[gamma_const] * len(quantum_h_terms)] * num_layers)
    ]
    expected_symbol_values = [
      ([eta_const * theta_const] * len(classical_h_terms)) +
      ([gamma_const] * len(quantum_h_terms))] * num_layers
    actual_qnn = circuit_model.QAIA(quantum_h_terms, classical_h_terms,
                                    num_layers)
    self.assertAllEqual(actual_qnn.qubits, expected_qubits)
    self.assertAllEqual(actual_qnn.symbol_names, expected_symbol_names)
    self.assertAllEqual(
        tfq.from_tensor(actual_qnn.pqc),
        tfq.from_tensor(tfq.convert_to_tensor([expected_pqc])))
    self.assertAllEqual(
        tfq.from_tensor(actual_qnn.inverse_pqc),
        tfq.from_tensor(tfq.convert_to_tensor([expected_pqc**-1])))

    actual_qnn.value_layers_inputs[0].assign([eta_const] * num_layers)
    actual_qnn.value_layers_inputs[1].assign([theta_const] *
                                             len(classical_h_terms))
    actual_qnn.value_layers_inputs[2].assign([[gamma_const] *
                                             len(quantum_h_terms)] * num_layers)
    self.assertAllClose(actual_qnn.value_layers_inputs,
                        expected_value_layers_inputs)
    self.assertAllClose(actual_qnn.symbol_values, expected_symbol_values)


if __name__ == "__main__":
  absl.logging.info("Running circuit_model_test.py ...")
  tf.test.main()
