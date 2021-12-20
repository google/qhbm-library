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

import itertools

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
    initial_values = tf.Variable(tf.random.uniform([1, 42]))
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


if __name__ == "__main__":
  logging.info("Running circuit_model_test.py ...")
  tf.test.main()
