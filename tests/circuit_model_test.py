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

import cirq
import math
import numpy as np
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib import circuit_model


def _pystr(x):
  return [str(y) for y in x]


class QuantumCircuit(t)

class DirectQuantumCircuitTest(tf.test.TestCase):
  """Tests the DirectQuantumCircuit class."""

  num_qubits = 5
  raw_symbols = [sympy.Symbol("s0"), sympy.Symbol("s1")]
  symbols = tf.constant([str(s) for s in raw_symbols])
  initializer = tf.keras.initializers.RandomUniform(-1.0, 1.0)
  qubits = cirq.GridQubit.rect(1, num_qubits)
  pqc = cirq.Circuit()
  for s in raw_symbols:
    for q in qubits:
      pqc += cirq.X(q)**s
      pqc += cirq.Z(q)**s
  inverse_pqc = pqc**-1
  pqc_tfq = tfq.convert_to_tensor([pqc])
  inverse_pqc_tfq = tfq.convert_to_tensor([inverse_pqc])
  name = "TestOE"

  def test_init(self):
    """Confirms QNN is initialized correctly."""
    actual_qnn = circuit_model.DirectQuantumCircuit(self.pqc, name=self.name)
    self.assertEqual(actual_qnn.name, self.name)
    self.assertAllEqual(actual_qnn.qubits, self.qubits)
    self.assertAllEqual(actual_qnn.symbols, self.symbols)
    self.assertAllEqual(
        tfq.from_tensor(actual_qnn._pqc), tfq.from_tensor(self.pqc_tfq))
    self.assertAllEqual(
        tfq.from_tensor(actual_qnn._inverse_pqc),
        tfq.from_tensor(self.inverse_pqc_tfq),
    )

  def test_alternative_init(self):
    """Confirms that `symbols` and `values` get set correctly."""
    expected_values = self.initializer(shape=[self.num_qubits])
    actual_qnn = circuit_model.DirectQuantumCircuit(
        self.pqc, symbols=self.symbols, values=expected_values)
    self.assertAllEqual(actual_qnn.symbols, self.symbols)
    self.assertAllEqual(actual_qnn.values, expected_values)

  def test_add(self):
    """Confirms two QNNs are added successfully."""
    num_qubits = 5
    qubits = cirq.GridQubit.rect(1, num_qubits)

    pqc_1 = cirq.Circuit()
    symbols_1_str = ["s_1_{n}" for n in range(num_qubits)]
    symbols_1_sympy = [sympy.Symbol(s) for s in symbols_1_str]
    symbols_1 = tf.constant(symbols_1_str)
    for s, q in zip(symbols_1_sympy, qubits):
      pqc_1 += cirq.rx(s)(q)
    values_1 = self.initializer(shape=[num_qubits])

    pqc_2 = cirq.Circuit()
    symbols_2_str = ["s_2_{n}" for n in range(num_qubits)]
    symbols_2_sympy = [sympy.Symbol(s) for s in symbols_2_str]
    symbols_2 = tf.constant(symbols_2_str)
    for s, q in zip(symbols_2_sympy, qubits):
      pqc_2 += cirq.ry(s)(q)
    values_2 = self.initializer(shape=[num_qubits])

    qnn_1 = circuit_model.DirectQuantumCircuit(
        pqc_1, symbols=symbols_1, values=values_1)
    qnn_2 = circuit_model.DirectQuantumCircuit(
        pqc_2, symbols=symbols_2, values=values_2)
    actual_added = qnn_1 + qnn_2

    self.assertAllEqual(
        tfq.from_tensor(actual_added.pqc)[0],
        tfq.from_tensor(tfq.convert_to_tensor([pqc_1 + pqc_2]))[0])
    self.assertAllEqual(actual_added.symbols,
                        tf.concat([symbols_1, symbols_2], 0))
    self.assertAllEqual(actual_added.values, tf.concat([values_1, values_2], 0))

  def test_pow(self):
    """Confirms inverse works correctly."""
    actual_qnn = circuit_model.DirectQuantumCircuit(self.pqc)
    with self.assertRaisesRegex(ValueError, expected_regex="Only the inverse"):
      _ = actual_qnn**-2

    inverse_qnn = actual_qnn**-1
    actual_pqc = tfq.from_tensor(inverse_qnn.pqc)
    expected_pqc = tfq.from_tensor(self.inverse_pqc_tfq)
    actual_inverse_pqc = tfq.from_tensor(inverse_qnn.inverse_pqc)
    expected_inverse_pqc = tfq.from_tensor(self.pqc_tfq)
    self.assertEqual(actual_pqc, expected_pqc)
    self.assertEqual(actual_inverse_pqc, expected_inverse_pqc)
    # Ensure swapping circuits was actually meaningful
    self.assertNotEqual(actual_pqc, actual_inverse_pqc)

  def test_copy(self):
    """Confirms copied QNN has correct attributes."""
    test_qnn = circuit_model.DirectQuantumCircuit(
        self.pqc, initializer=self.initializer, name=self.name)
    actual_qnn_copy = test_qnn.copy()
    self.assertEqual(actual_qnn_copy.name, test_qnn.name)
    self.assertAllClose(actual_qnn_copy.trainable_variables,
                        test_qnn.trainable_variables)
    self.assertAllEqual(actual_qnn_copy.symbols, test_qnn.symbols)
    self.assertAllEqual(
        tfq.from_tensor(actual_qnn_copy._pqc),
        tfq.from_tensor(test_qnn._pqc),
    )
    self.assertAllEqual(
        tfq.from_tensor(actual_qnn_copy._inverse_pqc),
        tfq.from_tensor(test_qnn._inverse_pqc),
    )
    self.assertAllEqual(actual_qnn_copy.qubits, test_qnn.qubits)

  def test_trainable_variables(self):
    test_qnn = circuit_model.DirectQuantumCircuit(self.pqc, name=self.name)

    self.assertAllEqual(test_qnn.values, test_qnn.trainable_variables[0])

    values = tf.random.uniform(tf.shape(test_qnn.trainable_variables[0]))
    test_qnn.trainable_variables = [values]
    self.assertAllEqual(values, test_qnn.trainable_variables[0])

    values = tf.Variable(values)
    test_qnn.trainable_variables = [values]
    self.assertAllEqual(values, test_qnn.trainable_variables[0])


if __name__ == "__main__":
  logging.info("Running circuit_model_test.py ...")
  tf.test.main()
