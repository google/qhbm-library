# Copyright 2021 The QHBM Library Authors.
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
"""Tests for the orthogonal_ensemble module."""

import cirq
import sympy
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_quantum as tfq

from qhbmlib import orthogonal_ensemble
from tests import test_util

# Global tolerance, set for float32.
ATOL = 1e-5


class BuildBitCircuitTest(tf.test.TestCase):
  """Test build_bit_circuit from the qhbm library."""

  def test_build_bit_circuit(self):
    """Confirm correct bit injector circuit creation."""
    my_qubits = [
        cirq.GridQubit(0, 2),
        cirq.GridQubit(1, 4),
        cirq.GridQubit(2, 2)
    ]
    identifier = "build_bit_test"
    test_circuit, test_symbols = orthogonal_ensemble.build_bit_circuit(
        my_qubits, identifier)
    expected_symbols = list(
        sympy.symbols(
            "_bit_build_bit_test_0 _bit_build_bit_test_1 _bit_build_bit_test_2")
    )
    expected_circuit = cirq.Circuit(
        [cirq.X(q)**s for q, s in zip(my_qubits, expected_symbols)])
    self.assertAllEqual(test_symbols, expected_symbols)
    self.assertEqual(test_circuit, expected_circuit)


class InputChecksTest(tf.test.TestCase):
  """Tests all the input checking functions used for QHBMs."""

  def test_upgrade_initial_values(self):
    """Confirms lists of values are properly upgraded to variables."""
    # Test allowed inputs.
    true_list = [-5.1, 2.8, -3.4, 4.8]
    true_tensor = tf.constant(true_list, dtype=tf.float32)
    true_variable = tf.Variable(true_tensor)
    self.assertAllClose(
        orthogonal_ensemble.upgrade_initial_values(true_list),
        true_variable,
        atol=ATOL)
    self.assertAllClose(
        orthogonal_ensemble.upgrade_initial_values(true_tensor),
        true_variable,
        atol=ATOL)
    self.assertAllClose(
        orthogonal_ensemble.upgrade_initial_values(true_variable),
        true_variable,
        atol=ATOL)
    # Check for bad inputs.
    with self.assertRaisesRegex(TypeError, "numeric type"):
      _ = orthogonal_ensemble.upgrade_initial_values("junk")
    with self.assertRaisesRegex(ValueError, "must be 1D"):
      _ = orthogonal_ensemble.upgrade_initial_values([[5.2]])

  def test_upgrade_symbols(self):
    """Confirms symbols are upgraded appropriately."""
    true_symbol_names = ["test1", "a", "my_symbol", "MySymbol2"]
    values = tf.constant([0 for _ in true_symbol_names])
    true_symbols = [sympy.Symbol(s) for s in true_symbol_names]
    true_symbols_t = tf.constant(true_symbol_names, dtype=tf.string)
    self.assertAllEqual(
        true_symbols_t,
        orthogonal_ensemble.upgrade_symbols(true_symbols, values))
    # Test bad inputs.
    with self.assertRaisesRegex(TypeError, "must be `sympy.Symbol`"):
      _ = orthogonal_ensemble.upgrade_symbols(true_symbols[:-1] + ["bad"],
                                              values)
    with self.assertRaisesRegex(ValueError, "must be unique"):
      _ = orthogonal_ensemble.upgrade_symbols(
          true_symbols[:-1] + true_symbols[:1], values)
    with self.assertRaisesRegex(ValueError, "symbol for every value"):
      _ = orthogonal_ensemble.upgrade_symbols(true_symbols, values[:-1])
    with self.assertRaisesRegex(TypeError, "must be an iterable"):
      _ = orthogonal_ensemble.upgrade_symbols(5, values)

  def test_upgrade_circuit(self):
    """Confirms circuits are upgraded appropriately."""
    qubits = cirq.GridQubit.rect(1, 5)
    true_symbol_names = ["a", "b"]
    true_symbols = [sympy.Symbol(s) for s in true_symbol_names]
    true_symbols_t = tf.constant(true_symbol_names, dtype=tf.string)
    true_circuit = cirq.Circuit()
    for q in qubits:
      for s in true_symbols:
        true_circuit += cirq.X(q)**s
    true_circuit_t = tfq.convert_to_tensor([true_circuit])
    self.assertEqual(
        tfq.from_tensor(true_circuit_t),
        tfq.from_tensor(
            orthogonal_ensemble.upgrade_circuit(true_circuit, true_symbols_t)),
    )
    # Test bad inputs.
    with self.assertRaisesRegex(TypeError, "must be a `cirq.Circuit`"):
      _ = orthogonal_ensemble.upgrade_circuit("junk", true_symbols_t)
    with self.assertRaisesRegex(TypeError, "must be a `tf.Tensor`"):
      _ = orthogonal_ensemble.upgrade_circuit(true_circuit, true_symbol_names)
    with self.assertRaisesRegex(TypeError, "dtype `tf.string`"):
      _ = orthogonal_ensemble.upgrade_circuit(true_circuit, tf.constant([5.5]))
    with self.assertRaisesRegex(ValueError, "must contain"):
      _ = orthogonal_ensemble.upgrade_circuit(true_circuit,
                                              tf.constant(["a", "junk"]))
    with self.assertRaisesRegex(ValueError, "Empty circuit"):
      _ = orthogonal_ensemble.upgrade_circuit(cirq.Circuit(),
                                              tf.constant([], dtype=tf.string))


class OrthogonalEnsembleTest(tf.test.TestCase):
  """Tests the OrthogonalEnsemble class."""

  num_bits = 5
  raw_phis_symbols = [sympy.Symbol("s0"), sympy.Symbol("s1")]
  phis_symbols = tf.constant([str(s) for s in raw_phis_symbols])
  initial_phis = tf.random.uniform([len(phis_symbols)], minval=-1.0)
  raw_qubits = cirq.GridQubit.rect(1, num_bits)
  u = cirq.Circuit()
  for s in raw_phis_symbols:
    for q in raw_qubits:
      u += cirq.X(q)**s
  name = "TestOE"
  raw_bit_circuit, raw_bit_symbols = orthogonal_ensemble.build_bit_circuit(
      raw_qubits, name)
  bit_symbols = tf.constant([str(s) for s in raw_bit_symbols])
  bit_circuit = tfq.convert_to_tensor([raw_bit_circuit])

  def test_init(self):
    """Confirms OrthogonalEnsemble is initialized correctly."""
    test_oe = orthogonal_ensemble.OrthogonalEnsemble(
        self.u,
        self.raw_phis_symbols,
        self.initial_phis,
        self.name,
    )
    self.assertEqual(self.name, test_oe.name)
    self.assertAllClose(self.initial_phis, test_oe.phis)
    self.assertAllEqual(self.phis_symbols, test_oe.phis_symbols)
    self.assertAllEqual(
        tfq.from_tensor(tfq.convert_to_tensor([self.u])),
        tfq.from_tensor(test_oe.u),
    )
    self.assertAllEqual(
        tfq.from_tensor(tfq.convert_to_tensor([self.u**-1])),
        tfq.from_tensor(test_oe.u_dagger),
    )
    self.assertAllEqual(self.raw_qubits, test_oe.raw_qubits)
    self.assertAllEqual(self.bit_symbols, test_oe.bit_symbols)
    self.assertEqual(tfq.from_tensor(self.bit_circuit), tfq.from_tensor(test_oe.bit_circuit))


if __name__ == "__main__":
  print("Running orthogonal_ensemble_test.py ...")
  tf.test.main()
