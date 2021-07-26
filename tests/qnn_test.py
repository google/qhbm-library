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
"""Tests for the qnn module."""

import itertools
from absl import logging

import cirq
import sympy
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_quantum as tfq

from qhbmlib import qnn
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
    test_circuit, test_symbols = qnn.build_bit_circuit(my_qubits, identifier)
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
        qnn.upgrade_initial_values(true_list), true_variable, atol=ATOL)
    self.assertAllClose(
        qnn.upgrade_initial_values(true_tensor), true_variable, atol=ATOL)
    self.assertAllClose(
        qnn.upgrade_initial_values(true_variable), true_variable, atol=ATOL)
    # Check for bad inputs.
    with self.assertRaisesRegex(TypeError, "numeric type"):
      _ = qnn.upgrade_initial_values("junk")
    with self.assertRaisesRegex(ValueError, "must be 1D"):
      _ = qnn.upgrade_initial_values([[5.2]])

  def test_upgrade_symbols(self):
    """Confirms symbols are upgraded appropriately."""
    true_symbol_names = ["test1", "a", "my_symbol", "MySymbol2"]
    values = tf.constant([0 for _ in true_symbol_names])
    true_symbols = [sympy.Symbol(s) for s in true_symbol_names]
    true_symbols_t = tf.constant(true_symbol_names, dtype=tf.string)
    self.assertAllEqual(true_symbols_t,
                        qnn.upgrade_symbols(true_symbols, values))
    # Test bad inputs.
    with self.assertRaisesRegex(TypeError, "must be `sympy.Symbol`"):
      _ = qnn.upgrade_symbols(true_symbols[:-1] + ["bad"], values)
    with self.assertRaisesRegex(ValueError, "must be unique"):
      _ = qnn.upgrade_symbols(true_symbols[:-1] + true_symbols[:1], values)
    with self.assertRaisesRegex(ValueError, "symbol for every value"):
      _ = qnn.upgrade_symbols(true_symbols, values[:-1])
    with self.assertRaisesRegex(TypeError, "must be an iterable"):
      _ = qnn.upgrade_symbols(5, values)

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
        tfq.from_tensor(qnn.upgrade_circuit(true_circuit, true_symbols_t)),
    )
    # Test bad inputs.
    with self.assertRaisesRegex(TypeError, "must be a `cirq.Circuit`"):
      _ = qnn.upgrade_circuit("junk", true_symbols_t)
    with self.assertRaisesRegex(TypeError, "must be a `tf.Tensor`"):
      _ = qnn.upgrade_circuit(true_circuit, true_symbol_names)
    with self.assertRaisesRegex(TypeError, "dtype `tf.string`"):
      _ = qnn.upgrade_circuit(true_circuit, tf.constant([5.5]))
    with self.assertRaisesRegex(ValueError, "must contain"):
      _ = qnn.upgrade_circuit(true_circuit, tf.constant(["a", "junk"]))
    with self.assertRaisesRegex(ValueError, "Empty circuit"):
      _ = qnn.upgrade_circuit(cirq.Circuit(), tf.constant([], dtype=tf.string))


class QNNTest(tf.test.TestCase):
  """Tests the QNN class."""

  num_bits = 5
  raw_phis_symbols = [sympy.Symbol("s0"), sympy.Symbol("s1")]
  phis_symbols = tf.constant([str(s) for s in raw_phis_symbols])
  initial_phis = tf.random.uniform([len(phis_symbols)], minval=-1.0)
  raw_qubits = cirq.GridQubit.rect(1, num_bits)
  u = cirq.Circuit()
  for s in raw_phis_symbols:
    for q in raw_qubits:
      u += cirq.X(q)**s
      u += cirq.Z(q)**s
  u_dagger = u**-1
  u_tfq = tfq.convert_to_tensor([u])
  u_dagger_tfq = tfq.convert_to_tensor([u_dagger])
  name = "TestOE"
  raw_bit_circuit, raw_bit_symbols = qnn.build_bit_circuit(raw_qubits, name)
  bit_symbols = tf.constant([str(s) for s in raw_bit_symbols])
  bit_circuit = tfq.convert_to_tensor([raw_bit_circuit])

  def test_init(self):
    """Confirms QNN is initialized correctly."""
    test_qnn = qnn.QNN(
        self.u,
        self.raw_phis_symbols,
        self.initial_phis,
        self.name,
    )
    self.assertEqual(self.name, test_qnn.name)
    self.assertAllClose(self.initial_phis, test_qnn.phis)
    self.assertAllEqual(self.phis_symbols, test_qnn.phis_symbols)
    self.assertAllEqual(
        tfq.from_tensor(self.u_tfq),
        tfq.from_tensor(test_qnn.u),
    )
    self.assertAllEqual(
        tfq.from_tensor(self.u_dagger_tfq),
        tfq.from_tensor(test_qnn.u_dagger),
    )
    self.assertAllEqual(self.raw_qubits, test_qnn.raw_qubits)
    self.assertAllEqual(self.bit_symbols, test_qnn.bit_symbols)
    self.assertEqual(
        tfq.from_tensor(self.bit_circuit),
        tfq.from_tensor(test_qnn.bit_circuit))

    self.assertEqual(
        tfq.from_tensor(test_qnn.resolved_u),
        tfq.from_tensor(
            tfq.resolve_parameters(self.u_tfq, self.phis_symbols,
                                   tf.expand_dims(self.initial_phis, 0))))
    self.assertEqual(
        tfq.from_tensor(test_qnn.resolved_u_dagger),
        tfq.from_tensor(
            tfq.resolve_parameters(self.u_dagger_tfq, self.phis_symbols,
                                   tf.expand_dims(self.initial_phis, 0))))

  def test_copy(self):
    """Confirms copied QNN has correct attributes."""
    test_qnn = qnn.QNN(
        self.u,
        self.raw_phis_symbols,
        self.initial_phis,
        self.name,
    )
    test_qnn_copy = test_qnn.copy()
    self.assertEqual(test_qnn_copy.name, test_qnn.name)
    self.assertAllClose(test_qnn_copy.phis, test_qnn.phis)
    self.assertAllEqual(test_qnn_copy.phis_symbols, test_qnn.phis_symbols)
    self.assertAllEqual(
        tfq.from_tensor(test_qnn_copy.u),
        tfq.from_tensor(test_qnn.u),
    )
    self.assertAllEqual(
        tfq.from_tensor(test_qnn_copy.u_dagger),
        tfq.from_tensor(test_qnn.u_dagger),
    )
    self.assertAllEqual(test_qnn_copy.raw_qubits, test_qnn.raw_qubits)
    self.assertAllEqual(test_qnn_copy.bit_symbols, test_qnn.bit_symbols)
    self.assertEqual(
        tfq.from_tensor(test_qnn_copy.bit_circuit),
        tfq.from_tensor(test_qnn.bit_circuit))
    self.assertEqual(
        tfq.from_tensor(test_qnn_copy.resolved_u),
        tfq.from_tensor(test_qnn.resolved_u))
    self.assertEqual(
        tfq.from_tensor(test_qnn_copy.resolved_u_dagger),
        tfq.from_tensor(test_qnn.resolved_u_dagger))

  def test_circuits(self):
    """Confirms bitstring injectors are prepended to u."""
    bitstrings = 2 * list(itertools.product([0, 1], repeat=self.num_bits))
    test_qnn = qnn.QNN(
        self.u,
        self.raw_phis_symbols,
        self.initial_phis,
        self.name,
    )
    test_circuits = test_qnn.circuits(tf.constant(bitstrings, dtype=tf.int8))
    test_circuits_deser = tfq.from_tensor(test_circuits)

    resolved_u = tfq.from_tensor(test_qnn.resolved_u)[0]
    bit_injectors = []
    for b in bitstrings:
      bit_injectors.append(
          cirq.Circuit(cirq.X(q)**b_i for q, b_i in zip(self.raw_qubits, b)))
    combined = [b + resolved_u for b in bit_injectors]

    for expected, test in zip(combined, test_circuits_deser):
      self.assertTrue(cirq.approx_eq(expected, test))

  def test_circuits(self):
    """Confirms bitstring injectors are prepended to u."""
    bitstrings = 2 * list(itertools.product([0, 1], repeat=self.num_bits))
    test_qnn = qnn.QNN(
        self.u,
        self.raw_phis_symbols,
        self.initial_phis,
        self.name,
    )
    test_circuits = test_qnn.circuits(tf.constant(bitstrings, dtype=tf.int8))
    test_circuits_deser = tfq.from_tensor(test_circuits)

    resolved_u = tfq.from_tensor(test_qnn.resolved_u)[0]
    bit_injectors = []
    for b in bitstrings:
      bit_injectors.append(
          cirq.Circuit(cirq.X(q)**b_i for q, b_i in zip(self.raw_qubits, b)))
    combined = [b + resolved_u for b in bit_injectors]

    for expected, test in zip(combined, test_circuits_deser):
      self.assertTrue(cirq.approx_eq(expected, test))

  def test_sample_basic(self):
    """Confirms correct sampling from identity, bit flip, and GHZ QNNs."""
    bitstrings = tf.constant(
        list(itertools.product([0, 1], repeat=self.num_bits)), dtype=tf.int8)
    counts = tf.random.uniform([tf.shape(bitstrings)[0]],
                               minval=10,
                               maxval=100,
                               dtype=tf.int32)

    ident_qnn = qnn.QNN(
        cirq.Circuit(cirq.I(q) for q in self.raw_qubits), [], [], "identity")
    test_samples = ident_qnn.sample(bitstrings, counts)
    for i, (b, c) in enumerate(zip(bitstrings, counts)):
      self.assertEqual(tf.shape(test_samples[i].to_tensor())[0], c)
      for j in range(c):
        self.assertAllEqual(test_samples[i][j], b)

    flip_qnn = qnn.QNN(
        cirq.Circuit(cirq.X(q) for q in self.raw_qubits), [], [], "flip")
    test_samples = flip_qnn.sample(bitstrings, counts)
    for i, (b, c) in enumerate(zip(bitstrings, counts)):
      self.assertEqual(tf.shape(test_samples[i].to_tensor())[0], c)
      for j in range(c):
        self.assertAllEqual(
            test_samples[i][j],
            tf.cast(tf.math.logical_not(tf.cast(b, tf.bool)), tf.int8))

    ghz_param = sympy.Symbol("ghz")
    ghz_circuit = cirq.Circuit(cirq.X(
        self.raw_qubits[0])**ghz_param) + cirq.Circuit(
            cirq.CNOT(q0, q1)
            for q0, q1 in zip(self.raw_qubits, self.raw_qubits[1:]))
    ghz_qnn = qnn.QNN(ghz_circuit, [ghz_param], [0.5], "ghz")
    test_samples = ghz_qnn.sample(
        tf.expand_dims(tf.constant([0] * self.num_bits, dtype=tf.int8), 0),
        tf.expand_dims(counts[0], 0))[0].to_tensor()
    # Both |0...0> and |1...1> should be among the measured bitstrings
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0] * self.num_bits, dtype=tf.int8), test_samples))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1] * self.num_bits, dtype=tf.int8), test_samples))

  def test_pulled_back_circuits(self):
    """Confirms the pulled back circuits correct for a variety of inputs."""
    num_data_states = 100
    data_states, _ = tfq.util.random_circuit_resolver_batch(
        self.raw_qubits, num_data_states)
    data_states_t = tfq.convert_to_tensor(data_states)
    test_qnn = qnn.QNN(
        self.u,
        self.raw_phis_symbols,
        self.initial_phis,
        self.name,
    )
    test_circuits = test_qnn.pulled_back_circuits(data_states_t)
    test_circuits_deser = tfq.from_tensor(test_circuits)

    resolved_u_dagger = tfq.from_tensor(test_qnn.resolved_u_dagger)[0]
    combined = tfq.from_tensor(
        tfq.convert_to_tensor([d + resolved_u_dagger for d in data_states]))
    for expected, test in zip(combined, test_circuits_deser):
      self.assertTrue(cirq.approx_eq(expected, test))

  def test_pulled_back_sample_basic(self):
    """Confirms correct pulled back sampling from GHZ QNN.

    The state preparation circuit for GHZ is not equal to its inverse,
    so it tests that the dagger is taken correctly before appending.
    """
    ghz_param = sympy.Symbol("ghz")
    ghz_circuit = cirq.Circuit(cirq.X(
        self.raw_qubits[0])**ghz_param) + cirq.Circuit(
            cirq.CNOT(q0, q1)
            for q0, q1 in zip(self.raw_qubits, self.raw_qubits[1:]))
    ghz_qnn = qnn.QNN(ghz_circuit, [ghz_param], [0.5], "ghz")
    flip_circuits = [cirq.Circuit(), cirq.Circuit(cirq.X(self.raw_qubits[0]))]
    flip_circuits_t = tfq.convert_to_tensor(flip_circuits)
    counts = tf.random.uniform([len(flip_circuits)],
                               minval=10,
                               maxval=100,
                               dtype=tf.int32)

    test_samples = ghz_qnn.pulled_back_sample(flip_circuits_t, counts)
    # The first circuit leaves only the Hadamard to superpose the first qubit
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0] * self.num_bits, dtype=tf.int8),
            test_samples[0].to_tensor()))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1] + [0] * (self.num_bits - 1), dtype=tf.int8),
            test_samples[0].to_tensor()))
    # The second circuit causes an additional bit flip
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0] + [1] + [0] * (self.num_bits - 2), dtype=tf.int8),
            test_samples[1].to_tensor()))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1, 1] + [0] * (self.num_bits - 2), dtype=tf.int8),
            test_samples[1].to_tensor()))


if __name__ == "__main__":
  logging.info("Running qnn_test.py ...")
  tf.test.main()
