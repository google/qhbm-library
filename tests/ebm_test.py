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
"""Tests for the ebm module."""
import itertools

import cirq
import tensorflow as tf

from qhbmlib import ebm
from qhbmlib import qhbm


class UniqueBitstringsWithCountsTest(tf.test.TestCase):
  """Test unique_with_counts from the qhbm library."""

  def test_identity(self):
    # Case when all entries are unique.
    test_bitstrings = tf.constant([[1], [0]], dtype=tf.int8)
    test_y, test_count = ebm.unique_bitstrings_with_counts(test_bitstrings)
    self.assertAllEqual(test_y, test_bitstrings)
    self.assertAllEqual(test_count, tf.constant([1, 1]))

  def test_short(self):
    # Case when bitstrings are length 1.
    test_bitstrings = tf.constant(
        [
            [0],
            [1],
            [0],
            [1],
            [1],
            [0],
            [1],
            [1],
        ],
        dtype=tf.int8,
    )
    test_y, test_count = ebm.unique_bitstrings_with_counts(test_bitstrings)
    self.assertAllEqual(test_y, tf.constant([[0], [1]]))
    self.assertAllEqual(test_count, tf.constant([3, 5]))

  def test_long(self):
    # Case when bitstrings are of length > 1.
    test_bitstrings = tf.constant(
        [
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 0, 1],
        ],
        dtype=tf.int8,
    )
    test_y, test_count = ebm.unique_bitstrings_with_counts(test_bitstrings)
    self.assertAllEqual(test_y, tf.constant([[1, 0, 1], [1, 1, 1], [0, 1, 1]]))
    self.assertAllEqual(test_count, tf.constant([4, 2, 2]))


class BernoulliTest(tf.test.TestCase):
  """Test the Bernoulli class."""

  num_bits = 5

  def test_init(self):
    """Test that components are initialized correctly."""
    init_const = 1.5
    test_b = ebm.Bernoulli(self.num_bits,
                           tf.keras.initializers.Constant(init_const))
    self.assertAllEqual(test_b.num_bits, self.num_bits)
    self.assertTrue(test_b.has_operator)
    self.assertFalse(test_b.is_analytic)
    self.assertAllEqual(test_b._variables, [init_const] * self.num_bits)

  def test_copy(self):
    """Test that the copy has the same values, but new variables."""
    test_b = ebm.Bernoulli(self.num_bits)
    test_b_copy = test_b.copy()
    self.assertAllEqual(test_b_copy.num_bits, test_b.num_bits)
    self.assertEqual(test_b_copy.has_operator, test_b.has_operator)
    self.assertEqual(test_b_copy.is_analytic, test_b.is_analytic)
    self.assertAllEqual(test_b_copy._variables, test_b._variables)
    self.assertNotEqual(id(test_b_copy._variables), id(test_b._variables))

  def test_energy_bernoulli(self):
    """Test Bernoulli.energy and its derivative.

    For a given bitstring b, the energy is
      $$E_\theta(b) = \sum_i (1-2b_i)\theta_i$$
    Then the derivative of the energy with respect to the thetas vector is
      $$\partial / \partial \theta E_\theta(b) = [(1-2b_i) for b_i in b]$$
    """
    test_b = ebm.Bernoulli(3, name="test")
    test_vars = tf.constant([10.0, -7.0, 1.0], dtype=tf.float32)
    test_b._variables.assign(test_vars)
    test_bitstring = tf.constant([[0, 0, 0]])
    test_spins = 1 - 2 * test_bitstring[0]
    with tf.GradientTape() as tape:
      test_energy = test_b.energy(test_bitstring)[0]
    test_energy_grad = tape.gradient(test_energy, test_b._variables)
    ref_energy = tf.reduce_sum(test_vars)
    self.assertAllClose(test_energy, ref_energy)
    self.assertAllClose(test_energy_grad, test_spins)

    test_vars = tf.constant([1.0, 1.7, -2.8], dtype=tf.float32)
    test_b._variables.assign(test_vars)
    test_bitstring = tf.constant([[1, 0, 1]])
    test_spins = 1 - 2 * test_bitstring[0]
    with tf.GradientTape() as tape:
      test_energy = test_b.energy(test_bitstring)[0]
    test_energy_grad = tape.gradient(test_energy, test_b._variables)
    ref_energy = -test_vars[0] + test_vars[1] - test_vars[2]
    self.assertAllClose(test_energy, ref_energy)
    self.assertAllClose(test_energy_grad, test_spins)

  def test_operator_shards(self):
    """Confirm operators are single qubit Z only."""
    test_b = ebm.Bernoulli(self.num_bits)
    qubits = cirq.GridQubit.rect(1, self.num_bits)
    test_ops = test_b.operator_shards(qubits)
    for i, q in enumerate(qubits):
      self.assertEqual(test_ops[i], cirq.PauliSum.from_pauli_strings(cirq.Z(q)))

  def test_operator_expectation(self):
    """Test combining expectations of operators in energy."""
    # Build Bernoulli
    num_bits = 3
    test_b = ebm.Bernoulli(num_bits)
    qubits = cirq.GridQubit.rect(1, num_bits)
    # Pin at bitstring [1, 0, 1]
    test_b._variables.assign(tf.constant([1000.0, -1000.0, 1000.0]))
    operators = test_b.operator_shards(qubits)

    # True energy
    bitstring = tf.constant([[0, 0, 1]])  # not the pinned bitstring
    ref_energy = test_b.energy(bitstring)[0]

    # Test energy
    circuit = cirq.Circuit(
        [cirq.I(qubits[0]),
         cirq.I(qubits[1]),
         cirq.X(qubits[2])])
    output_state_vector = cirq.Simulator().simulate(circuit).final_state_vector
    op_expectations = []
    qubit_map = {q: i for i, q in enumerate(qubits)}
    for op in operators:
      op_expectations.append(
          op.expectation_from_state_vector(output_state_vector, qubit_map).real)
    test_energy = test_b.operator_expectation(op_expectations)
    self.assertAllClose(test_energy, ref_energy, atol=1e-4)

  def test_sampler_bernoulli(self):
    """Confirm that bitstrings are sampled as expected."""
    test_b = ebm.Bernoulli(1, name="test")

    # For single factor Bernoulli, theta = 0 is 50% chance of 1.
    test_b._variables.assign(tf.constant([0.0]))
    num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
    bitstrings, counts = test_b.sample(num_bitstrings)
    # check that we got both bitstrings
    self.assertTrue(
        tf.reduce_any(tf.equal(tf.constant([0], dtype=tf.int8), bitstrings)))
    self.assertTrue(
        tf.reduce_any(tf.equal(tf.constant([1], dtype=tf.int8), bitstrings)))
    # Check that the fraction is approximately 0.5 (equal counts)
    self.assertAllClose(1.0, counts[0] / counts[1], atol=1e-3)

    # Large value of theta pins the bit.
    test_b._variables.assign(tf.constant([1000.0]))
    num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
    bitstrings, counts = test_b.sample(num_bitstrings)
    # check that we got only one bitstring
    self.assertAllEqual(bitstrings, [[1]])

    # Two bit tests.
    test_b = ebm.Bernoulli(2, name="test")
    test_b._variables.assign(tf.constant([0.0, 0.0]))
    num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
    bitstrings, counts = test_b.sample(num_bitstrings)

    @tf.function
    def check_bitstring_exists(bitstring, bitstring_list):
      print("retracing: check_bitstring_exists")
      return tf.math.reduce_any(
          tf.reduce_all(tf.math.equal(bitstring, bitstring_list), 1))

    self.assertTrue(
        check_bitstring_exists(tf.constant([0, 0], dtype=tf.int8), bitstrings))
    self.assertTrue(
        check_bitstring_exists(tf.constant([0, 1], dtype=tf.int8), bitstrings))
    self.assertTrue(
        check_bitstring_exists(tf.constant([1, 0], dtype=tf.int8), bitstrings))
    self.assertTrue(
        check_bitstring_exists(tf.constant([1, 1], dtype=tf.int8), bitstrings))

    # Check that the fraction is approximately 0.25 (equal counts)
    self.assertAllClose(
        [0.25] * 4,
        [counts[i].numpy() / num_bitstrings for i in range(4)],
        atol=1e-3,
    )
    test_b._variables.assign(tf.constant([-1000.0, 1000.0]))
    num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
    bitstrings, counts = test_b.sample(num_bitstrings)
    # check that we only get 01.
    self.assertFalse(
        check_bitstring_exists(tf.constant([0, 0], dtype=tf.int8), bitstrings))
    self.assertTrue(
        check_bitstring_exists(tf.constant([0, 1], dtype=tf.int8), bitstrings))
    self.assertFalse(
        check_bitstring_exists(tf.constant([1, 0], dtype=tf.int8), bitstrings))
    self.assertFalse(
        check_bitstring_exists(tf.constant([1, 1], dtype=tf.int8), bitstrings))
    self.assertAllEqual(counts, [num_bitstrings])

  def test_log_partition_bernoulli(self):
    # TODO
    pass

  def test_entropy_bernoulli(self):
    """Confirm that the entropy conforms to S(p) = -sum_x p(x)ln(p(x))"""
    test_thetas = tf.constant([-1.5, 0.6, 2.1])
    # logits = 2 * thetas
    test_probs = ebm.logit_to_probability(2 * test_thetas).numpy()
    all_probs = tf.constant([
        (1 - test_probs[0]) * (1 - test_probs[1]) * (1 - test_probs[2]),
        (1 - test_probs[0]) * (1 - test_probs[1]) * (test_probs[2]),
        (1 - test_probs[0]) * (test_probs[1]) * (1 - test_probs[2]),
        (1 - test_probs[0]) * (test_probs[1]) * (test_probs[2]),
        (test_probs[0]) * (1 - test_probs[1]) * (1 - test_probs[2]),
        (test_probs[0]) * (1 - test_probs[1]) * (test_probs[2]),
        (test_probs[0]) * (test_probs[1]) * (1 - test_probs[2]),
        (test_probs[0]) * (test_probs[1]) * (test_probs[2]),
    ])
    # probabilities sum to 1
    self.assertAllClose(1.0, tf.reduce_sum(all_probs))
    expected_entropy = -1.0 * tf.reduce_sum(all_probs * tf.math.log(all_probs))
    test_b = ebm.Bernoulli(3, name="test")
    test_b._variables.assign(test_thetas)
    test_entropy = test_b.entropy()
    self.assertAllClose(expected_entropy, test_entropy)


class KOBETest(tf.test.TestCase):
  """Test the ebm.KOBE energy function."""

  # TODO: test more orders, and compatibility with Bernoulli at order==1
  order = 2

  def test_init(self):
    """Confirm internal values are set correctly."""
    num_bits = 3
    init_const = -3.1
    test_k = ebm.KOBE(num_bits, self.order,
                      tf.keras.initializers.Constant(init_const))
    self.assertEqual(test_k.num_bits, num_bits)
    self.assertEqual(test_k.order, self.order)
    self.assertTrue(test_k.has_operator)

    ref_indices = [[0], [1], [2], [0, 1], [0, 2], [1, 2]]
    self.assertAllClose(test_k._indices, ref_indices)
    self.assertAllClose(test_k._variables, [init_const] * sum(
        len(list(itertools.combinations(range(num_bits), i)))
        for i in range(1, self.order + 1)))

  def test_copy(self):
    """Test that the copy has the same values, but new variables."""
    num_bits = 3
    init_const = -3.1
    test_k = ebm.KOBE(num_bits, self.order,
                      tf.keras.initializers.Constant(init_const))
    test_k_copy = test_k.copy()
    self.assertEqual(test_k_copy.num_bits, test_k.num_bits)
    self.assertEqual(test_k_copy.order, test_k.order)
    self.assertTrue(test_k_copy.has_operator, test_k.has_operator)
    self.assertAllClose(test_k_copy._indices, test_k._indices)
    self.assertAllClose(test_k_copy._variables, test_k._variables)
    self.assertNotEqual(id(test_k_copy._variables), id(test_k._variables))

  def test_energy(self):
    """Test every energy on two bits."""
    n_nodes = 2
    test_thetas = tf.Variable([1.5, 2.7, -4.0])
    expected_energies = tf.constant([0.2, 2.8, 5.2, -8.2])
    test_k = ebm.KOBE(n_nodes, self.order)
    test_k._variables.assign(test_thetas)
    all_strings = tf.constant(
        list(itertools.product([0, 1], repeat=n_nodes)), dtype=tf.int8)
    test_energies = test_k.energy(all_strings)
    self.assertAllClose(expected_energies, test_energies)

  def test_operator_shards(self):
    """Confirm correct operators for a simple Boltzmann."""
    num_bits = 3
    test_k = ebm.KOBE(num_bits, self.order)
    qubits = cirq.GridQubit.rect(1, num_bits)
    test_ops = test_k.operator_shards(qubits)
    ref_ops = [
        cirq.Z(qubits[0]),
        cirq.Z(qubits[1]),
        cirq.Z(qubits[2]),
        cirq.Z(qubits[0]) * cirq.Z(qubits[1]),
        cirq.Z(qubits[0]) * cirq.Z(qubits[2]),
        cirq.Z(qubits[1]) * cirq.Z(qubits[2])
    ]
    for t_op, r_op in zip(test_ops, ref_ops):
      self.assertEqual(t_op, cirq.PauliSum.from_pauli_strings(r_op))

  def test_operator_expectation(self):
    """Confirm the expectations combine to the correct total energy."""
    # Build simple Boltzmann
    num_bits = 3
    test_b = ebm.KOBE(num_bits, 2)
    qubits = cirq.GridQubit.rect(1, num_bits)

    # Pin at bitstring [1, 0, 1]
    test_b._variables.assign(tf.constant([100.0, -200.0, 300.0, 10, -20, 30]))
    operators = test_b.operator_shards(qubits)

    # True energy
    bitstring = tf.constant([[0, 0, 1]])  # not the pinned bitstring
    ref_energy = test_b.energy(bitstring)[0]

    # Test energy
    circuit = cirq.Circuit(
        [cirq.I(qubits[0]),
         cirq.I(qubits[1]),
         cirq.X(qubits[2])])
    output_state_vector = cirq.Simulator().simulate(circuit).final_state_vector
    op_expectations = []
    qubit_map = {q: i for i, q in enumerate(qubits)}
    for op in operators:
      op_expectations.append(
          op.expectation_from_state_vector(output_state_vector, qubit_map).real)
    test_energy = test_b.operator_expectation(op_expectations)
    self.assertAllClose(test_energy, ref_energy, atol=1e-4)

  def test_sampler(self):
    """Confirm bitstrings are sampled as expected."""
    # Single bit test.
    test_k = ebm.EBM(ebm.KOBE(1, self.order), None, is_analytic=True)
    # For single factor Bernoulli, theta=0 is 50% chance of 1.
    test_thetas = tf.constant([0.0])
    test_k._energy_function._variables.assign(test_thetas)
    num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
    bitstrings, counts = test_k.sample(num_bitstrings)
    # check that we got both bitstrings
    self.assertTrue(
        tf.reduce_any(tf.equal(tf.constant([0], dtype=tf.int8), bitstrings)))
    self.assertTrue(
        tf.reduce_any(tf.equal(tf.constant([1], dtype=tf.int8), bitstrings)))
    # Check that the fraction is approximately 0.5 (equal counts)
    self.assertAllClose(1.0, counts[0] / counts[1], atol=1e-3)
    # Large energy penalty pins the bit.
    test_thetas = tf.constant([100.0])
    test_k._energy_function._variables.assign(test_thetas)
    num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
    bitstrings, _ = test_k.sample(num_bitstrings)
    # check that we got only one bitstring
    self.assertFalse(
        tf.reduce_any(tf.equal(tf.constant([0], dtype=tf.int8), bitstrings)))
    self.assertTrue(
        tf.reduce_any(tf.equal(tf.constant([1], dtype=tf.int8), bitstrings)))

    # Three bit tests.
    # First a uniform sampling test.
    test_k = ebm.EBM(
        ebm.KOBE(3, self.order, tf.keras.initializers.Constant(0.0)),
        None,
        is_analytic=True)
    num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
    bitstrings, counts = test_k.sample(num_bitstrings)

    @tf.function
    def check_bitstring_exists(bitstring, bitstring_list):
      print("retracing: check_bitstring_exists")
      return tf.math.reduce_any(
          tf.reduce_all(tf.math.equal(bitstring, bitstring_list), 1))

    self.assertTrue(
        check_bitstring_exists(
            tf.constant([0, 0, 0], dtype=tf.int8), bitstrings))
    self.assertTrue(
        check_bitstring_exists(
            tf.constant([0, 0, 1], dtype=tf.int8), bitstrings))
    self.assertTrue(
        check_bitstring_exists(
            tf.constant([0, 1, 0], dtype=tf.int8), bitstrings))
    self.assertTrue(
        check_bitstring_exists(
            tf.constant([0, 1, 1], dtype=tf.int8), bitstrings))
    self.assertTrue(
        check_bitstring_exists(
            tf.constant([1, 0, 0], dtype=tf.int8), bitstrings))
    self.assertTrue(
        check_bitstring_exists(
            tf.constant([1, 0, 1], dtype=tf.int8), bitstrings))
    self.assertTrue(
        check_bitstring_exists(
            tf.constant([1, 1, 0], dtype=tf.int8), bitstrings))
    self.assertTrue(
        check_bitstring_exists(
            tf.constant([1, 1, 1], dtype=tf.int8), bitstrings))
    # Check that the fraction is approximately 0.125 (equal counts)
    self.assertAllClose(
        [0.125] * 8,
        [counts[i].numpy() / num_bitstrings for i in range(8)],
        atol=1e-3,
    )
    # Confirm correlated spins.
    test_thetas = tf.constant([100.0, 0.0, 0.0, -100.0, 0.0, 100.0])
    test_k._energy_function._variables.assign(test_thetas)
    num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
    bitstrings, _ = test_k.sample(num_bitstrings)
    # Confirm we only get the 110 bitstring.
    self.assertFalse(
        check_bitstring_exists(
            tf.constant([0, 0, 0], dtype=tf.int8), bitstrings))
    self.assertFalse(
        check_bitstring_exists(
            tf.constant([0, 0, 1], dtype=tf.int8), bitstrings))
    self.assertFalse(
        check_bitstring_exists(
            tf.constant([0, 1, 0], dtype=tf.int8), bitstrings))
    self.assertFalse(
        check_bitstring_exists(
            tf.constant([0, 1, 1], dtype=tf.int8), bitstrings))
    self.assertFalse(
        check_bitstring_exists(
            tf.constant([1, 0, 0], dtype=tf.int8), bitstrings))
    self.assertFalse(
        check_bitstring_exists(
            tf.constant([1, 0, 1], dtype=tf.int8), bitstrings))
    self.assertTrue(
        check_bitstring_exists(
            tf.constant([1, 1, 0], dtype=tf.int8), bitstrings))
    self.assertFalse(
        check_bitstring_exists(
            tf.constant([1, 1, 1], dtype=tf.int8), bitstrings))

  def test_log_partition_boltzmann(self):
    """Confirm correct value of the log partition function."""
    test_thetas = tf.Variable([1.5, 2.7, -4.0])
    expected_log_partition = tf.math.log(tf.constant(3641.8353))
    pre_k = ebm.KOBE(2, 2)
    pre_k._variables.assign(test_thetas)
    test_k = ebm.EBM(pre_k, None, is_analytic=True)
    test_log_partition = test_k.log_partition_function()
    self.assertAllClose(expected_log_partition, test_log_partition)

  def test_entropy_boltzmann(self):
    """Confirm correct value of the entropy function."""
    test_thetas = tf.Variable([1.5, 2.7, -4.0])
    expected_entropy = tf.constant(0.00233551808)
    pre_k = ebm.KOBE(2, 2)
    pre_k._variables.assign(test_thetas)
    test_k = ebm.EBM(pre_k, None, is_analytic=True)
    test_entropy = test_k.entropy()
    self.assertAllClose(expected_entropy, test_entropy)


if __name__ == "__main__":
  print("Running ebm_test.py ...")
  tf.test.main()
