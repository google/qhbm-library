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
"""Tests for the energy_model module."""

import itertools

import cirq
import tensorflow as tf

from qhbmlib import energy_model


class BernoulliTest(tf.test.TestCase):
  """Test the Bernoulli class."""

  def test_init(self):
    """Test that components are initialized correctly."""
    bits = [i for in in range(5)]
    init_const = 1.5
    test_b = energy_model.Bernoulli(bits,
                                    tf.keras.initializers.Constant(init_const))
    self.assertAllEqual(test_b.num_bits, len(bits))
    self.assertAllEqual(test_b.bits, bits)
    self.assertAllEqual(test_b.kernel, [init_const] * self.num_bits)

  # def test_copy(self):
  #   """Test that the copy has the same values, but new variables."""
  #   test_b = ebm.Bernoulli(self.num_bits)
  #   test_b_copy = test_b.copy()
  #   self.assertAllEqual(test_b_copy.num_bits, test_b.num_bits)
  #   self.assertEqual(test_b_copy.has_operator, test_b.has_operator)
  #   self.assertEqual(test_b_copy.is_analytic, test_b.is_analytic)
  #   self.assertAllEqual(test_b_copy.kernel, test_b.kernel)
  #   self.assertNotEqual(id(test_b_copy.kernel), id(test_b.kernel))

  # def test_energy_bernoulli(self):
  #   """Test Bernoulli.energy and its derivative.

  #   For a given bitstring b, the energy is
  #     $$E_\theta(b) = \sum_i (1-2b_i)\theta_i$$
  #   Then the derivative of the energy with respect to the thetas vector is
  #     $$\partial / \partial \theta E_\theta(b) = [(1-2b_i) for b_i in b]$$
  #   """
  #   test_b = ebm.Bernoulli(3, name="test")
  #   test_vars = tf.constant([10.0, -7.0, 1.0], dtype=tf.float32)
  #   test_b.kernel.assign(test_vars)
  #   test_bitstring = tf.constant([[0, 0, 0]])
  #   test_spins = 1 - 2 * test_bitstring[0]
  #   with tf.GradientTape() as tape:
  #     test_energy = test_b.energy(test_bitstring)[0]
  #   test_energy_grad = tape.gradient(test_energy, test_b.kernel)
  #   ref_energy = tf.reduce_sum(test_vars)
  #   self.assertAllClose(test_energy, ref_energy)
  #   self.assertAllClose(test_energy_grad, test_spins)

  #   test_vars = tf.constant([1.0, 1.7, -2.8], dtype=tf.float32)
  #   test_b.kernel.assign(test_vars)
  #   test_bitstring = tf.constant([[1, 0, 1]])
  #   test_spins = 1 - 2 * test_bitstring[0]
  #   with tf.GradientTape() as tape:
  #     test_energy = test_b.energy(test_bitstring)[0]
  #   test_energy_grad = tape.gradient(test_energy, test_b.kernel)
  #   ref_energy = -test_vars[0] + test_vars[1] - test_vars[2]
  #   self.assertAllClose(test_energy, ref_energy)
  #   self.assertAllClose(test_energy_grad, test_spins)

  # def test_operator_shards(self):
  #   """Confirm operators are single qubit Z only."""
  #   test_b = ebm.Bernoulli(self.num_bits)
  #   qubits = cirq.GridQubit.rect(1, self.num_bits)
  #   test_ops = test_b.operator_shards(qubits)
  #   for i, q in enumerate(qubits):
  #     self.assertEqual(test_ops[i], cirq.PauliSum.from_pauli_strings(cirq.Z(q)))

  # def test_operator_expectation(self):
  #   """Test combining expectations of operators in energy."""
  #   # Build Bernoulli
  #   num_bits = 3
  #   test_b = ebm.Bernoulli(num_bits)
  #   qubits = cirq.GridQubit.rect(1, num_bits)
  #   # Pin at bitstring [1, 0, 1]
  #   test_b.kernel.assign(tf.constant([1000.0, -1000.0, 1000.0]))
  #   operators = test_b.operator_shards(qubits)

  #   # True energy
  #   bitstring = tf.constant([[0, 0, 1]])  # not the pinned bitstring
  #   ref_energy = test_b.energy(bitstring)[0]

  #   # Test energy
  #   circuit = cirq.Circuit(
  #       [cirq.I(qubits[0]),
  #        cirq.I(qubits[1]),
  #        cirq.X(qubits[2])])
  #   output_state_vector = cirq.Simulator().simulate(circuit).final_state_vector
  #   op_expectations = []
  #   qubit_map = {q: i for i, q in enumerate(qubits)}
  #   for op in operators:
  #     op_expectations.append(
  #         op.expectation_from_state_vector(output_state_vector, qubit_map).real)
  #   test_energy = test_b.operator_expectation(op_expectations)
  #   self.assertAllClose(test_energy, ref_energy, atol=1e-4)

  # def test_sampler_bernoulli(self):
  #   """Confirm that bitstrings are sampled as expected."""
  #   test_b = ebm.Bernoulli(1, name="test")

  #   # For single factor Bernoulli, theta = 0 is 50% chance of 1.
  #   test_b.kernel.assign(tf.constant([0.0]))
  #   num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
  #   bitstrings, counts = test_b.sample(num_bitstrings)
  #   # check that we got both bitstrings
  #   self.assertTrue(
  #       tf.reduce_any(tf.equal(tf.constant([0], dtype=tf.int8), bitstrings)))
  #   self.assertTrue(
  #       tf.reduce_any(tf.equal(tf.constant([1], dtype=tf.int8), bitstrings)))
  #   # Check that the fraction is approximately 0.5 (equal counts)
  #   self.assertAllClose(1.0, counts[0] / counts[1], atol=1e-3)

  #   # Large value of theta pins the bit.
  #   test_b.kernel.assign(tf.constant([1000.0]))
  #   num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
  #   bitstrings, counts = test_b.sample(num_bitstrings)
  #   # check that we got only one bitstring
  #   self.assertAllEqual(bitstrings, [[1]])

  #   # Two bit tests.
  #   test_b = ebm.Bernoulli(2, name="test")
  #   test_b.kernel.assign(tf.constant([0.0, 0.0]))
  #   num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
  #   bitstrings, counts = test_b.sample(num_bitstrings)

  #   def check_bitstring_exists(bitstring, bitstring_list):
  #     return tf.math.reduce_any(
  #         tf.reduce_all(tf.math.equal(bitstring, bitstring_list), 1))

  #   self.assertTrue(
  #       check_bitstring_exists(tf.constant([0, 0], dtype=tf.int8), bitstrings))
  #   self.assertTrue(
  #       check_bitstring_exists(tf.constant([0, 1], dtype=tf.int8), bitstrings))
  #   self.assertTrue(
  #       check_bitstring_exists(tf.constant([1, 0], dtype=tf.int8), bitstrings))
  #   self.assertTrue(
  #       check_bitstring_exists(tf.constant([1, 1], dtype=tf.int8), bitstrings))

  #   # Check that the fraction is approximately 0.25 (equal counts)
  #   self.assertAllClose(
  #       [0.25] * 4,
  #       [counts[i].numpy() / num_bitstrings for i in range(4)],
  #       atol=1e-3,
  #   )
  #   test_b.kernel.assign(tf.constant([-1000.0, 1000.0]))
  #   num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
  #   bitstrings, counts = test_b.sample(num_bitstrings)
  #   # check that we only get 01.
  #   self.assertFalse(
  #       check_bitstring_exists(tf.constant([0, 0], dtype=tf.int8), bitstrings))
  #   self.assertTrue(
  #       check_bitstring_exists(tf.constant([0, 1], dtype=tf.int8), bitstrings))
  #   self.assertFalse(
  #       check_bitstring_exists(tf.constant([1, 0], dtype=tf.int8), bitstrings))
  #   self.assertFalse(
  #       check_bitstring_exists(tf.constant([1, 1], dtype=tf.int8), bitstrings))
  #   self.assertAllEqual(counts, [num_bitstrings])

  # def test_log_partition_bernoulli(self):
  #   # TODO
  #   pass

  # def test_entropy_bernoulli(self):
  #   """Confirm that the entropy conforms to S(p) = -sum_x p(x)ln(p(x))"""
  #   test_thetas = tf.constant([-1.5, 0.6, 2.1])
  #   # logits = 2 * thetas
  #   test_probs = ebm.logit_to_probability(2 * test_thetas).numpy()
  #   all_probs = tf.constant([
  #       (1 - test_probs[0]) * (1 - test_probs[1]) * (1 - test_probs[2]),
  #       (1 - test_probs[0]) * (1 - test_probs[1]) * (test_probs[2]),
  #       (1 - test_probs[0]) * (test_probs[1]) * (1 - test_probs[2]),
  #       (1 - test_probs[0]) * (test_probs[1]) * (test_probs[2]),
  #       (test_probs[0]) * (1 - test_probs[1]) * (1 - test_probs[2]),
  #       (test_probs[0]) * (1 - test_probs[1]) * (test_probs[2]),
  #       (test_probs[0]) * (test_probs[1]) * (1 - test_probs[2]),
  #       (test_probs[0]) * (test_probs[1]) * (test_probs[2]),
  #   ])
  #   # probabilities sum to 1
  #   self.assertAllClose(1.0, tf.reduce_sum(all_probs))
  #   expected_entropy = -1.0 * tf.reduce_sum(all_probs * tf.math.log(all_probs))
  #   test_b = ebm.Bernoulli(3, name="test")
  #   test_b.kernel.assign(test_thetas)
  #   test_entropy = test_b.entropy()
  #   self.assertAllClose(expected_entropy, test_entropy)

  # def test_trainable_variables_bernoulli(self):
  #   test_b = ebm.Bernoulli(self.num_bits, name="test")
  #   self.assertAllEqual(test_b.kernel, test_b.trainable_variables[0])

  #   kernel = tf.random.uniform([self.num_bits])
  #   test_b.trainable_variables = [kernel]
  #   self.assertAllEqual(kernel, test_b.trainable_variables[0])

  #   kernel = tf.Variable(kernel)
  #   test_b.trainable_variables = [kernel]
  #   self.assertAllEqual(kernel, test_b.trainable_variables[0])
