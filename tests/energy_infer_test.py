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
"""Tests for the energy_infer module."""

import itertools

import cirq
import tensorflow as tf

from qhbmlib import energy_infer
from qhbmlib import energy_model


class BernoulliSamplerTest(tf.test.TestCase):
  """Tests the BernoulliSampler class."""

  num_bits = 5

  def test_sampler_bernoulli(self):
    """Confirm that bitstrings are sampled as expected."""
    test_b_sampler = energy_infer.BernoulliSampler()

    # For single factor Bernoulli, theta = 0 is 50% chance of 1.
    test_b = energy_model.Bernoulli([0])
    test_b.kernel.assign(tf.constant([0.0]))
    num_samples = tf.constant(int(1e7), dtype=tf.int32)
    bitstrings, counts = test_b_sampler(test_b, num_samples)
    # Confirm shapes are correct
    self.assertAllEqual(tf.shape(bitstrings), [2, 1])
    self.assertAllEqual(tf.shape(counts), [2])
    self.assertEqual(tf.math.reduce_sum(counts), num_samples)
    
    # check that we got both bitstrings
    self.assertTrue(
        tf.reduce_any(tf.equal(tf.constant([0], dtype=tf.int8), bitstrings)))
    self.assertTrue(
        tf.reduce_any(tf.equal(tf.constant([1], dtype=tf.int8), bitstrings)))
    # Check that the fraction is approximately 0.5 (equal counts)
    self.assertAllClose(1.0, counts[0] / counts[1], atol=1e-3)

    # Large value of theta pins the bit.
    test_b.kernel.assign(tf.constant([1000.0]))
    num_samples = tf.constant(int(1e7), dtype=tf.int32)
    bitstrings, counts = test_b_sampler(test_b, num_samples)
    # check that we got only one bitstring
    self.assertAllEqual(bitstrings, [[1]])

#     # Two bit tests.
#     test_b = ebm.Bernoulli(2, name="test")
#     test_b.kernel.assign(tf.constant([0.0, 0.0]))
#     num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
#     bitstrings, counts = test_b.sample(num_bitstrings)

#     def check_bitstring_exists(bitstring, bitstring_list):
#       return tf.math.reduce_any(
#           tf.reduce_all(tf.math.equal(bitstring, bitstring_list), 1))

#     self.assertTrue(
#         check_bitstring_exists(tf.constant([0, 0], dtype=tf.int8), bitstrings))
#     self.assertTrue(
#         check_bitstring_exists(tf.constant([0, 1], dtype=tf.int8), bitstrings))
#     self.assertTrue(
#         check_bitstring_exists(tf.constant([1, 0], dtype=tf.int8), bitstrings))
#     self.assertTrue(
#         check_bitstring_exists(tf.constant([1, 1], dtype=tf.int8), bitstrings))

#     # Check that the fraction is approximately 0.25 (equal counts)
#     self.assertAllClose(
#         [0.25] * 4,
#         [counts[i].numpy() / num_bitstrings for i in range(4)],
#         atol=1e-3,
#     )
#     test_b.kernel.assign(tf.constant([-1000.0, 1000.0]))
#     num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
#     bitstrings, counts = test_b.sample(num_bitstrings)
#     # check that we only get 01.
#     self.assertFalse(
#         check_bitstring_exists(tf.constant([0, 0], dtype=tf.int8), bitstrings))
#     self.assertTrue(
#         check_bitstring_exists(tf.constant([0, 1], dtype=tf.int8), bitstrings))
#     self.assertFalse(
#         check_bitstring_exists(tf.constant([1, 0], dtype=tf.int8), bitstrings))
#     self.assertFalse(
#         check_bitstring_exists(tf.constant([1, 1], dtype=tf.int8), bitstrings))
#     self.assertAllEqual(counts, [num_bitstrings])

#   def test_log_partition_bernoulli(self):
#     # TODO
#     pass

#   def test_entropy_bernoulli(self):
#     """Confirm that the entropy conforms to S(p) = -sum_x p(x)ln(p(x))"""
#     test_thetas = tf.constant([-1.5, 0.6, 2.1])
#     # logits = 2 * thetas
#     test_probs = ebm.logit_to_probability(2 * test_thetas).numpy()
#     all_probs = tf.constant([
#         (1 - test_probs[0]) * (1 - test_probs[1]) * (1 - test_probs[2]),
#         (1 - test_probs[0]) * (1 - test_probs[1]) * (test_probs[2]),
#         (1 - test_probs[0]) * (test_probs[1]) * (1 - test_probs[2]),
#         (1 - test_probs[0]) * (test_probs[1]) * (test_probs[2]),
#         (test_probs[0]) * (1 - test_probs[1]) * (1 - test_probs[2]),
#         (test_probs[0]) * (1 - test_probs[1]) * (test_probs[2]),
#         (test_probs[0]) * (test_probs[1]) * (1 - test_probs[2]),
#         (test_probs[0]) * (test_probs[1]) * (test_probs[2]),
#     ])
#     # probabilities sum to 1
#     self.assertAllClose(1.0, tf.reduce_sum(all_probs))
#     expected_entropy = -1.0 * tf.reduce_sum(all_probs * tf.math.log(all_probs))
#     test_b = ebm.Bernoulli(3, name="test")
#     test_b.kernel.assign(test_thetas)
#     test_entropy = test_b.entropy()
#     self.assertAllClose(expected_entropy, test_entropy)

#   def test_trainable_variables_bernoulli(self):
#     test_b = ebm.Bernoulli(self.num_bits, name="test")
#     self.assertAllEqual(test_b.kernel, test_b.trainable_variables[0])

#     kernel = tf.random.uniform([self.num_bits])
#     test_b.trainable_variables = [kernel]
#     self.assertAllEqual(kernel, test_b.trainable_variables[0])

#     kernel = tf.Variable(kernel)
#     test_b.trainable_variables = [kernel]
#     self.assertAllEqual(kernel, test_b.trainable_variables[0])


# class KOBETest(tf.test.TestCase):
#   """Test the ebm.KOBE energy function."""

#   # TODO: test more orders, and compatibility with Bernoulli at order==1
#   order = 2

#   def test_init(self):
#     """Confirm internal values are set correctly."""
#     num_bits = 3
#     init_const = -3.1
#     test_k = ebm.KOBE(num_bits, self.order,
#                       tf.keras.initializers.Constant(init_const))
#     self.assertEqual(test_k.num_bits, num_bits)
#     self.assertEqual(test_k.order, self.order)
#     self.assertTrue(test_k.has_operator)

#     ref_indices = [[0], [1], [2], [0, 1], [0, 2], [1, 2]]
#     self.assertAllClose(test_k._indices, ref_indices)
#     self.assertAllClose(test_k.kernel, [init_const] * sum(
#         len(list(itertools.combinations(range(num_bits), i)))
#         for i in range(1, self.order + 1)))

#   def test_copy(self):
#     """Test that the copy has the same values, but new variables."""
#     num_bits = 3
#     init_const = -3.1
#     test_k = ebm.KOBE(num_bits, self.order,
#                       tf.keras.initializers.Constant(init_const))
#     test_k_copy = test_k.copy()
#     self.assertEqual(test_k_copy.num_bits, test_k.num_bits)
#     self.assertEqual(test_k_copy.order, test_k.order)
#     self.assertTrue(test_k_copy.has_operator, test_k.has_operator)
#     self.assertAllClose(test_k_copy._indices, test_k._indices)
#     self.assertAllClose(test_k_copy.kernel, test_k.kernel)
#     self.assertNotEqual(id(test_k_copy.kernel), id(test_k.kernel))

#   def test_energy(self):
#     """Test every energy on two bits."""
#     n_nodes = 2
#     test_thetas = tf.Variable([1.5, 2.7, -4.0])
#     expected_energies = tf.constant([0.2, 2.8, 5.2, -8.2])
#     test_k = ebm.KOBE(n_nodes, self.order)
#     test_k.kernel.assign(test_thetas)
#     all_strings = tf.constant(
#         list(itertools.product([0, 1], repeat=n_nodes)), dtype=tf.int8)
#     test_energies = test_k.energy(all_strings)
#     self.assertAllClose(expected_energies, test_energies)

#   def test_operator_shards(self):
#     """Confirm correct operators for a simple Boltzmann."""
#     num_bits = 3
#     test_k = ebm.KOBE(num_bits, self.order)
#     qubits = cirq.GridQubit.rect(1, num_bits)
#     test_ops = test_k.operator_shards(qubits)
#     ref_ops = [
#         cirq.Z(qubits[0]),
#         cirq.Z(qubits[1]),
#         cirq.Z(qubits[2]),
#         cirq.Z(qubits[0]) * cirq.Z(qubits[1]),
#         cirq.Z(qubits[0]) * cirq.Z(qubits[2]),
#         cirq.Z(qubits[1]) * cirq.Z(qubits[2])
#     ]
#     for t_op, r_op in zip(test_ops, ref_ops):
#       self.assertEqual(t_op, cirq.PauliSum.from_pauli_strings(r_op))

#   def test_operator_expectation(self):
#     """Confirm the expectations combine to the correct total energy."""
#     # Build simple Boltzmann
#     num_bits = 3
#     test_b = ebm.KOBE(num_bits, 2)
#     qubits = cirq.GridQubit.rect(1, num_bits)

#     # Pin at bitstring [1, 0, 1]
#     test_b.kernel.assign(tf.constant([100.0, -200.0, 300.0, 10, -20, 30]))
#     operators = test_b.operator_shards(qubits)

#     # True energy
#     bitstring = tf.constant([[0, 0, 1]])  # not the pinned bitstring
#     ref_energy = test_b.energy(bitstring)[0]

#     # Test energy
#     circuit = cirq.Circuit(
#         [cirq.I(qubits[0]),
#          cirq.I(qubits[1]),
#          cirq.X(qubits[2])])
#     output_state_vector = cirq.Simulator().simulate(circuit).final_state_vector
#     op_expectations = []
#     qubit_map = {q: i for i, q in enumerate(qubits)}
#     for op in operators:
#       op_expectations.append(
#           op.expectation_from_state_vector(output_state_vector, qubit_map).real)
#     test_energy = test_b.operator_expectation(op_expectations)
#     self.assertAllClose(test_energy, ref_energy, atol=1e-4)

#   def test_sampler(self):
#     """Confirm bitstrings are sampled as expected."""
#     # Single bit test.
#     test_k = ebm.EBM(ebm.KOBE(1, self.order), None, is_analytic=True)
#     # For single factor Bernoulli, theta=0 is 50% chance of 1.
#     test_thetas = tf.constant([0.0])
#     test_k._energy_function.kernel.assign(test_thetas)
#     num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
#     bitstrings, counts = test_k.sample(num_bitstrings)
#     # check that we got both bitstrings
#     self.assertTrue(
#         tf.reduce_any(tf.equal(tf.constant([0], dtype=tf.int8), bitstrings)))
#     self.assertTrue(
#         tf.reduce_any(tf.equal(tf.constant([1], dtype=tf.int8), bitstrings)))
#     # Check that the fraction is approximately 0.5 (equal counts)
#     self.assertAllClose(1.0, counts[0] / counts[1], atol=1e-3)
#     # Large energy penalty pins the bit.
#     test_thetas = tf.constant([100.0])
#     test_k._energy_function.kernel.assign(test_thetas)
#     num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
#     bitstrings, _ = test_k.sample(num_bitstrings)
#     # check that we got only one bitstring
#     self.assertFalse(
#         tf.reduce_any(tf.equal(tf.constant([0], dtype=tf.int8), bitstrings)))
#     self.assertTrue(
#         tf.reduce_any(tf.equal(tf.constant([1], dtype=tf.int8), bitstrings)))

#     # Three bit tests.
#     # First a uniform sampling test.
#     test_k = ebm.EBM(
#         ebm.KOBE(3, self.order, tf.keras.initializers.Constant(0.0)),
#         None,
#         is_analytic=True)
#     num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
#     bitstrings, counts = test_k.sample(num_bitstrings)

#     def check_bitstring_exists(bitstring, bitstring_list):
#       return tf.math.reduce_any(
#           tf.reduce_all(tf.math.equal(bitstring, bitstring_list), 1))

#     self.assertTrue(
#         check_bitstring_exists(
#             tf.constant([0, 0, 0], dtype=tf.int8), bitstrings))
#     self.assertTrue(
#         check_bitstring_exists(
#             tf.constant([0, 0, 1], dtype=tf.int8), bitstrings))
#     self.assertTrue(
#         check_bitstring_exists(
#             tf.constant([0, 1, 0], dtype=tf.int8), bitstrings))
#     self.assertTrue(
#         check_bitstring_exists(
#             tf.constant([0, 1, 1], dtype=tf.int8), bitstrings))
#     self.assertTrue(
#         check_bitstring_exists(
#             tf.constant([1, 0, 0], dtype=tf.int8), bitstrings))
#     self.assertTrue(
#         check_bitstring_exists(
#             tf.constant([1, 0, 1], dtype=tf.int8), bitstrings))
#     self.assertTrue(
#         check_bitstring_exists(
#             tf.constant([1, 1, 0], dtype=tf.int8), bitstrings))
#     self.assertTrue(
#         check_bitstring_exists(
#             tf.constant([1, 1, 1], dtype=tf.int8), bitstrings))
#     # Check that the fraction is approximately 0.125 (equal counts)
#     self.assertAllClose(
#         [0.125] * 8,
#         [counts[i].numpy() / num_bitstrings for i in range(8)],
#         atol=1e-3,
#     )
#     # Confirm correlated spins.
#     test_thetas = tf.constant([100.0, 0.0, 0.0, -100.0, 0.0, 100.0])
#     test_k._energy_function.kernel.assign(test_thetas)
#     num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
#     bitstrings, _ = test_k.sample(num_bitstrings)
#     # Confirm we only get the 110 bitstring.
#     self.assertFalse(
#         check_bitstring_exists(
#             tf.constant([0, 0, 0], dtype=tf.int8), bitstrings))
#     self.assertFalse(
#         check_bitstring_exists(
#             tf.constant([0, 0, 1], dtype=tf.int8), bitstrings))
#     self.assertFalse(
#         check_bitstring_exists(
#             tf.constant([0, 1, 0], dtype=tf.int8), bitstrings))
#     self.assertFalse(
#         check_bitstring_exists(
#             tf.constant([0, 1, 1], dtype=tf.int8), bitstrings))
#     self.assertFalse(
#         check_bitstring_exists(
#             tf.constant([1, 0, 0], dtype=tf.int8), bitstrings))
#     self.assertFalse(
#         check_bitstring_exists(
#             tf.constant([1, 0, 1], dtype=tf.int8), bitstrings))
#     self.assertTrue(
#         check_bitstring_exists(
#             tf.constant([1, 1, 0], dtype=tf.int8), bitstrings))
#     self.assertFalse(
#         check_bitstring_exists(
#             tf.constant([1, 1, 1], dtype=tf.int8), bitstrings))

#   def test_log_partition_boltzmann(self):
#     """Confirm correct value of the log partition function."""
#     test_thetas = tf.Variable([1.5, 2.7, -4.0])
#     expected_log_partition = tf.math.log(tf.constant(3641.8353))
#     pre_k = ebm.KOBE(2, 2)
#     pre_k.kernel.assign(test_thetas)
#     test_k = ebm.EBM(pre_k, None, is_analytic=True)
#     test_log_partition = test_k.log_partition_function()
#     self.assertAllClose(expected_log_partition, test_log_partition)

#   def test_entropy_boltzmann(self):
#     """Confirm correct value of the entropy function."""
#     test_thetas = tf.Variable([1.5, 2.7, -4.0])
#     expected_entropy = tf.constant(0.00233551808)
#     pre_k = ebm.KOBE(2, 2)
#     pre_k.kernel.assign(test_thetas)
#     test_k = ebm.EBM(pre_k, None, is_analytic=True)
#     test_entropy = test_k.entropy()
#     self.assertAllClose(expected_entropy, test_entropy)

#   def test_trainable_variables_kobe(self):
#     num_bits = 3
#     test_k = ebm.KOBE(num_bits, self.order, name="test")
#     self.assertAllEqual(test_k.kernel, test_k.trainable_variables[0])

#     kernel = tf.random.uniform([num_bits])
#     test_k.trainable_variables = [kernel]
#     self.assertAllEqual(kernel, test_k.trainable_variables[0])

#     kernel = tf.Variable(kernel)
#     test_k.trainable_variables = [kernel]
#     self.assertAllEqual(kernel, test_k.trainable_variables[0])


# class MLPTest(tf.test.TestCase):
#   num_bits = 5

#   def test_trainable_variables_mlp(self):
#     test_mlp = ebm.MLP(
#         self.num_bits, [4, 3, 2], activations=['relu', 'tanh', 'sigmoid'])

#     i = 0
#     for layer in test_mlp.layers:
#       self.assertAllEqual(layer.kernel, test_mlp.trainable_variables[i])
#       self.assertAllEqual(layer.bias, test_mlp.trainable_variables[i + 1])
#       i += 2

#     variables = [
#         tf.random.uniform(tf.shape(v)) for v in test_mlp.trainable_variables
#     ]
#     test_mlp.trainable_variables = variables
#     for i in range(len(test_mlp.trainable_variables)):
#       self.assertAllEqual(variables[i], test_mlp.trainable_variables[i])

#     variables = [tf.Variable(v) for v in variables]
#     test_mlp.trainable_variables = variables
#     for i in range(len(test_mlp.trainable_variables)):
#       self.assertAllEqual(variables[i], test_mlp.trainable_variables[i])


if __name__ == "__main__":
  print("Running ebm_test.py ...")
  tf.test.main()
