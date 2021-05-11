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
"""Tests for the ebm module."""
import itertools

import tensorflow as tf

from qhbm_library import ebm
from qhbm_library import util


class ProbTest(tf.test.TestCase):
    """Test the probability<-->logit functions."""

    def test_probability_to_logit(self):
        """If p is the probability of drawing 1 from a Bernoulli distribution, then
        logit = ln(p/(1-p))
        """
        num_vals = 11
        probs = tf.random.uniform([num_vals])
        expected_logits = tf.math.log(probs / (tf.ones([num_vals]) - probs))
        test_logits = ebm.probability_to_logit(probs)
        self.assertAllClose(expected_logits, test_logits)

    def test_logit_to_probability(self):
        """If L is the log-odds of drawing a 1 from a Bernoulli distribution, then
        p = exp(L)/(1 + exp(L))
        """
        num_vals = 17
        logits = tf.random.uniform([num_vals], minval=-1000, maxval=1000)
        expected_probs = tf.math.exp(logits) / (1 + tf.math.exp(logits))
        test_probs = ebm.logit_to_probability(logits)
        self.assertAllClose(expected_probs, test_probs)


class BernoulliTest(tf.test.TestCase):
    """Test the returns from ebm.build_bernoulli."""

    def test_energy_bernoulli(self):
        """Test ebm.build_bernoulli.

        The analytic value of the energy function is E(logits, bitstring)
            = sum_i [ ln(1 + e^{logits[i]}) - bitstring[i] * logits[i] ].

        For example, if bitstring[i] = 0 for all i, then
            E(logits, bitstring) = sum_i [ ln(1 + e^{logits[i]}) ].

        Picking the logits to have values as logs of integers makes our reference
        values simpler.
        """
        energy_bernoulli, _, _, _, _ = ebm.build_bernoulli(3, "test")

        # Test energy function.
        test_logits = tf.math.log(tf.constant([10.0, 7.0, 1.0], dtype=tf.float32))
        test_bitstring = tf.constant([0, 0, 0])
        test_energy = energy_bernoulli(test_logits, test_bitstring)
        ref_energy = tf.reduce_sum(
            tf.math.log(tf.constant([11.0, 8.0, 2.0], dtype=tf.float32))
        )
        self.assertAllClose(test_energy, ref_energy)
        test_logits = tf.math.log(tf.constant([1.0, 1.7, 2.8], dtype=tf.float32))
        test_bitstring = tf.constant([1, 0, 1])
        test_energy = energy_bernoulli(test_logits, test_bitstring)
        ref_energy = (
            tf.reduce_sum(tf.math.log(tf.constant([2.0, 2.7, 3.8], dtype=tf.float32)))
            - tf.math.log(1.0)
            - tf.math.log(2.8)
        )
        self.assertAllClose(test_energy, ref_energy)

    def test_energy_bernoulli_gradients(self):
        """Test gradients of energy_bernoulli.

        The analytic gradient of the energy function w.r.t. a given logit i is
        (d/(d logit[i])) E(logits, bitstring)
            = (e^{logit[i]} / (1 + e^{logit[i]})) - bitstring[i]
        """
        (
            energy_bernoulli,
            sampler_bernoulli,
            log_partition_bernoulli,
            entropy_bernoulli,
            num_nodes,
        ) = ebm.build_bernoulli(3, "test")
        test_logits = tf.math.log(tf.constant([10.0, 7.0, 1.0]))
        test_bitstring = tf.constant([True, False, True])
        with tf.GradientTape() as tape:
            tape.watch(test_logits)
            test_energy = energy_bernoulli(test_logits, test_bitstring)
        test_gradients = tape.gradient(test_energy, test_logits)
        ref_gradients = [10 / 11 - 1, 7 / 8, 1 / 2 - 1]
        self.assertAllClose(test_gradients, ref_gradients)

    def test_sampler_bernoulli(self):
        """Confirm that bitstrings are sampled as expected."""
        # Single bit test.
        _, sampler_bernoulli, _, _, _ = ebm.build_bernoulli(1, "test")
        # For single factor Bernoulli, 0 logit is 50% chance of 1.
        test_logits = tf.constant([0.0])
        num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
        bitstrings = sampler_bernoulli(test_logits, num_bitstrings)
        _, _, counts = util.unique_with_counts(bitstrings)
        # check that we got both bitstrings
        self.assertTrue(
            tf.reduce_any(tf.equal(tf.constant([0], dtype=tf.int8), bitstrings))
        )
        self.assertTrue(
            tf.reduce_any(tf.equal(tf.constant([1], dtype=tf.int8), bitstrings))
        )
        # Check that the fraction is approximately 0.5 (equal counts)
        self.assertAllClose(1.0, counts[0] / counts[1], atol=1e-3)
        # Large logit pins the bit.
        test_logits = tf.constant([1000.0])
        num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
        bitstrings = sampler_bernoulli(test_logits, num_bitstrings)
        # check that we got only one bitstring
        self.assertFalse(
            tf.reduce_any(tf.equal(tf.constant([0], dtype=tf.int8), bitstrings))
        )
        self.assertTrue(
            tf.reduce_any(tf.equal(tf.constant([1], dtype=tf.int8), bitstrings))
        )

        # Two bit tests.
        _, sampler_bernoulli, _, _, _ = ebm.build_bernoulli(2, "test")
        test_logits = tf.constant([0.0, 0.0])
        num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
        bitstrings = sampler_bernoulli(test_logits, num_bitstrings)
        _, _, counts = util.unique_with_counts(bitstrings)
        # check that we got all bitstrings.
        @tf.function
        def check_bitstring_exists(bitstring, bitstring_list):
            print("retracing: check_bitstring_exists")
            return tf.math.reduce_any(
                tf.reduce_all(tf.math.equal(bitstring, bitstring_list), 1)
            )

        self.assertTrue(
            check_bitstring_exists(tf.constant([0, 0], dtype=tf.int8), bitstrings)
        )
        self.assertTrue(
            check_bitstring_exists(tf.constant([0, 1], dtype=tf.int8), bitstrings)
        )
        self.assertTrue(
            check_bitstring_exists(tf.constant([1, 0], dtype=tf.int8), bitstrings)
        )
        self.assertTrue(
            check_bitstring_exists(tf.constant([1, 1], dtype=tf.int8), bitstrings)
        )

        # Check that the fraction is approximately 0.25 (equal counts)
        self.assertAllClose(
            [0.25] * 4,
            [counts[i].numpy() / num_bitstrings for i in range(4)],
            atol=1e-3,
        )
        test_logits = tf.constant([-1000.0, 1000.0])
        num_bitstrings = tf.constant(int(1e6), dtype=tf.int32)
        bitstrings = sampler_bernoulli(test_logits, num_bitstrings)
        # check that we only get 01.
        self.assertFalse(
            check_bitstring_exists(tf.constant([0, 0], dtype=tf.int8), bitstrings)
        )
        self.assertTrue(
            check_bitstring_exists(tf.constant([0, 1], dtype=tf.int8), bitstrings)
        )
        self.assertFalse(
            check_bitstring_exists(tf.constant([1, 0], dtype=tf.int8), bitstrings)
        )
        self.assertFalse(
            check_bitstring_exists(tf.constant([1, 1], dtype=tf.int8), bitstrings)
        )

    def test_log_partition_bernoulli(self):
        """Our definition of the energy leads to Z = 1.0 for any parameters."""
        n_nodes = 7
        _, _, log_partition_bernoulli, _, _ = ebm.build_bernoulli(n_nodes, "test")
        test_thetas = tf.random.uniform([n_nodes], minval=-10, maxval=10)
        expected_log_partition = 0.0
        test_log_partition = log_partition_bernoulli(test_thetas)
        self.assertAllClose(expected_log_partition, test_log_partition)

    def test_entropy_bernoulli(self):
        """Confirm that the entropy conforms to S(p) = -sum_x p(x)ln(p(x))"""
        test_thetas = tf.constant([-1.5, 0.6, 2.1])
        test_probs = ebm.logit_to_probability(test_thetas).numpy()
        all_probs = tf.constant(
            [
                (1 - test_probs[0]) * (1 - test_probs[1]) * (1 - test_probs[2]),
                (1 - test_probs[0]) * (1 - test_probs[1]) * (test_probs[2]),
                (1 - test_probs[0]) * (test_probs[1]) * (1 - test_probs[2]),
                (1 - test_probs[0]) * (test_probs[1]) * (test_probs[2]),
                (test_probs[0]) * (1 - test_probs[1]) * (1 - test_probs[2]),
                (test_probs[0]) * (1 - test_probs[1]) * (test_probs[2]),
                (test_probs[0]) * (test_probs[1]) * (1 - test_probs[2]),
                (test_probs[0]) * (test_probs[1]) * (test_probs[2]),
            ]
        )
        # probabilities sum to 1
        self.assertAllClose(1.0, tf.reduce_sum(all_probs))
        expected_entropy = -1.0 * tf.reduce_sum(all_probs * tf.math.log(all_probs))
        _, _, _, entropy_bernoulli, _ = ebm.build_bernoulli(2, "test")
        test_entropy = entropy_bernoulli(test_thetas)
        self.assertAllClose(expected_entropy, test_entropy)

    def test_num_thetas_bernoulli(self):
        """The number of parameters is the same as the number of nodes."""
        num_nodes = tf.random.uniform([], minval=1, maxval=100, dtype=tf.int32)
        _, _, _, _, test_num_thetas = ebm.build_bernoulli(num_nodes, "test")
        self.assertEqual(num_nodes, test_num_thetas)


class BoltzmannTest(tf.test.TestCase):
    """Test ebm.build_boltzmann."""

    def test_energy_boltzmann(self):
        """The Boltzmann energy function is defined as: TODO

        Test every energy on two bits.
        """
        n_nodes = 2
        test_thetas = tf.Variable([1.5, 2.7, -4.0])
        expected_energies = tf.constant([0.2, 2.8, 5.2, -8.2])
        energy_boltzmann, _, _, _, _ = ebm.build_boltzmann(n_nodes, "test")
        all_strings = tf.constant(
            list(itertools.product([0, 1], repeat=n_nodes)), dtype=tf.int8
        )
        test_energies = tf.map_fn(
            lambda x: energy_boltzmann(test_thetas, x),
            all_strings,
            fn_output_signature=tf.float32,
        )
        self.assertAllClose(expected_energies, test_energies)

    def test_sampler_boltzmann(self):
        """Confirm bitstrings are sampled as expected."""
        # Single bit test.
        _, sampler_boltzmann, _, _, _ = ebm.build_boltzmann(1, "test_single_bit")
        # For single factor Bernoulli, 0 logit is 50% chance of 1.
        test_thetas = tf.constant([0.0])
        num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
        bitstrings = sampler_boltzmann(test_thetas, num_bitstrings)
        _, _, counts = util.unique_with_counts(bitstrings)
        # check that we got both bitstrings
        self.assertTrue(
            tf.reduce_any(tf.equal(tf.constant([0], dtype=tf.int8), bitstrings))
        )
        self.assertTrue(
            tf.reduce_any(tf.equal(tf.constant([1], dtype=tf.int8), bitstrings))
        )
        # Check that the fraction is approximately 0.5 (equal counts)
        self.assertAllClose(1.0, counts[0] / counts[1], atol=1e-3)
        # Large energy penalty pins the bit.
        test_thetas = tf.constant([100.0])
        num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
        bitstrings = sampler_boltzmann(test_thetas, num_bitstrings)
        # check that we got only one bitstring
        self.assertFalse(
            tf.reduce_any(tf.equal(tf.constant([0], dtype=tf.int8), bitstrings))
        )
        self.assertTrue(
            tf.reduce_any(tf.equal(tf.constant([1], dtype=tf.int8), bitstrings))
        )

        # Three bit tests.
        _, sampler_boltzmann, _, _, _ = ebm.build_boltzmann(3, "test")
        # First a uniform sampling test.
        test_thetas = tf.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
        bitstrings = sampler_boltzmann(test_thetas, num_bitstrings)
        _, _, counts = util.unique_with_counts(bitstrings)
        # check that we got all bitstrings.
        @tf.function
        def check_bitstring_exists(bitstring, bitstring_list):
            print("retracing: check_bitstring_exists")
            return tf.math.reduce_any(
                tf.reduce_all(tf.math.equal(bitstring, bitstring_list), 1)
            )

        self.assertTrue(
            check_bitstring_exists(tf.constant([0, 0, 0], dtype=tf.int8), bitstrings)
        )
        self.assertTrue(
            check_bitstring_exists(tf.constant([0, 0, 1], dtype=tf.int8), bitstrings)
        )
        self.assertTrue(
            check_bitstring_exists(tf.constant([0, 1, 0], dtype=tf.int8), bitstrings)
        )
        self.assertTrue(
            check_bitstring_exists(tf.constant([0, 1, 1], dtype=tf.int8), bitstrings)
        )
        self.assertTrue(
            check_bitstring_exists(tf.constant([1, 0, 0], dtype=tf.int8), bitstrings)
        )
        self.assertTrue(
            check_bitstring_exists(tf.constant([1, 0, 1], dtype=tf.int8), bitstrings)
        )
        self.assertTrue(
            check_bitstring_exists(tf.constant([1, 1, 0], dtype=tf.int8), bitstrings)
        )
        self.assertTrue(
            check_bitstring_exists(tf.constant([1, 1, 1], dtype=tf.int8), bitstrings)
        )
        # Check that the fraction is approximately 0.125 (equal counts)
        self.assertAllClose(
            [0.125] * 8,
            [counts[i].numpy() / num_bitstrings for i in range(8)],
            atol=1e-3,
        )
        # Confirm correlated spins.
        test_thetas = tf.constant([100.0, 0.0, 0.0, -100.0, 0.0, 100.0])
        num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
        bitstrings = sampler_boltzmann(test_thetas, num_bitstrings)
        # Confirm we only get the 110 bitstring.
        self.assertFalse(
            check_bitstring_exists(tf.constant([0, 0, 0], dtype=tf.int8), bitstrings)
        )
        self.assertFalse(
            check_bitstring_exists(tf.constant([0, 0, 1], dtype=tf.int8), bitstrings)
        )
        self.assertFalse(
            check_bitstring_exists(tf.constant([0, 1, 0], dtype=tf.int8), bitstrings)
        )
        self.assertFalse(
            check_bitstring_exists(tf.constant([0, 1, 1], dtype=tf.int8), bitstrings)
        )
        self.assertFalse(
            check_bitstring_exists(tf.constant([1, 0, 0], dtype=tf.int8), bitstrings)
        )
        self.assertFalse(
            check_bitstring_exists(tf.constant([1, 0, 1], dtype=tf.int8), bitstrings)
        )
        self.assertTrue(
            check_bitstring_exists(tf.constant([1, 1, 0], dtype=tf.int8), bitstrings)
        )
        self.assertFalse(
            check_bitstring_exists(tf.constant([1, 1, 1], dtype=tf.int8), bitstrings)
        )

    def test_log_partition_boltzmann(self):
        """Confirm correct value of the log partition function."""
        n_nodes = 2
        test_thetas = tf.Variable([1.5, 2.7, -4.0])
        expected_log_partition = tf.math.log(tf.constant(3641.8353))
        _, _, log_partition_boltzmann, _, _ = ebm.build_boltzmann(n_nodes, "test")
        test_log_partition = log_partition_boltzmann(test_thetas)
        self.assertAllClose(expected_log_partition, test_log_partition)

    def test_entropy_boltzmann(self):
        """Confirm correct value of the entropy function."""
        n_nodes = 2
        test_thetas = tf.Variable([1.5, 2.7, -4.0])
        expected_entropy = tf.constant(0.00233551808)
        _, _, _, entropy_boltzmann, _ = ebm.build_boltzmann(n_nodes, "test")
        test_entropy = entropy_boltzmann(test_thetas)
        self.assertAllClose(expected_entropy, test_entropy)

    def test_num_thetas_boltzmann(self):
        """Confirm the correct number of parameters are requested.

        There should be one parameter per bit, as well as an additional parameter
        per pair of bits.
        """
        _, _, _, _, test_num_thetas = ebm.build_boltzmann(1, "test")
        self.assertEqual(1, test_num_thetas)
        _, _, _, _, test_num_thetas = ebm.build_boltzmann(2, "test")
        self.assertEqual(3, test_num_thetas)
        _, _, _, _, test_num_thetas = ebm.build_boltzmann(3, "test")
        self.assertEqual(6, test_num_thetas)
        _, _, _, _, test_num_thetas = ebm.build_boltzmann(4, "test")
        self.assertEqual(10, test_num_thetas)

    # TODO(zaqqwerty): tests for general EBM functions and operators


if __name__ == "__main__":
    print("Running ebm_test.py ...")
    tf.test.main()
