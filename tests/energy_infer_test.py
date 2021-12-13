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

import tensorflow as tf

from qhbmlib import energy_infer
from qhbmlib import energy_model


class BitstringDistributionTest(tf.test.TestCase):
  """Tests a basic child class of BitstringDistribution."""

  def test_init(self):
    pass


class AnalyticDistributionTest(tf.test.TestCase):
  """Tests the AnalyticDistribution class."""

  def test_init(self):
    """Confirm internal values are set correctly."""
    bits = [0, 1, 3]
    order = 2
    test_k = energy_model.KOBE(bits, order)
    expected_name = "test_analytic_dist_name"

    actual_dist = energy_infer.AnalyticDistribution(test_k, expected_name)
    self.assertEqual(test_k.num_bits, num_bits)
    self.assertEqual(test_k.order, self.order)
    self.assertTrue(test_k.has_operator)

    ref_indices = [[0], [1], [2], [0, 1], [0, 2], [1, 2]]
    self.assertAllClose(test_k._indices, ref_indices)
    self.assertAllClose(test_k.kernel, [init_const] * sum(
        len(list(itertools.combinations(range(num_bits), i)))
        for i in range(1, self.order + 1)))

  def test_sampler(self):
    """Confirm bitstrings are sampled as expected."""
    # Single bit test.
    test_k = ebm.EBM(ebm.KOBE(1, self.order), None, is_analytic=True)
    # For single factor Bernoulli, theta=0 is 50% chance of 1.
    test_thetas = tf.constant([0.0])
    test_k._energy_function.kernel.assign(test_thetas)
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
    test_k._energy_function.kernel.assign(test_thetas)
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

    def check_bitstring_exists(bitstring, bitstring_list):
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
    test_k._energy_function.kernel.assign(test_thetas)
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
    pre_k.kernel.assign(test_thetas)
    test_k = ebm.EBM(pre_k, None, is_analytic=True)
    test_log_partition = test_k.log_partition_function()
    self.assertAllClose(expected_log_partition, test_log_partition)

  def test_entropy_boltzmann(self):
    """Confirm correct value of the entropy function."""
    test_thetas = tf.Variable([1.5, 2.7, -4.0])
    expected_entropy = tf.constant(0.00233551808)
    pre_k = ebm.KOBE(2, 2)
    pre_k.kernel.assign(test_thetas)
    test_k = ebm.EBM(pre_k, None, is_analytic=True)
    test_entropy = test_k.entropy()
    self.assertAllClose(expected_entropy, test_entropy)

  
class BernoulliDistributionTest(tf.test.TestCase):
  """Tests the BernoulliDistribution class."""

  num_bits = 5

  def test_init(self):
    """Test that components are initialized correctly."""
    init_const = 1.5
    test_b = ebm.Bernoulli(self.num_bits,
                           tf.keras.initializers.Constant(init_const))
    self.assertAllEqual(test_b.num_bits, self.num_bits)
    self.assertTrue(test_b.has_operator)
    self.assertFalse(test_b.is_analytic)
    self.assertAllEqual(test_b.kernel, [init_const] * self.num_bits)

  def test_copy(self):
    """Test that the copy has the same values, but new variables."""
    test_b = ebm.Bernoulli(self.num_bits)
    test_b_copy = test_b.copy()
    self.assertAllEqual(test_b_copy.num_bits, test_b.num_bits)
    self.assertEqual(test_b_copy.has_operator, test_b.has_operator)
    self.assertEqual(test_b_copy.is_analytic, test_b.is_analytic)
    self.assertAllEqual(test_b_copy.kernel, test_b.kernel)
    self.assertNotEqual(id(test_b_copy.kernel), id(test_b.kernel))

  def test_sampler_bernoulli(self):
    """Confirm that bitstrings are sampled as expected."""
    test_b = ebm.Bernoulli(1, name="test")

    # For single factor Bernoulli, theta = 0 is 50% chance of 1.
    test_b.kernel.assign(tf.constant([0.0]))
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
    test_b.kernel.assign(tf.constant([1000.0]))
    num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
    bitstrings, counts = test_b.sample(num_bitstrings)
    # check that we got only one bitstring
    self.assertAllEqual(bitstrings, [[1]])

    # Two bit tests.
    test_b = ebm.Bernoulli(2, name="test")
    test_b.kernel.assign(tf.constant([0.0, 0.0]))
    num_bitstrings = tf.constant(int(1e7), dtype=tf.int32)
    bitstrings, counts = test_b.sample(num_bitstrings)

    def check_bitstring_exists(bitstring, bitstring_list):
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
    test_b.kernel.assign(tf.constant([-1000.0, 1000.0]))
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
    test_b.kernel.assign(test_thetas)
    test_entropy = test_b.entropy()
    self.assertAllClose(expected_entropy, test_entropy)


if __name__ == "__main__":
  print("Running energy_infer_test.py ...")
  tf.test.main()
