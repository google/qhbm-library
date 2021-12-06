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
import random

import cirq
import tensorflow as tf

from qhbmlib import energy_infer
from qhbmlib import energy_model


def check_bitstring_exists(bitstring, bitstring_list):
  """Check if the given bitstring exists in the given list."""
  return tf.math.reduce_any(
    tf.reduce_all(tf.math.equal(bitstring, bitstring_list), 1))


class BitstringSamplerTest(tf.test.TestCase):
  """Tests sampler base class."""

  class ConstantSampler(energy_infer.BitstringSampler):
    """Dummy class for testing."""

    def sample(dist, num_samples):
      """Ignores the input distribution and returns constants."""
      return tf.constant([[1] * self.num_bits]), tf.constant([num_samples])
  
  def test_init(self):
    """Confirms a dummy class instantiates correctly."""
    expected_num_bits = random.choice(range(100))
    expected_name = "test_constant_sampler"
    actual_sampler = self.ConstantSampler(expected_num_bits, expected_name)
    self.assertEqual(actual_sampler.num_bits, expected_num_bits)
    self.assertEqual(actual_sampler.name, expected_name)

  def test_init_error(self):
    """Confirms bad inputs are caught."""
    with self.assertRaisesRegex(TypeError, expected_regex="an integer"):
      _ = self.ConstantSampler("junk")
    with self.assertRaisesRegex(ValueError, expected_regex="a positive integer"):
      _ = self.ConstantSampler(-5)


class BernoulliSamplerTest(tf.test.TestCase):
  """Tests the BernoulliSampler class."""

  def test_sample(self):
    """Confirm that bitstrings are sampled as expected."""
    test_b_sampler = energy_infer.BernoulliSampler(1)

    @tf.function
    def test_sampler_traced(dist, num_samples):
      return test_b_sampler(dist, num_samples)

    for sampler in [test_b_sampler, test_sampler_traced]:
      # For single factor Bernoulli, theta = 0 is 50% chance of 1.
      test_b = energy_model.Bernoulli([0])
      test_b.kernel.assign(tf.constant([0.0]))
      num_samples = tf.constant(int(1e7), dtype=tf.int32)
      bitstrings, counts = sampler(test_b, num_samples)
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
      bitstrings, counts = sampler(test_b, num_samples)
      # check that we got only one bitstring
      self.assertAllEqual(bitstrings, [[1]])

    # Two bit tests.
    test_b_sampler = energy_infer.BernoulliSampler(2)

    @tf.function
    def test_sampler_traced(dist, num_samples):
      return test_b_sampler(dist, num_samples)

    for sampler in [test_b_sampler, test_sampler_traced]:
      test_b = energy_model.Bernoulli([1, 2])
      test_b.kernel.assign(tf.constant([0.0, 0.0]))
      num_samples = tf.constant(int(1e7), dtype=tf.int32)
      bitstrings, counts = sampler(test_b, num_samples)

      for bitstring in itertools.product([0, 1], repeat=2):
        self.assertTrue(
          check_bitstring_exists(tf.constant(bitstring, dtype=tf.int8), bitstrings))

      # Check that the fraction is approximately 0.25 (equal counts)
      self.assertAllClose(
        [0.25] * 4,
        [counts[i].numpy() / num_samples for i in range(4)],
        atol=1e-3,
      )
      test_b.kernel.assign(tf.constant([-1000.0, 1000.0]))
      num_samples = tf.constant(int(1e7), dtype=tf.int32)
      bitstrings, counts = sampler(test_b, num_samples)
      # check that we only get 01.
      self.assertFalse(
        check_bitstring_exists(tf.constant([0, 0], dtype=tf.int8), bitstrings))
      self.assertTrue(
        check_bitstring_exists(tf.constant([0, 1], dtype=tf.int8), bitstrings))
      self.assertFalse(
        check_bitstring_exists(tf.constant([1, 0], dtype=tf.int8), bitstrings))
      self.assertFalse(
        check_bitstring_exists(tf.constant([1, 1], dtype=tf.int8), bitstrings))
      self.assertAllEqual(counts, [num_samples])


class AnalyticSamplerTest(tf.test.TestCase):
  """Tests the AnalyticSampler class."""

  def test_init(self):
    """Checks that sampler is initialized correctly."""
    num_bits = random.choice(range(1, 6))
    actual_sampler = energy_infer.AnalyticSampler(num_bits)
    self.assertAllEqual(actual_sampler._all_bitstrings, list(itertools.product([0, 1], repeat=num_bits)))

  def test_sample_bernoulli(self):
    """Confirms that bitstrings are sampled from Bernoulli as expected."""
    # Uniform distribution
    num_bits = random.choice(range(1, 6))
    initializer = tf.keras.initializers.Constant(0)
    test_b = energy_model.Bernoulli(list(range(num_bits)), initializer)

    test_analytic_sampler = energy_infer.AnalyticSampler(num_bits)
    num_samples = tf.constant(int(1e7), dtype=tf.int32)
    bitstrings, counts = test_analytic_sampler(test_b, num_samples)

    for bitstring in itertools.product([0, 1], repeat=num_bits):
      self.assertTrue(
        check_bitstring_exists(tf.constant(bitstring, dtype=tf.int8), bitstrings))
    actual_fractions = counts / num_samples
    expected_fractions = [1/(2**num_bits)] * 2**num_bits
    self.assertAllClose(actual_fractions, expected_fractions, atol=1e-3)

    # Biased distribution, compare to BernoulliSampler
    num_bits = random.choice(range(1, 6))
    initializer = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)
    test_b = energy_model.Bernoulli(list(range(num_bits)), initializer)

    test_analytic_sampler = energy_infer.AnalyticSampler(num_bits)
    bernoulli_sampler = energy_infer.BernoulliSampler(num_bits)
    num_samples = tf.constant(int(1e7), dtype=tf.int32)
    actual_bitstrings, actual_counts = test_analytic_sampler(test_b, num_samples)
    expected_bitstrings, expected_counts = bernoulli_sampler(test_b, num_samples)

    # Ensure the sampled bitstrings are the same set.
    for b in itertools.product([0, 1], repeat=num_bits):
      if check_bitstring_exists(b, expected_bitstrings):
        self.assertTrue(check_bitstring_exists(b, actual_bitstrings))
      else:
        self.assertFalse(check_bitstring_exists(b, actual_bitstrings))

    # Compare the count associated with each bitstring.
    for i, expected_b in enumerate(expected_bitstrings):
      for j, actual_b in enumerate(actual_bitstrings):
        if tf.math.reduce_all(tf.math.equal(expected_b, actual_b)):
          self.assertAllClose(expected_counts[i]/actual_counts[j], 1.0, atol=1e-3)

  def test_sample_mlp(self):
    """Confirms that bitstrings are sampled from MLP as expected."""
    # Uniform distribution
    num_bits = random.choice(range(1, 6))
    num_layers = random.choice(range(10))
    units = random.sample(range(1, 100), num_layers)
    activations = ["relu"] * num_layers
    initializer = tf.keras.initializers.Constant(0)
    test_b = energy_model.MLP(list(range(num_bits)), units, activations, initializer, initializer)

    test_analytic_sampler = energy_infer.AnalyticSampler(num_bits)
    num_samples = tf.constant(int(1e7), dtype=tf.int32)
    bitstrings, counts = test_analytic_sampler(test_b, num_samples)

    for bitstring in itertools.product([0, 1], repeat=num_bits):
      self.assertTrue(
        check_bitstring_exists(tf.constant(bitstring, dtype=tf.int8), bitstrings))
    actual_fractions = counts / num_samples
    expected_fractions = [1/(2**num_bits)] * 2**num_bits
    self.assertAllClose(actual_fractions, expected_fractions, atol=1e-3)


if __name__ == "__main__":
  print("Running ebm_test.py ...")
  tf.test.main()
