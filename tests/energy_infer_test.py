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

import tensorflow as tf
import tensorflow_probability as tfp

from qhbmlib import energy_infer
from qhbmlib import energy_model
from qhbmlib import energy_model_utils
from qhbmlib import utils

from tests import test_util


class EnergyInferenceTest(self):
  """Tests a simple instantiation of EnergyInference."""

  class TwoOutcomes(energy_infer.EnergyInference):
    """EnergyInference which is independent of the input energy."""
    
    def __init__(self, bitstring_1, bitstring_2, p_1):
      """Initializes a simple inference class.

      Args:
        bitstring_1: First bitstring to sample.
        bitstring_2: Second bitstring to sample.
        p_1: probability of sampling the first bitstring.
      """
      self.bitstring_1 = tf.constant([bitstring_1], dtype=tf.int8)
      self.bitstring_2 = tf.constant([bitstring_2], dtype=tf.int8)
      self.p_1 = p_1

    def infer(self, energy):
      """Ignores the energy."""
      del energy

    def sample(self, n):
      """Deterministically samples bitstrings."""
      n_1 = round(self.p_1 * n)
      n_2 = n - n_1
      bitstring_1_tile = tf.tile(self.bitstring_1, [n_1, 1])
      bitstring_2_tile = tf.tile(self.bitstring_2, [n_2, 1])
      return tf.concat([bitstring_1_tile, bitstring_2_tile], 0)

    def entropy(self):
      """Not implemented in this test class."""
      raise NotImplementedError()

    def log_partition(self):
      """Not implemented in this test class."""
      raise NotImplementedError()

  class TestLayer(tf.keras.layers.Layer):
    """Simple test layer to send to the expectation method."""

    def __init__(self, bits, order):
      """Initializes a spin conversion and parity in the TestLayer."""
      self.spins_from_bitstrings = energy_model_utils.SpinsFromBitstrings()
      self.parity = energy_model_utils.Parity(bits, order)

    def call(self, bitstrings):
      """Apply the test layer to input bitstrings."""
      return self.parity(self.spins_from_bitstrings)

  def setUp(self):
    """Initializes test objects."""
    super().setUp()
    self.bitstring_1 = [1, 1, 0, 1, 0]
    self.bitstring_2 = [0, 0, 0, 1, 1]
    self.p_1 = 0.1
    self.e_infer = self.TwoOutcomes(self.bitstring_1, self.bitstring_2, self.p_1)
    self.test_layer = self.TestLayer(list(range(5)), 2)

  def test_expectation(self):
    """Confirms correct averaging over input function."""
    values = []
    for b in [[self.bitstring_1], [self.bitstring_2]]:
      values.append(self.test_layer(b)[0])
    expected_expectation = self.p_1 * values[0] + (1 - self.p_1) * values[1]

    num_samples = 1e6
    actual_expectation = self.e_infer.expectation(self.test_layer, num_samples)
    
    self.assertAllClose(actual_expectation, expected_expectation)


class AnalyticEnergyInferenceTest(tf.test.TestCase):
  """Tests the AnalyticEnergyInference class."""

  def test_init(self):
    """Confirms internal values are set correctly."""
    bits = [0, 1, 3]
    order = 2
    expected_name = "test_analytic_dist_name"
    actual_layer = energy_infer.AnalyticEnergyInference(
        len(bits), expected_name)
    self.assertEqual(actual_layer.name, expected_name)

    expected_bitstrings = tf.constant(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1],
         [1, 1, 0], [1, 1, 1]],
        dtype=tf.int8)
    self.assertAllEqual(actual_layer.all_bitstrings, expected_bitstrings)

    energy = energy_model.KOBE(bits, order)
    expected_energies = energy(expected_bitstrings)
    actual_layer.infer(energy)
    self.assertAllClose(actual_layer.all_energies, expected_energies)

  def test_sample(self):
    """Confirms bitstrings are sampled as expected."""
    n_samples = 1e7
    seed = tf.constant([1, 2], tf.int32)

    # Single bit test.
    one_bit_energy = energy_model.KOBE([0], 1)
    one_bit_energy.build([None, one_bit_energy.num_bits])
    actual_layer = energy_infer.AnalyticEnergyInference(1, seed=seed)
    # For single factor Bernoulli, theta=0 is 50% chance of 1.
    one_bit_energy.set_weights([tf.constant([0.0])])

    # TODO(#115)
    actual_layer.infer(one_bit_energy)
    samples = actual_layer.sample(n_samples)
    # check that we got both bitstrings
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0], dtype=tf.int8), samples))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1], dtype=tf.int8), samples))
    # Check that the fraction is approximately 0.5 (equal counts)
    _, counts = utils.unique_bitstrings_with_counts(samples)
    self.assertAllClose(1.0, counts[0] / counts[1], atol=1e-3)

    # Large energy penalty pins the bit.
    one_bit_energy.set_weights([tf.constant([100.0])])
    actual_layer.infer(one_bit_energy)
    samples = actual_layer.sample(n_samples)
    # check that we got only one bitstring
    self.assertFalse(
        test_util.check_bitstring_exists(
            tf.constant([0], dtype=tf.int8), samples))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1], dtype=tf.int8), samples))

    # Three bit tests.
    # First a uniform sampling test.
    three_bit_energy = energy_model.KOBE([0, 1, 2], 3,
                                         tf.keras.initializers.Constant(0.0))
    actual_layer = energy_infer.AnalyticEnergyInference(3, seed=seed)
    actual_layer.infer(three_bit_energy)
    samples = actual_layer.sample(n_samples)

    for b in [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1],
              [1, 1, 0], [1, 1, 1]]:
      b_tf = tf.constant([b], dtype=tf.int8)
      self.assertTrue(test_util.check_bitstring_exists(b_tf, samples))

    _, counts = utils.unique_bitstrings_with_counts(samples)
    # Check that the fraction is approximately 0.125 (equal counts)
    self.assertAllClose(
        [0.125] * 8,
        tf.cast(counts, tf.float32) / tf.cast(n_samples, tf.float32),
        atol=1e-3,
    )

    # Confirm correlated spins.
    three_bit_energy.set_weights(
        [tf.constant([100.0, 0.0, 0.0, -100.0, 0.0, 100.0, 0.0])])
    actual_layer.infer(three_bit_energy)
    samples = actual_layer.sample(n_samples)
    # Confirm we only get the 110 bitstring.

    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1, 1, 0], dtype=tf.int8), samples))
    for b in [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        # [1, 1, 0],
        [1, 1, 1]
    ]:
      b_tf = tf.constant([b], dtype=tf.int8)
      self.assertFalse(test_util.check_bitstring_exists(b_tf, samples))

  def test_samples_seeded(self):
    """Confirm seeding fixes samples for given energy."""
    num_bits = 5
    seed = tf.constant([1, 2], tf.int32)  # seed in TFP style
    num_samples = 1e6
    energy = energy_model.KOBE(list(range(num_bits)), 2)
    energy.build([None, num_bits])
    actual_layer = energy_infer.AnalyticEnergyInference(num_bits, seed=seed)
    actual_layer.infer(energy)
    samples_1 = actual_layer.sample(num_samples)
    samples_2 = actual_layer.sample(num_samples)
    self.assertAllEqual(samples_1, samples_2)

    # check unseeding lets samples be different again
    actual_layer.seed = None
    samples_1 = actual_layer.sample(num_samples)
    samples_2 = actual_layer.sample(num_samples)
    self.assertNotAllEqual(samples_1, samples_2)

  def test_log_partition(self):
    """Confirms correct value of the log partition function."""
    test_thetas = tf.constant([1.5, 2.7, -4.0])
    expected_log_partition = tf.math.log(tf.constant(3641.8353))

    energy = energy_model.KOBE([0, 1], 2)
    energy.build([None, energy.num_bits])
    actual_layer = energy_infer.AnalyticEnergyInference(2)
    energy.set_weights([test_thetas])
    actual_layer.infer(energy)
    actual_log_partition = actual_layer.log_partition()
    self.assertAllClose(actual_log_partition, expected_log_partition)

  def test_entropy(self):
    """Confirms correct value of the entropy function."""
    test_thetas = tf.constant([1.5, 2.7, -4.0])
    expected_entropy = tf.constant(0.00233551808)

    energy = energy_model.KOBE([0, 1], 2)
    energy.build([None, energy.num_bits])
    actual_layer = energy_infer.AnalyticEnergyInference(2)
    energy.set_weights([test_thetas])
    actual_layer.infer(energy)
    actual_entropy = actual_layer.entropy()
    self.assertAllClose(actual_entropy, expected_entropy)

  def test_call(self):
    """Confirms that call behaves correctly."""
    seed = tf.constant([1, 2], tf.int32)
    one_bit_energy = energy_model.KOBE([0], 1,
                                       tf.keras.initializers.Constant(0.0))
    actual_layer = energy_infer.AnalyticEnergyInference(1, seed=seed)
    self.assertIsNone(actual_layer.current_dist)
    with self.assertRaisesRegex(
        RuntimeError, expected_regex="`infer` must be called"):
      _ = actual_layer(None)
    actual_layer.infer(one_bit_energy)
    actual_dist = actual_layer(None)
    self.assertIsInstance(actual_dist, tfp.distributions.Categorical)

    n_samples = 1e7
    samples = actual_layer(n_samples)
    # check that we got both bitstrings
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0], dtype=tf.int8), samples))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1], dtype=tf.int8), samples))
    # Check that the fraction is approximately 0.5 (equal counts)
    _, counts = utils.unique_bitstrings_with_counts(samples)
    self.assertAllClose(1.0, counts[0] / counts[1], atol=1e-3)


class BernoulliEnergyInferenceTest(tf.test.TestCase):
  """Tests the BernoulliEnergyInference class."""

  def test_init(self):
    """Tests that components are initialized correctly."""
    expected_name = "test_analytic_dist_name"
    actual_layer = energy_infer.BernoulliEnergyInference(expected_name)
    self.assertEqual(actual_layer.name, expected_name)

  def test_sample(self):
    """Confirms that bitstrings are sampled as expected."""
    n_samples = 1e7
    seed = tf.constant([1, 2], tf.int32)
    energy = energy_model.BernoulliEnergy([1])
    energy.build([None, energy.num_bits])
    actual_layer = energy_infer.BernoulliEnergyInference(seed=seed)

    # For single factor Bernoulli, theta = 0 is 50% chance of 1.
    energy.set_weights([tf.constant([0.0])])
    actual_layer.infer(energy)
    samples = actual_layer.sample(n_samples)
    # check that we got both bitstrings
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0], dtype=tf.int8), samples))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1], dtype=tf.int8), samples))
    # Check that the fraction is approximately 0.5 (equal counts)
    _, counts = utils.unique_bitstrings_with_counts(samples)
    self.assertAllClose(1.0, counts[0] / counts[1], atol=1e-3)

    # Large value of theta pins the bit.
    energy.set_weights([tf.constant([1000.0])])
    actual_layer.infer(energy)
    samples = actual_layer.sample(n_samples)
    # check that we got only one bitstring
    bitstrings, _ = utils.unique_bitstrings_with_counts(samples)
    self.assertAllEqual(bitstrings, [[1]])

    # Two bit tests.
    energy = energy_model.BernoulliEnergy([0, 1],
                                          tf.keras.initializers.Constant(0.0))
    energy.build([None, energy.num_bits])
    actual_layer = energy_infer.BernoulliEnergyInference(seed=seed)
    actual_layer.infer(energy)
    samples = actual_layer.sample(n_samples)
    for b in [[0, 0], [0, 1], [1, 0], [1, 1]]:
      b_tf = tf.constant([b], dtype=tf.int8)
      self.assertTrue(test_util.check_bitstring_exists(b_tf, samples))
    # Check that the fraction is approximately 0.25 (equal counts)
    _, counts = utils.unique_bitstrings_with_counts(samples)
    self.assertAllClose(
        [0.25] * 4,
        tf.cast(counts, tf.float32) / tf.cast(n_samples, tf.float32),
        atol=1e-3,
    )

    # Test one pinned, one free bit
    energy.set_weights([tf.constant([-1000.0, 0.0])])
    actual_layer.infer(energy)
    samples = actual_layer.sample(n_samples)
    # check that we get 00 and 01.
    for b in [[0, 0], [0, 1]]:
      b_tf = tf.constant([b], dtype=tf.int8)
      self.assertTrue(test_util.check_bitstring_exists(b_tf, samples))
    for b in [[1, 0], [1, 1]]:
      b_tf = tf.constant([b], dtype=tf.int8)
      self.assertFalse(test_util.check_bitstring_exists(b_tf, samples))
    _, counts = utils.unique_bitstrings_with_counts(samples)
    self.assertAllClose(counts, [n_samples / 2] * 2, atol=n_samples / 1000)

  def test_samples_seeded(self):
    """Confirm seeding fixes samples for given energy."""
    num_bits = 5
    seed = tf.constant([1, 2], tf.int32)  # seed in TFP style
    num_samples = 1e6
    energy = energy_model.BernoulliEnergy(list(range(num_bits)))
    energy.build([None, num_bits])
    actual_layer = energy_infer.BernoulliEnergyInference(seed=seed)
    actual_layer.infer(energy)
    samples_1 = actual_layer.sample(num_samples)
    samples_2 = actual_layer.sample(num_samples)
    self.assertAllEqual(samples_1, samples_2)

    # check unseeding lets samples be different again
    actual_layer.seed = None
    samples_1 = actual_layer.sample(num_samples)
    samples_2 = actual_layer.sample(num_samples)
    self.assertNotAllEqual(samples_1, samples_2)

  def test_log_partition(self):
    """Confirms correct value of the log partition function."""
    all_bitstrings = tf.constant([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                                  [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                                 dtype=tf.int8)
    energy = energy_model.BernoulliEnergy([5, 6, 7])
    energy.build([None, energy.num_bits])
    actual_layer = energy_infer.BernoulliEnergyInference()
    actual_layer.infer(energy)
    expected_log_partition = tf.reduce_logsumexp(-1.0 * energy(all_bitstrings))
    actual_log_partition = actual_layer.log_partition()
    self.assertAllClose(actual_log_partition, expected_log_partition)

  def test_entropy(self):
    r"""Confirms that the entropy is S(p) = -\sum_x p(x)\ln(p(x)).

    For logit $\eta$ and probability of 1 $p$, we have
    $\eta = \log(p / 1-p)$, so $p = \frac{e^{\eta}}{1 + e^{\eta}}$.
    """
    test_thetas = tf.constant([-1.5, 0.6, 2.1])
    logits = 2 * test_thetas
    num = tf.math.exp(logits)
    denom = 1 + num
    test_probs = (num / denom).numpy()
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

    energy = energy_model.BernoulliEnergy([0, 1, 2])
    energy.build([None, energy.num_bits])
    actual_layer = energy_infer.BernoulliEnergyInference()
    energy.set_weights([test_thetas])
    actual_layer.infer(energy)
    actual_entropy = actual_layer.entropy()
    self.assertAllClose(actual_entropy, expected_entropy)

  def test_call(self):
    """Confirms that calling the layer works correctly."""
    seed = tf.constant([1, 2], tf.int32)
    energy = energy_model.BernoulliEnergy([1],
                                          tf.keras.initializers.Constant(0.0))
    energy.build([None, energy.num_bits])
    actual_layer = energy_infer.BernoulliEnergyInference(seed=seed)
    self.assertIsNone(actual_layer.current_dist)
    with self.assertRaisesRegex(
        RuntimeError, expected_regex="`infer` must be called"):
      _ = actual_layer(None)
    actual_layer.infer(energy)
    actual_dist = actual_layer(None)
    self.assertIsInstance(actual_dist, tfp.distributions.Bernoulli)

    # For single factor Bernoulli, theta = 0 is 50% chance of 1.
    n_samples = 1e7
    samples = actual_layer(n_samples)
    # check that we got both bitstrings
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0], dtype=tf.int8), samples))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1], dtype=tf.int8), samples))
    # Check that the fraction is approximately 0.5 (equal counts)
    _, counts = utils.unique_bitstrings_with_counts(samples)
    self.assertAllClose(1.0, counts[0] / counts[1], atol=1e-3)


if __name__ == "__main__":
  print("Running energy_infer_test.py ...")
  tf.test.main()
