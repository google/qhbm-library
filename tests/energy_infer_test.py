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
import tensorflow_probability as tfp

from qhbmlib import energy_infer
from qhbmlib import energy_model
from qhbmlib import util

from tests import test_util


class AnalyticInferenceLayerTest(tf.test.TestCase):
  """Tests the AnalyticInferenceLayer class."""

  def test_init(self):
    """Confirms internal values are set correctly."""
    bits = [0, 1, 3]
    order = 2
    expected_energy = energy_model.KOBE(bits, order)
    expected_name = "test_analytic_dist_name"
    actual_layer = energy_infer.AnalyticInferenceLayer(expected_energy,
                                                       expected_name)
    self.assertEqual(actual_layer.energy, expected_energy)
    self.assertEqual(actual_layer.name, expected_name)

    expected_bitstrings = tf.constant(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1],
         [1, 1, 0], [1, 1, 1]],
        dtype=tf.int8)
    self.assertAllEqual(actual_layer.all_bitstrings, expected_bitstrings)

    expected_energies = expected_energy(expected_bitstrings)
    self.assertAllClose(actual_layer.all_energies, expected_energies)

  def test_sample(self):
    """Confirms bitstrings are sampled as expected."""
    n_samples = 1e7

    # Single bit test.
    one_bit_energy = energy_model.KOBE([0], 1)
    actual_layer = energy_infer.AnalyticInferenceLayer(one_bit_energy)
    actual_layer.build([])
    # For single factor Bernoulli, theta=0 is 50% chance of 1.
    one_bit_energy.set_weights([tf.constant([0.0])])

    # TODO(#115)
    actual_layer.infer()
    samples = actual_layer.sample(n_samples)
    # check that we got both bitstrings
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0], dtype=tf.int8), samples))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1], dtype=tf.int8), samples))
    # Check that the fraction is approximately 0.5 (equal counts)
    _, counts = util.unique_bitstrings_with_counts(samples)
    self.assertAllClose(1.0, counts[0] / counts[1], atol=1e-3)

    # Large energy penalty pins the bit.
    one_bit_energy.set_weights([tf.constant([100.0])])
    actual_layer.infer()
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
    actual_layer = energy_infer.AnalyticInferenceLayer(three_bit_energy)
    actual_layer.infer()
    samples = actual_layer.sample(n_samples)

    for b in [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1],
              [1, 1, 0], [1, 1, 1]]:
      b_tf = tf.constant([b], dtype=tf.int8)
      self.assertTrue(test_util.check_bitstring_exists(b_tf, samples))

    _, counts = util.unique_bitstrings_with_counts(samples)
    # Check that the fraction is approximately 0.125 (equal counts)
    self.assertAllClose(
        [0.125] * 8,
        tf.cast(counts, tf.float32) / tf.cast(n_samples, tf.float32),
        atol=1e-3,
    )

    # Confirm correlated spins.
    three_bit_energy.set_weights(
        [tf.constant([100.0, 0.0, 0.0, -100.0, 0.0, 100.0, 0.0])])
    actual_layer.infer()
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

  def test_log_partition(self):
    """Confirms correct value of the log partition function."""
    test_thetas = tf.constant([1.5, 2.7, -4.0])
    expected_log_partition = tf.math.log(tf.constant(3641.8353))

    energy = energy_model.KOBE([0, 1], 2)
    actual_layer = energy_infer.AnalyticInferenceLayer(energy)
    actual_layer.build([])
    energy.set_weights([test_thetas])
    actual_layer.infer()
    actual_log_partition = actual_layer.log_partition()
    self.assertAllClose(actual_log_partition, expected_log_partition)

  def test_entropy(self):
    """Confirms correct value of the entropy function."""
    test_thetas = tf.constant([1.5, 2.7, -4.0])
    expected_entropy = tf.constant(0.00233551808)

    energy = energy_model.KOBE([0, 1], 2)
    actual_layer = energy_infer.AnalyticInferenceLayer(energy)
    actual_layer.build([])
    energy.set_weights([test_thetas])
    actual_layer.infer()
    actual_entropy = actual_layer.entropy()
    self.assertAllClose(actual_entropy, expected_entropy)

  def test_call(self):
    """Confirms that call behaves correctly."""
    one_bit_energy = energy_model.KOBE([0], 1,
                                       tf.keras.initializers.Constant(0.0))
    actual_layer = energy_infer.AnalyticInferenceLayer(one_bit_energy)
    self.assertIsNone(actual_layer._current_dist)
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
    _, counts = util.unique_bitstrings_with_counts(samples)
    self.assertAllClose(1.0, counts[0] / counts[1], atol=1e-3)


class BernoulliInferenceLayerTest(tf.test.TestCase):
  """Tests the BernoulliInferenceLayer class."""

  def test_init(self):
    """Tests that components are initialized correctly."""
    bits = [0, 1, 3]
    expected_energy = energy_model.BernoulliEnergy(bits)
    expected_name = "test_analytic_dist_name"
    actual_layer = energy_infer.BernoulliInferenceLayer(expected_energy,
                                                        expected_name)
    self.assertEqual(actual_layer.energy, expected_energy)
    self.assertEqual(actual_layer.name, expected_name)

  def test_sample(self):
    """Confirms that bitstrings are sampled as expected."""
    n_samples = 1e7
    energy = energy_model.BernoulliEnergy([1])
    actual_layer = energy_infer.BernoulliInferenceLayer(energy)
    actual_layer.build([])

    # For single factor Bernoulli, theta = 0 is 50% chance of 1.
    energy.set_weights([tf.constant([0.0])])
    actual_layer.infer()
    samples = actual_layer.sample(n_samples)
    # check that we got both bitstrings
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0], dtype=tf.int8), samples))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1], dtype=tf.int8), samples))
    # Check that the fraction is approximately 0.5 (equal counts)
    _, counts = util.unique_bitstrings_with_counts(samples)
    self.assertAllClose(1.0, counts[0] / counts[1], atol=1e-3)

    # Large value of theta pins the bit.
    energy.set_weights([tf.constant([1000.0])])
    actual_layer.infer()
    samples = actual_layer.sample(n_samples)
    # check that we got only one bitstring
    bitstrings, _ = util.unique_bitstrings_with_counts(samples)
    self.assertAllEqual(bitstrings, [[1]])

    # Two bit tests.
    energy = energy_model.BernoulliEnergy([0, 1],
                                          tf.keras.initializers.Constant(0.0))
    actual_layer = energy_infer.BernoulliInferenceLayer(energy)
    actual_layer.build([])
    actual_layer.infer()
    samples = actual_layer.sample(n_samples)
    for b in [[0, 0], [0, 1], [1, 0], [1, 1]]:
      b_tf = tf.constant([b], dtype=tf.int8)
      self.assertTrue(test_util.check_bitstring_exists(b_tf, samples))
    # Check that the fraction is approximately 0.25 (equal counts)
    _, counts = util.unique_bitstrings_with_counts(samples)
    self.assertAllClose(
        [0.25] * 4,
        tf.cast(counts, tf.float32) / tf.cast(n_samples, tf.float32),
        atol=1e-3,
    )

    # Test one pinned, one free bit
    energy.set_weights([tf.constant([-1000.0, 0.0])])
    actual_layer.infer()
    samples = actual_layer.sample(n_samples)
    # check that we get 00 and 01.
    for b in [[0, 0], [0, 1]]:
      b_tf = tf.constant([b], dtype=tf.int8)
      self.assertTrue(test_util.check_bitstring_exists(b_tf, samples))
    for b in [[1, 0], [1, 1]]:
      b_tf = tf.constant([b], dtype=tf.int8)
      self.assertFalse(test_util.check_bitstring_exists(b_tf, samples))
    _, counts = util.unique_bitstrings_with_counts(samples)
    self.assertAllClose(counts, [n_samples / 2] * 2, atol=n_samples / 1000)

  def test_log_partition(self):
    """Confirms correct value of the log partition function."""
    all_bitstrings = tf.constant([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                                  [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                                 dtype=tf.int8)
    energy = energy_model.BernoulliEnergy([5, 6, 7])
    actual_layer = energy_infer.BernoulliInferenceLayer(energy)
    actual_layer.build([])
    actual_layer.infer()
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
    actual_layer = energy_infer.BernoulliInferenceLayer(energy)
    actual_layer.build([])
    energy.set_weights([test_thetas])
    actual_layer.infer()
    actual_entropy = actual_layer.entropy()
    self.assertAllClose(actual_entropy, expected_entropy)

  def test_call(self):
    """Confirms that calling the layer works correctly."""
    energy = energy_model.BernoulliEnergy([1],
                                          tf.keras.initializers.Constant(0.0))
    actual_layer = energy_infer.BernoulliInferenceLayer(energy)
    actual_layer.build([])
    self.assertIsNone(actual_layer._current_dist)
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
    _, counts = util.unique_bitstrings_with_counts(samples)
    self.assertAllClose(1.0, counts[0] / counts[1], atol=1e-3)


if __name__ == "__main__":
  print("Running energy_infer_test.py ...")
  tf.test.main()
