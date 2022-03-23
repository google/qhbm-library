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
"""Tests for qhbmlib.inference.ebm_utils"""

import itertools
import random

import tensorflow as tf

from qhbmlib import inference
from qhbmlib import models
from qhbmlib import utils


class ProbabilitiesTest(tf.test.TestCase):
  """Tests the probabilities function."""

  def test_probabilities(self):
    """Confirms probabilities are correct for an MLP."""
    num_bits = 5
    num_layers = 3
    units = random.sample(range(1, 100), num_layers)
    activations = random.sample([
        "elu", "exponential", "gelu", "hard_sigmoid", "linear", "relu", "selu",
        "sigmoid", "softmax", "softplus", "softsign", "swish", "tanh"
    ], num_layers)
    expected_layer_list = []
    for i in range(num_layers):
      expected_layer_list.append(
          tf.keras.layers.Dense(units[i], activation=activations[i]))
    expected_layer_list.append(tf.keras.layers.Dense(1))
    expected_layer_list.append(utils.Squeeze(-1))
    actual_energy = models.BitstringEnergy(
        list(range(num_bits)), expected_layer_list)

    num_expectation_samples = 1  # Required but unused
    infer = inference.AnalyticEnergyInference(actual_energy,
                                              num_expectation_samples)
    expected_probabilities = infer.distribution.probs_parameter()

    probabilities_wrapped = tf.function(inference.probabilities)
    actual_probabilities = probabilities_wrapped(actual_energy)
    self.assertAllClose(actual_probabilities, expected_probabilities)


class RelaxedCategoricalTest(tf.test.TestCase):
  """Tests relaxed categorical utilities."""

  def test_relaxed_categorical_probabilities(self):
    """Checks probabilities of all bitstrings."""
    num_bits = 3
    category_bitstrings = tf.constant([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 1, 0],
        [1, 1, 1]], dtype=tf.int8)
    num_unique = 4
    categorical_prob_contributions = tf.constant([3.0/7, 0.0, 1.0/7, 0.0, 0.0, 0.0, 2.0/7, 1.0/7])
    
    uniform_prob_contributions = tf.constant([1.0/8] * 8)
    ratio = inference.ebm_utils._relaxed_categorical_ratio(num_bits, num_unique)
    expected_probabilities = ratio * categorical_prob_contributions + (1.0 - ratio) * uniform_prob_contributions
    self.assertAllClose(tf.reduce_sum(expected_probabilities), 1.0)
    
    input_bitstrings = tf.constant(
        list(itertools.product([0, 1], repeat=num_bits)),
        dtype=tf.int8)
    actual_probabilities = inference.relaxed_categorical_probabilities(category_bitstrings, input_bitstrings)
    self.assertAllClose(actual_probabilities, expected_probabilities)


  def test_relaxed_categorical_samples(self):
    """Confirms sample probabilities match expected probabilities."""
    num_bits = 5
    all_bitstrings = list(itertools.product([0, 1], repeat=num_bits))
    all_bitstrings_tensor = tf.constant(all_bitstrings, dtype=tf.int8)
    num_random_bitstrings = 100
    min_counts = int(1e2)
    max_counts = int(1e5)
    num_relaxation_samples = int(1e7)

    # Choose some random bitstrings and their corresponding counts.
    category_bitstrings_list = []
    for _ in range(num_random_bitstrings):
      b = tf.constant([random.choice(all_bitstrings)], dtype=tf.int8)
      n = random.randint(min_counts, max_counts)
      tiled_b = tf.tile(b, [n, 1])
      category_bitstrings_list.append(tiled_b)
    category_bitstrings = tf.concat(category_bitstrings_list, 0)

    expected_probabilities = inference.relaxed_categorical_probabilities(category_bitstrings, all_bitstrings_tensor)

    relaxed_bitstrings = inference.relaxed_categorical_samples(category_bitstrings, num_relaxation_samples)
    unique_relaxed, _, counts_relaxed = utils.unique_bitstrings_with_counts(relaxed_bitstrings)
    num_unique_relaxed = tf.shape(unique_relaxed)[0]
    relaxed_probabilities = tf.cast(counts_relaxed, tf.float32) / num_relaxation_samples

    # Need to sort the probabilities.
    def relaxed_prob_choice(bitstring):
      """Returns the sample relaxed probability of the input."""
      tiled_bitstring = tf.tile(tf.expand_dims(bitstring, 0), [num_unique_relaxed, 1])
      relaxed_matches = tf.math.reduce_all(tf.math.equal(tiled_bitstring, unique_relaxed), 1)
      if tf.math.reduce_any(relaxed_matches):
        index = tf.where(relaxed_matches)[0][0]
        prob = relaxed_probabilities[index]
      else:
        prob = 0
      return prob

    actual_probabilities = tf.map_fn(relaxed_prob_choice, all_bitstrings_tensor, fn_output_signature=tf.float32)
    self.assertAllClose(actual_probabilities, expected_probabilities, atol=2e-4)


if __name__ == "__main__":
  print("Running ebm_utils_test.py ...")
  tf.test.main()
