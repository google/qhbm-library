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
"""Tests for qhbmlib.utils"""

from absl.testing import parameterized

import tensorflow as tf

from qhbmlib import utils
from tests import test_util


class SqueezeTest(tf.test.TestCase):
  """Tests the Squeeze layer."""

  @test_util.eager_mode_toggle
  def test_layer(self):
    """Confirms the layer squeezes correctly."""
    inputs = tf.constant([[[1]], [[2]]])
    expected_axis = 1
    expected_outputs = tf.constant([[1], [2]])
    actual_layer = utils.Squeeze(expected_axis)

    @tf.function
    def wrapper(inputs):
      return actual_layer(inputs)

    actual_outputs = wrapper(inputs)
    self.assertAllEqual(actual_outputs, expected_outputs)


class WeightedAverageTest(tf.test.TestCase):
  """Tests the weighted average function."""

  @test_util.eager_mode_toggle
  def test_explicit(self):
    """Uses explicit averaging to test the function."""
    raw_counts = [37, 5]
    count_sum = sum(raw_counts)
    counts = tf.constant(raw_counts, dtype=tf.int32)
    raw_values = [[2.7, -5.9], [-0.5, 3.2]]
    values = tf.constant(raw_values, dtype=tf.float32)
    expected_average = [(raw_counts[0] / count_sum) * raw_values[0][0] +
                        (raw_counts[1] / count_sum) * raw_values[1][0],
                        (raw_counts[0] / count_sum) * raw_values[0][1] +
                        (raw_counts[1] / count_sum) * raw_values[1][1]]

    @tf.function
    def wrapper(counts, values):
      return utils.weighted_average(counts, values)

    actual_average = wrapper(counts, values)
    self.assertAllClose(actual_average, expected_average)

    # test 1D values
    raw_values_1d = [-5.6, 3.1]
    values_1d = tf.constant(raw_values_1d, dtype=tf.float32)
    expected_average = (raw_counts[0] / count_sum) * raw_values_1d[0] + (
        raw_counts[1] / count_sum) * raw_values_1d[1]
    actual_average = wrapper(counts, values_1d)
    self.assertAllClose(actual_average, expected_average)


class UniqueBitstringsWithCountsTest(parameterized.TestCase, tf.test.TestCase):
  """Test unique_bitstrings_with_counts and its inverse."""

  @parameterized.parameters(
      [{
          "bit_type": bit_type,
          "out_idx": out_idx
      }
       for bit_type in
       [tf.dtypes.int8, tf.dtypes.int32, tf.dtypes.int64, tf.dtypes.float32]
       for out_idx in [tf.dtypes.int32, tf.dtypes.int64]])
  @test_util.eager_mode_toggle
  def test_identity(self, bit_type, out_idx):
    """Case when all entries are unique."""
    expected_bitstrings = tf.constant([[1], [0]], dtype=bit_type)
    expected_y = expected_bitstrings
    expected_idx = tf.constant([0, 1], dtype=out_idx)
    expected_count = tf.constant([1, 1], dtype=out_idx)

    unique_bitstrings_with_counts_wrapper = tf.function(
        utils.unique_bitstrings_with_counts)
    actual_y, actual_idx, actual_count = unique_bitstrings_with_counts_wrapper(
        expected_bitstrings, out_idx=out_idx)
    self.assertAllEqual(actual_y, expected_y)
    self.assertAllEqual(actual_idx, expected_idx)
    self.assertAllEqual(actual_count, expected_count)

    expand_unique_results_wrapper = tf.function(utils.expand_unique_results)
    actual_bitstrings = expand_unique_results_wrapper(actual_y, actual_idx)
    self.assertAllEqual(actual_bitstrings, expected_bitstrings)

  @test_util.eager_mode_toggle
  def test_short(self):
    """Case when bitstrings are length 1."""
    expected_bitstrings = tf.constant([
        [1],
        [0],
        [0],
        [1],
        [1],
        [0],
        [1],
        [1],
    ],)
    expected_y = tf.constant([[1], [0]])
    expected_idx = tf.constant([
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        0,
    ])
    expected_count = tf.constant([5, 3])

    unique_bitstrings_with_counts_wrapper = tf.function(
        utils.unique_bitstrings_with_counts)
    actual_y, actual_idx, actual_count = unique_bitstrings_with_counts_wrapper(
        expected_bitstrings)
    self.assertAllEqual(actual_y, expected_y)
    self.assertAllEqual(actual_idx, expected_idx)
    self.assertAllEqual(actual_count, expected_count)

    expand_unique_results_wrapper = tf.function(utils.expand_unique_results)
    actual_bitstrings = expand_unique_results_wrapper(actual_y, actual_idx)
    self.assertAllEqual(actual_bitstrings, expected_bitstrings)

    compressed_outputs = tf.random.uniform([2], dtype=tf.float32)
    expected_outputs = tf.constant(
        [compressed_outputs.numpy()[j] for j in expected_idx.numpy().tolist()])
    actual_outputs = expand_unique_results_wrapper(compressed_outputs,
                                                   actual_idx)
    self.assertAllEqual(actual_outputs, expected_outputs)

  @test_util.eager_mode_toggle
  def test_long(self):
    """Case when bitstrings are of length > 1."""
    expected_bitstrings = tf.constant([
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 0, 1],
    ],)
    expected_y = tf.constant([[1, 0, 1], [1, 1, 1], [0, 1, 1]])
    expected_idx = tf.constant([0, 1, 2, 0, 1, 2, 0, 0])
    expected_count = tf.constant([4, 2, 2])

    unique_bitstrings_with_counts_wrapper = tf.function(
        utils.unique_bitstrings_with_counts)
    actual_y, actual_idx, actual_count = unique_bitstrings_with_counts_wrapper(
        expected_bitstrings)
    self.assertAllEqual(actual_y, expected_y)
    self.assertAllEqual(actual_idx, expected_idx)
    self.assertAllEqual(actual_count, expected_count)

    expand_unique_results_wrapper = tf.function(utils.expand_unique_results)
    actual_bitstrings = expand_unique_results_wrapper(actual_y, actual_idx)
    self.assertAllEqual(actual_bitstrings, expected_bitstrings)

    compressed_outputs = tf.random.uniform([3], dtype=tf.float32)
    expected_outputs = tf.constant(
        [compressed_outputs.numpy()[j] for j in expected_idx.numpy().tolist()])
    actual_outputs = expand_unique_results_wrapper(compressed_outputs,
                                                   actual_idx)
    self.assertAllEqual(actual_outputs, expected_outputs)


if __name__ == "__main__":
  print("Running utils_test.py ...")
  tf.test.main()
