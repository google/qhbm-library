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
"""Tests for the energy_model_utils module."""

from absl.testing import parameterized

import tensorflow as tf
from tensorflow_probability.python.internal import test_util as tfp_test_util

from qhbmlib import energy_model_utils


@tfp_test_util.test_all_tf_execution_regimes
class CheckHelpersTest(parameterized.TestCase, tf.test.TestCase):
  """Tests the input check functions."""

  def test_check_bits(self):
    """Confirms bad inputs are caught and good inputs pass through."""
    expected_bits = [1, 6, 222, 13223]
    actual_bits = energy_model_utils.check_bits(expected_bits)
    self.assertAllEqual(actual_bits, expected_bits)
    with self.assertRaisesRegex(ValueError, expected_regex="must be unique"):
      _ = energy_model_utils.check_bits([1, 1])

  def test_check_order(self):
    """Confirms bad inputs are caught and good inputs pass through."""
    expected_order = 5
    actual_order = energy_model_utils.check_order(expected_order)
    self.assertEqual(actual_order, expected_order)
    with self.assertRaisesRegex(ValueError, expected_regex="greater than zero"):
      _ = energy_model_utils.check_order(0)


@tfp_test_util.test_all_tf_execution_regimes
class SpinsFromBitstringsTest(parameterized.TestCase, tf.test.TestCase):
  """Tests the SpinsFromBitstrings layer."""

  def test_layer(self):
    """Confirms the layer outputs correct spins."""
    test_layer = energy_model_utils.SpinsFromBitstrings()
    test_bitstrings = tf.constant([[1, 0, 0], [0, 1, 0]])
    expected_spins = tf.constant([[-1, 1, 1], [1, -1, 1]])
    actual_spins = test_layer(test_bitstrings)
    self.assertAllEqual(actual_spins, expected_spins)


# TODO(#136): Get this error in the graph mode decorator test:
#             ERROR    tensorflow:test_util.py:1911 Could not find
#             variable variable_dot/kernel.
class VariableDotTest(parameterized.TestCase, tf.test.TestCase):
  """Tests the VariableDot layer."""

  def setUp(self):
    """Initializes test objects."""
    super().setUp()
    self.constant = 2.5
    self.actual_layer_constant = energy_model_utils.VariableDot(
        tf.keras.initializers.Constant(self.constant))
    self.actual_layer_constant.build([None, 3])

    self.actual_layer_default = energy_model_utils.VariableDot()
    self.actual_layer_default.build([None, 3])

    self.inputs_list = [[1, 5, 9]]
    self.inputs = tf.constant(self.inputs_list, dtype=tf.float32)

  @tfp_test_util.test_all_tf_execution_regimes
  def test_layer(self):
    """Confirms the layer dots with inputs correctly."""
    actual_outputs = self.actual_layer_constant(self.inputs)
    expected_outputs = tf.math.reduce_sum(self.inputs * self.constant, -1)
    self.assertAllEqual(actual_outputs, expected_outputs)

    actual_outputs = self.actual_layer_default(self.inputs)
    expected_outputs = []
    for inner in self.inputs_list:
      inner_sum = []
      for i, k in zip(inner, self.actual_layer_default.kernel.numpy()):
        inner_sum.append(i * k)
      expected_outputs.append(sum(inner_sum))
    self.assertAllClose(actual_outputs, expected_outputs)


@tfp_test_util.test_all_tf_execution_regimes
class ParityTest(parameterized.TestCase, tf.test.TestCase):
  """Tests the Parity layer."""

  def test_layer(self):
    """Confirms the layer outputs correct parities."""
    inputs = tf.constant([[-1, 1, -1, -1]])
    bits = [1, 2, 3, 4]
    order = 3
    actual_layer = energy_model_utils.Parity(bits, order)

    expected_indices = [[0], [1], [2], [3], [0, 1], [0, 2], [0, 3], [1, 2],
                        [1, 3], [2, 3], [0, 1, 2], [0, 1, 3], [0, 2, 3],
                        [1, 2, 3]]
    expected_num_terms = 14
    self.assertAllEqual(actual_layer.indices, expected_indices)
    self.assertAllEqual(actual_layer.num_terms, expected_num_terms)

    expected_outputs = tf.constant([[-1, 1, -1, -1]  # single bit parities
                                    + [-1, 1, 1, -1, -1, 1]  # two bit parities
                                    + [1, 1, -1, 1]])  # three bit parities
    actual_outputs = actual_layer(inputs)
    self.assertAllEqual(actual_outputs, expected_outputs)


if __name__ == "__main__":
  print("Running energy_model_utils_test.py ...")
  tf.test.main()
