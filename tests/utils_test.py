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
"""Tests for the utils module."""

import tensorflow as tf

from qhbmlib import utils


class SqueezeTest(tf.test.TestCase):
  """Tests the Squeeze layer."""

  def test_layer(self):
    """Confirms the layer squeezes correctly."""
    inputs = tf.constant([[[1]], [[2]]])
    expected_axis = 1
    expected_outputs = tf.constant([[1], [2]])
    actual_layer = utils.Squeeze(expected_axis)
    actual_outputs = actual_layer(inputs)
    self.assertAllEqual(actual_outputs, expected_outputs)


if __name__ == "__main__":
  print("Running utils_test.py ...")
  tf.test.main()
