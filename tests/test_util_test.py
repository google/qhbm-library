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
"""Tests for the test_util module."""

import tensorflow as tf

from tests import test_util


class EagerModeToggle(tf.test.TestCase):
  """Tests eager_mode_toggle."""

  def test_eager_mode_toggle(self):
    """Ensure eager mode really gets toggled."""

    def fail_in_eager():
      """Raises AssertionError if run in eager."""
      if tf.config.functions_run_eagerly():
        raise AssertionError()

    def fail_out_of_eager():
      """Raises AssertionError if run outside of eager."""
      if not tf.config.functions_run_eagerly():
        raise AssertionError()

    with self.assertRaises(AssertionError):
      test_util.eager_mode_toggle(fail_in_eager)()

    # Ensure eager mode still turned off even though exception was raised.
    self.assertFalse(tf.config.functions_run_eagerly())

    with self.assertRaises(AssertionError):
      test_util.eager_mode_toggle(fail_out_of_eager)()
