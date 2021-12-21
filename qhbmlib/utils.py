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
"""Utilities used across more than one file."""

import tensorflow as tf


class Squeeze(tf.keras.layers.Layer):
  """Wraps tf.squeeze in a Keras Layer."""

  def __init__(self, axis=None):
    """Initializes a Squeeze layer.

    Args:
      axis: An optional list of ints. Defaults to []. If specified, only
        squeezes the dimensions listed. The dimension index starts at 0. It is
        an error to squeeze a dimension that is not 1. Must be in the range
        [-rank(input), rank(input)). Must be specified if input is
        a RaggedTensor.
    """
    super().__init__()
    if axis is None:
      axis = []
    self._axis = axis

  def call(self, inputs):
    """Applies tf.squeeze to the inputs."""
    return tf.squeeze(inputs, axis=self._axis)
