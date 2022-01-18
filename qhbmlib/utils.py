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


def weighted_average(counts: tf.Tensor, values: tf.Tensor):
  """Returns the weighted average of input values.

  Row `i` of `values` is multiplied by `counts[i]`, resulting in a weighted
  version of values; the mean is then taken across the first dimension.

  Args:
    counts: Non-negative integers of shape [batch_size].
    values: Floats of shape [batch_size, n].
  """
  expanded_counts = tf.expand_dims(tf.cast(counts, tf.float32), -1)
  weighted_values = expanded_counts * values
  return tf.reduce_sum(weighted_values, 0) / tf.reduce_sum(expanded_counts)


def unique_bitstrings_with_counts(bitstrings, out_idx=tf.dtypes.int32):
  """Extract the unique bitstrings in the given bitstring tensor.

    Args:
      bitstrings: 2-D `tf.Tensor`, interpreted as a list of bitstrings.
      out_idx: An optional `tf.DType` from: `tf.int32`, `tf.int64`. Defaults to
        `tf.int32`.  Specifies the type of `count` output.

    Returns:
      y: 2-D `tf.Tensor` of same dtype as `bitstrings`, containing the unique
        0-axis entries of `bitstrings`.
      count: 1-D `tf.Tensor` of dtype `out_idx` such that `count[i]` is the
        number of occurences of `y[i]` in `bitstrings`.
  """
  y, _, counts = tf.raw_ops.UniqueWithCountsV2(
      x=bitstrings, axis=[0], out_idx=out_idx)
  return y, counts
