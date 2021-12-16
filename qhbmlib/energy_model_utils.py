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
"""Utilities for the energy_model module."""

import itertools
from typing import List

import tensorflow as tf


def check_bits(bits: List[int]) -> List[int]:
  """Confirms the input is a valid bit index list."""
  if len(set(bits)) != len(bits):
    raise ValueError("All entries of `bits` must be unique.")
  return bits


def check_order(order: int) -> int:
  """Confirms the input is a valid parity order."""
  if not isinstance(order, int):
    raise TypeError("`order` must be an integer.")
  if order <= 0:
    raise ValueError("`order` must be greater than zero.")
  return order


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


class SpinsFromBitstrings(tf.keras.layers.Layer):
  """Simple layer taking bits to spins."""

  def __init__(self):
    """Initializes a SpinsFromBitstrings."""
    super().__init__(trainable=False)

  def call(self, inputs):
    """Returns the spins corresponding to the input bitstrings.

    Note that this maps |0> -> +1 and |1> -> -1.  This is in accordance with
    the usual interpretation of the Bloch sphere.
    """
    return tf.cast(1 - 2 * inputs, tf.float32)


class VariableDot(tf.keras.layers.Layer):
  """Utility layer for dotting input with a same-sized variable."""

  def __init__(self,
               initializer: tf.keras.initializers.Initializer = tf.keras
               .initializers.RandomUniform()):
    """Initializes a VariableDot layer.

    Args:
      initializer: A `tf.keras.initializers.Initializer` which specifies how to
        initialize the values of the parameters.
    """
    super().__init__()
    self._initializer = initializer

  def build(self, input_shape):
    """Initializes the internal variables."""
    self.kernel = tf.Variable(
        initial_value=self._initializer((input_shape[-1],)),
        name="kernel",
        trainable=True)

  def call(self, inputs):
    """Returns the dot product between the inputs and this layer's variables."""
    return tf.reduce_sum(inputs * self.kernel, -1)


class Parity(tf.keras.layers.Layer):
  """Computes the parities of input spins."""

  def __init__(self, bits: List[int], order: int):
    """Initializes a Parity layer."""
    super().__init__(trainable=False)
    bits = check_bits(bits)
    order = check_order(order)
    indices_list = []
    for i in range(1, order + 1):
      combos = itertools.combinations(range(len(bits)), i)
      indices_list.extend(list(combos))
    self.indices = tf.ragged.stack(indices_list)
    self.num_terms = len(indices_list)

  def call(self, inputs):
    """Returns a batch of parities corresponding to the input bitstrings."""
    parities_t = tf.zeros([self.num_terms, tf.shape(inputs)[0]])
    for i in tf.range(self.num_terms):
      parity = tf.reduce_prod(tf.gather(inputs, self.indices[i], axis=-1), -1)
      parities_t = tf.tensor_scatter_nd_update(parities_t, [[i]], [parity])
    return tf.transpose(parities_t)
