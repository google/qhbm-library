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
"""Tools for defining energy functions."""

import abc
import collections
import itertools

import cirq
import tensorflow as tf
import tensorflow_probability as tfp

from qhbmlib import util


class BitstringDistribution(tf.keras.Model, abc.ABC):
  """Class for representing a probability distribution over bitstrings."""

  def __init__(self, name=None):
    super().__init__(name=name)

  @property
  def num_bits(self):
    return len(self.bits)

  @property
  @abc.abstractmethod
  def bits(self):
    """List of integer labels for the bits on which this distribution acts."""
    raise NotImplementedError()

  @property
  def trainable_variables(self):
    return super().trainable_variables

  @trainable_variables.setter
  @abc.abstractmethod
  def trainable_variables(self, value):
    raise NotImplementedError()

  @abc.abstractmethod
  def energy(self, bitstrings):
    raise NotImplementedError()


class Bernoulli(BitstringDistribution):
  """Tensor product of coin flip distributions."""

  def __init__(self,
               bits: list[int],
               initializer=tf.keras.initializers.RandomUniform(),
               name=None):
    super().__init__(name=name)
    self.bits = bits
    self.kernel = self.add_weight(
        name=f'kernel',
        shape=[self.num_bits],
        initializer=initializer,
        trainable=True)

  @property
  def bits(self):
    return self.bits
 
  @property
  def trainable_variables(self):
    return [self.kernel]

  @trainable_variables.setter
  def trainable_variables(self, value):
    self.kernel = value[0]

  def copy(self):
    bernoulli = Bernoulli(self.bits, name=self.name)
    bernoulli.kernel.assign(self.kernel)
    return bernoulli

  def energy(self, bitstrings):
    return tf.reduce_sum(
        tf.cast(1 - 2 * bitstrings, tf.float32) * self.kernel, -1)
