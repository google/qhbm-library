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
"""Tools for modelling energy functions."""

import abc
from typing import List, Union

import cirq
import tensorflow as tf


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


class PauliBitstringDistribution(BitstringDistribution):
  """Augments BitstringDistribution with a Pauli Z representation."""

  @abc.abstractmethod
  def operator_shards(self, qubits):
    """Parameter independent Pauli Z strings to measure."""
    raise NotImplementedError()

  @abc.abstractmethod
  def operator_expectation(self, expectation_shards):
    """Computes the average energy given operator shard expectation values."""
    raise NotImplementedError()


class Bernoulli(PauliBitstringDistribution):
  """Tensor product of coin flip distributions."""

  def __init__(self,
               bits: List[int],
               initializer=tf.keras.initializers.RandomUniform(),
               name=None):
    """Initializes a Bernoulli distribution.

    Args:
      bits: Each entry is an index on which the distribution is supported.
      initializer: A `tf.keras.initializers.Initializer` which specifies how to
        initialize the values of the parameters.
    """
    super().__init__(name=name)
    if not isinstance(bits, list) or not all(isinstance(i, int) for i in bits):
      raise TypeError("`bits` must be a list of integers.")
    if len(set(bits)) != len(bits):
      raise ValueError("All entries of `bits` must be unique.")
    self._bits = bits
    self.kernel = self.add_weight(
        name=f'kernel',
        shape=[self.num_bits],
        initializer=initializer,
        trainable=True)

  @property
  def bits(self):
    return self._bits
 
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

  def operator_shards(self, qubits):
    return [
        cirq.PauliSum.from_pauli_strings(cirq.Z(qubits[i]))
        for i in range(self.num_bits)
    ]

  def operator_expectation(self, expectation_shards):
    return tf.reduce_sum(expectation_shards * self.kernel, -1)
