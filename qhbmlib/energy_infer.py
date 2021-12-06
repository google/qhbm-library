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
"""Tools for inference on energy functions."""

import abc
import itertools
from typing import List, Union

import tensorflow as tf
import tensorflow_probability as tfp

from qhbmlib import energy_model
from qhbmlib import util


class BitstringSampler(tf.keras.Model, abc.ABC):
  """Class for sampling from BitstringDistributions."""

  def __init__(self,
               num_bits: int,
               name: Union[None, str]=None):
    """Initializes a BitstringDistribution.

    Args:
      num_bits: Number of bits in drawn samples.
      name: Optional name for the model.
    """
    super().__init__(name=name)
    if not isinstance(num_bits, int):
      raise TypeError("`num_bits` must be an integer.")
    if num_bits < 1:
      raise ValueError("`num_bits` must be a positive integer.")
    self._num_bits = num_bits

  @property
  def num_bits(self):
    return self._num_bits

  @abc.abstractmethod
  def sample(self, dist: energy_model.BitstringDistribution, num_samples):
    """Draw `num_samples` samples from `dist`.

    Returns a pair `(bitstrings, counts)` where `bitstrings` is a 2D `tf.int8`
    tensor of unique bitstrings, and `counts` is a 1D `tf.int32` tensor such
    that `counts[i]` is the number of times `bitstrings[i]` was sampled.

    We also have `sum(tf.shape(counts)) == num_samples`.
    """
    raise NotImplementedError()

  def call(self, dist: energy_model.BitstringDistribution, num_samples):
    return self.sample(dist, num_samples)


class BernoulliSampler(BitstringSampler):
  """Sampler for Bernoulli distributions."""

  def sample(self, dist: energy_model.Bernoulli, num_samples: int):
    """See base class docstring."""
    samples = tfp.distributions.Bernoulli(
        logits=2 * dist.kernel, dtype=tf.int8).sample(num_samples)
    return util.unique_bitstrings_with_counts(samples)
  

class AnalyticSampler(BitstringSampler):
  """Sampler which calculates all probabilities and samples as categorical."""

  def __init__(self, num_bits, name=None):
    """Instantiates an AnalyticSampler.

    Internally, this class saves all possible bitstrings as a tensor,
    whose energies are calculated relative to input distributions for sampling.
    """
    super().__init__(num_bits, name=name)
    self._all_bitstrings = tf.constant(
      list(itertools.product([0, 1], repeat=num_bits)), dtype=tf.int8)

  def _energies(self, dist: energy_model.BitstringDistribution):
    """Returns the current energy of every bitstring."""
    return dist.energy(self._all_bitstrings)

  def sample(self, dist, num_samples):
    """See base class docstring."""
    samples = tf.gather(
      self._all_bitstrings,
      tfp.distributions.Categorical(logits=-1 *
                                    self._energies(dist)).sample(num_samples))
    return util.unique_bitstrings_with_counts(samples)
