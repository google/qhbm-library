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

import tensorflow as tf
import tensorflow_probability as tfp

from qhbmlib import energy_model
from qhbmlib import util


class BitstringSampler(tf.keras.Model, abc.ABC):
  """Class for sampling from BitstringDistributions."""

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
    """See description in base class."""
    samples = tfp.distributions.Bernoulli(
        logits=2 * dist.kernel, dtype=tf.int8).sample(num_samples)
    return util.unique_bitstrings_with_counts(samples)
  

class AnalyticSampler(BitstringSampler):
  """Sampler which calculates all probabilities and samples as categorical."""

  def __init__(self, num_bits: int):
    """Instantiates an AnalyticSampler.

    Args:
      num_bits: number of bits on which the BitstringDistribution to be sampled
        is supported.
    """
    self._all_bitstrings = tf.constant(
      list(itertools.product([0, 1], repeat=num_bits)), dtype=tf.int8)

  def _energies(self, dist: energy_model.BitstringDistribution):
    """Returns the current energy of every bitstring."""
    return dist.energy(self._all_bitstrings)

  def sample(self, dist, num_samples):
    samples = tf.gather(
      self._all_bitstrings,
      tfp.distributions.Categorical(logits=-1 *
                                    self._energies(dist)).sample(num_samples))
    return util.unique_bitstrings_with_counts(samples)
