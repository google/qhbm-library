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

from qhbmlib import energy_model
from qhbmlib import util


class BitstringSampler(abc.ABC):
  """Class for sampling from BitstringDistributions."""

  @abc.abstractmethod
  def sample(self, dist, num_samples):
    """Returns `num_samples` samples from `dist`."""
    raise NotImplementedError()


class Bernoulli(BitstringSampler):
  """Sampler for Bernoulli distributions."""

  def sample(self, dist: energy_model.Bernoulli, num_samples: int):
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

  def sample(
