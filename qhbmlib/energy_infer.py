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

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from qhbmlib import energy_model


class BitstringDistribution(tfd.Distribution, abc.ABC):
  """Class for inference on BitstringEnergy.

  In contrast to implementations of `tfd.Distribution` in TFP, child classes
  are free define approximate or inefficient implementations of class methods.
  """

  def __init__(self, energy: energy_model.BitstringEnergy, name=None):
    """Initializes a BitstringDistribution.

    Args:
      energy: The parameterized energy function which defines this distribution
        via the equations of an energy based model.
      name: Optional python `str` name prefixed to Ops created by this class.
        Default: subclass name.
    """
    super().__init__(name=name)
    self.energy = energy

  @abc.abstractmethod
  def _sample_n(self, n):
    """Returns `n` samples approximately drawn from the EBM corresponding to
    `self.energy`."""
    raise NotImplementedError()

  @abc.abstractmethod
  def log_partition(self):
    """Returns an estimate of the log partition function."""
    raise NotImplementedError()

  @abc.abstractmethod
  def _entropy(self):
    """Returns an estimate of the entropy."""
    raise NotImplementedError()


class AnalyticDistribution(BitstringDistribution):
  """Uses an explicit categorical distribution to implement parent functions."""

  def __init__(self, energy: energy_model.BitstringEnergy, name=None):
    """Instantiates an AnalyticDistribution.

    Internally, this class saves all possible bitstrings as a tensor,
    whose energies are calculated relative to input distributions for sampling
    and other inference tasks.  Thus this distribution is limited to energy functions on

    Args:
      energy: The parameterized energy function which defines this distribution
        via the equations of an energy based model.
      name: Optional python `str` name prefixed to Ops created by this class.
        Default: subclass name.
    """
    super().__init__(energy, name=name)
    self._all_bitstrings = tf.constant(
      list(itertools.product([0, 1], repeat=self.energy.num_bits)),
      dtype=tf.int8)
    test_validity = self.inner_distribution(validate_args=True)
    del(test_validity)

  @property
  def inner_distribution(self, validate_args=False):
    """Returns a Categorical distribution."""
    return tfp.distributions.Categorical(logits=-1 * self.all_energies,
                                         validate_args=validate_args)

  @property
  def all_bitstrings(self):
    """Returns every bitstring."""

  @property
  def all_energies(self):
    """Returns the energy of every bitstring."""
    return self.energy(self.all_bitstrings)

  def _sample_n(self, n):
    """Returns `n` samples from the distribution defined by `self.energy`."""
    return tf.gather(self.all_bitstrings, self.inner_distribution.sample(n))

  def log_partition(self):
    """Returns the exact log partition function."""
    return tf.reduce_logsumexp(-1 * self.all_energies)

  def _entropy(self):
    """Returns the exact entropy."""
    return self.inner_distribution.entropy()


class BernoulliDistribution(BitstringDistribution):
  """Distribution for a Bernoulli defined by spin energies."""

  def __init__(self, energy: energy_model.BernoulliEnergy, name=None):
    """Initializes a BernoulliDistribution.

    Args:
      energy: The parameterized energy function which defines this distribution
        via the equations of an energy based model.
      name: Optional python `str` name prefixed to Ops created by this class.
        Default: subclass name.
    """
    super().__init__(energy, name=name)
    test_validity = self.inner_distribution(validate_args=True)
    del(test_validity)

  @property
  def inner_distribution(self, validate_args=False):
    """Returns a Bernoulli distribution."""
    return tfp.distributions.Bernoulli(
      logits=self.energy.logits,
      validate_args=validate_args,
      dtype=tf.int8)

  def _sample_n(self, n):
    """Returns `n` samples from the distribution defined by `self.energy`."""
    return self.inner_distribution.sample(n)

  def log_partition(self):
    r"""Returns the exact log partition function.

    For a single spin of energy $\theta$, the partition function is
    $$Z_\theta = \exp(\theta) + \exp(-\theta).$$
    Since each spin is independent, the total log partition function is
    the sum of the individual spin log partition functions.
    """
    thetas = 0.5 * self.energy.logits
    single_log_partitions = tf.math.log(tf.math.exp(thetas) + tf.math.exp(-1.0 * thetas))
    return tf.math.reduce_sum(single_log_partitions)

  def _entropy(self):
    """Returns the exact entropy."""
    return self.inner_distribution.entropy()
