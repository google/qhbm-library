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
"""Tools for inference on energy functions represented by a BitstringEnergy."""

import abc
import itertools
from typing import Union

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from qhbmlib import energy_model


class InferenceLayer(tf.keras.layers.Layer, abc.ABC):
  """Sets the methods required for inference on BitstringEnergy objects."""

  def __init__(self,
               energy: energy_model.BitstringEnergy,
               name: Union[None, str] = None):
    """Initializes an InferenceLayer.

    Args:
      energy: The parameterized energy function which defines this distribution
        via the equations of an energy based model.
      name: Optional name for the model.
    """
    super().__init__(name=name)
    self._energy = energy

  @property
  def energy(self):
    """The energy model on which this layer performs inference."""
    return self._energy

  @abc.abstractmethod
  def infer(self):
    """Do the work to ready this layer for use.

    This should be called each time the underlying model in
    `self.energy` is updated.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def sample(self, n):
    """Returns samples from the EBM corresponding to `self.energy`.

    This can be an approximate sampling.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def entropy(self):
    """Returns an estimate of the entropy."""
    raise NotImplementedError()

  @abc.abstractmethod
  def log_partition(self):
    """Returns an estimate of the log partition function."""
    raise NotImplementedError()

  def build(self, input_shape):
    """Builds the internal energy function.

    `input_shape` is unused because it is known to be `[]`, since calls are
    given a scalar, the number of samples to draw from the distribution.
    """
    del input_shape
    self.energy.build([None, self.energy.num_bits])

  def call(self, inputs):
    """Returns the number of samples specified in the inputs."""
    return self.sample(inputs)


class AnalyticInferenceLayer(InferenceLayer):
  """Uses an explicit categorical distribution to implement parent functions."""

  def __init__(self,
               energy: energy_model.BitstringEnergy,
               name: Union[None, str] = None):
    """Initializes an AnalyticInferenceLayer.

    Internally, this class saves all possible bitstrings as a tensor, whose
    energies are calculated relative to an input energy function for sampling
    and other inference tasks.

    Args:
      energy: The parameterized energy function which defines this distribution
        via the equations of an energy based model.
      name: Optional name for the model.
    """
    super().__init__(energy, name=name)
    self._all_bitstrings = tf.constant(
        list(itertools.product([0, 1], repeat=self.energy.num_bits)),
        dtype=tf.int8)
    self._dist_realization = tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: tfd.Categorical(logits=-1 * t))
    self._current_dist = None

  @property
  def all_bitstrings(self):
    """Returns every bitstring."""
    return self._all_bitstrings

  @property
  def all_energies(self):
    """Returns the energy of every bitstring."""
    return self.energy(self.all_bitstrings)

  @property
  def current_dist(self):
    """Bernoulli distribution set during last call to `self.infer`."""
    return self._current_dist

  def infer(self):
    """See base class docstring."""
    x = tf.squeeze(self.all_energies)
    self._current_dist = self._dist_realization(x)

  def sample(self, n):
    """See base class docstring"""
    return tf.gather(self.all_bitstrings, self._current_dist.sample(n), axis=0)

  def entropy(self):
    """See base class docstring"""
    return self._current_dist.entropy()

  def log_partition(self):
    """See base class docstring"""
    # TODO(#115)
    return tf.reduce_logsumexp(self._current_dist.logits_parameter())

  def call(self, inputs):
    if self._current_dist is None:
      self.infer()
    if inputs is None:
      return self._current_dist
    else:
      return self.sample(inputs)


class BernoulliInferenceLayer(InferenceLayer):
  """Manages inference for a Bernoulli defined by spin energies."""

  def __init__(self,
               energy: energy_model.BitstringEnergy,
               name: Union[None, str] = None):
    """Initializes a BernoulliInferenceLayer.

    Args:
      energy: The parameterized energy function which defines this distribution
        via the equations of an energy based model.
      name: Optional name for the model.
    """
    super().__init__(energy, name=name)
    self._dist_realization = tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: tfd.Bernoulli(logits=t, dtype=tf.int8))
    self._current_dist = None

  @property
  def current_dist(self):
    """Categorical distribution set during last call to `self.infer`."""
    return self._current_dist

  def infer(self):
    """See base class docstring."""
    self._current_dist = self._dist_realization(self.energy.logits)

  def sample(self, n):
    """See base class docstring"""
    return self._current_dist.sample(n)

  def entropy(self):
    """Returns the exact entropy.

    The total entropy of a set of spins is the sum of each individual spin's
    entropies.
    """
    return tf.reduce_sum(self._current_dist.entropy())

  def log_partition(self):
    r"""Returns the exact log partition function.

    For a single spin of energy $\theta$, the partition function is
    $$Z_\theta = \exp(\theta) + \exp(-\theta).$$
    Since each spin is independent, the total log partition function is
    the sum of the individual spin log partition functions.
    """
    thetas = 0.5 * self.energy.logits
    single_log_partitions = tf.math.log(
        tf.math.exp(thetas) + tf.math.exp(-1.0 * thetas))
    return tf.math.reduce_sum(single_log_partitions)

  def call(self, inputs):
    if self._current_dist is None:
      self.infer()
    if inputs is None:
      return self._current_dist
    else:
      return self.sample(inputs)
