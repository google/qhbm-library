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
import functools
import itertools
from typing import Union

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from qhbmlib import energy_model
from qhbmlib import utils


def preface_inference(f):
  """Wraps given function with things to run before every inference call.

  Args:
    f: The method of `EnergyInference` to wrap.

  Returns:
    wrapper: The wrapped function.
  """

  @functools.wraps(f)
  def wrapper(self, *args, **kwargs):
    self._preface_inference()  # pylint: disable=protected-access
    return f(self, *args, **kwargs)

  return wrapper


class EnergyInferenceBase(tf.keras.layers.Layer, abc.ABC):
  r"""Defines the interface for inference on BitstringEnergy objects.

  Let $E$ be the energy function defined by a given `BitstringEnergy`, and let
  $X$ be the set of bitstrings in the domain of $E$.  Associated with $E$ is
  a probability distribution
  $$p(x) = \frac{e^{-E(x)}}{\sum_{y\in X} e^{-E(y)}},$$
  which we call the Energy Based Model (EBM) associated with $E$.  Inference
  in this class means estimating quantities of interest relative to the EBM.
  """

  def __init__(self,
               initial_seed: Union[None, tf.Tensor] = None,
               name: Union[None, str] = None):
    """Initializes an EnergyInferenceBase.

    Args:
      initial_seed: PRNG seed; see tfp.random.sanitize_seed for details. This
        seed will be used in the `sample` method.  If None, the seed is updated
        after every inference call.  Otherwise, the seed is fixed.
      name: Optional name for the model.
    """
    super().__init__(name=name)
    if initial_seed is None:
      self._update_seed = tf.Variable(True, trainable=False)
    else:
      self._update_seed = tf.Variable(False, trainable=False)
    self._seed = tf.Variable(
        tfp.random.sanitize_seed(initial_seed), trainable=False)
    self._first_inference = tf.Variable(True, trainable=False)

  @property
  def energy(self):
    """The energy function which sets the probabilities for this EBM."""
    return self._energy

  @property
  def seed(self):
    """Current TFP compatible seed controlling sampling behavior.

    PRNG seed; see tfp.random.sanitize_seed for details. This seed will be used
    in the `sample` method.  If None, the seed is updated after every inference
    call.  Otherwise, the seed is fixed.
    """
    return self._seed

  @seed.setter
  def seed(self, initial_seed: Union[None, tf.Tensor]):
    """Sets a new value of the random seed.

    Args:
      initial_seed: see `self.seed` for details.
    """
    if initial_seed is None:
      self._update_seed.assign(True)
    else:
      self._update_seed.assign(False)
    self._seed.assign(tfp.random.sanitize_seed(initial_seed))

  def _preface_inference(self):
    """Things all energy inference methods do before proceeding.

    Called by `preface_inference` before the wrapped inference method.
    Currently includes:
      - run `self.infer` if this is the first call of a wrapped function
      - change the seed if not set by the user during initialization

    Note: subclasses should take care to call the superclass method.
    """
    if self._first_inference:
      self.infer(self.energy)
      self._first_inference.assign(False)
    if self._update_seed:
      new_seed, _ = tfp.random.split_seed(self.seed)
      self._seed.assign(new_seed)

  @preface_inference
  def call(self, inputs, *args, **kwargs):
    """Calls this layer on the given inputs."""
    return self._call(inputs, *args, **kwargs)

  @preface_inference
  def entropy(self):
    """Returns an estimate of the entropy."""
    return self._entropy()

  @preface_inference
  def expectation(self, function):
    """Returns an estimate of the expectation value of the given function.

    Args:
      function: Mapping from a 2D tensor of bitstrings to a possibly nested
        structure.  The structure must have atomic elements all of which are
        float tensors with the same batch size as the input bitstrings.
    """
    return self._expectation(function)

  @preface_inference
  def log_partition(self):
    """Returns an estimate of the log partition function."""
    return self._log_partition()

  @preface_inference
  def sample(self, num_samples: int):
    """Returns samples from the EBM corresponding to `self.energy`.

    Args:
      num_samples: Number of samples to draw from the EBM.
    """
    return self._sample(num_samples)

  @abc.abstractmethod
  def _call(self, inputs, *args, **kwargs):
    """Default implementation wrapped by `self.call`."""
    raise NotImplementedError()

  @abc.abstractmethod
  def _entropy(self):
    """Default implementation wrapped by `self.entropy`."""
    raise NotImplementedError()

  @abc.abstractmethod
  def _expectation(self, function):
    """Default implementation wrapped by `self.expectation`."""
    raise NotImplementedError()

  @abc.abstractmethod
  def _log_partition(self):
    """Default implementation wrapped by `self.log_partition`."""
    raise NotImplementedError()

  @abc.abstractmethod
  def _sample(self, num_samples: int):
    """Default implementation wrapped by `self.sample`."""
    raise NotImplementedError()

  @abc.abstractmethod
  def infer(self, energy: energy_model.BitstringEnergy):
    """Do the work to ready this layer for use.

    This should be called each time the underlying model is updated.

    Args:
      energy: The parameterized energy function which defines this distribution
        via the equations of an energy based model.
    """
    raise NotImplementedError()


class EnergyInference(EnergyInferenceBase):
  """Provides some default method implementations."""

  def __init__(self,
               num_expectation_samples: int,
               initial_seed: Union[None, tf.Tensor] = None,
               name: Union[None, str] = None):
    """Initializes an EnergyInference.

    Args:
      num_expectation_samples: Number of samples to draw and use for estimating
        the expectation value.
      initial_seed: PRNG seed; see tfp.random.sanitize_seed for details. This
        seed will be used in the `sample` method.  If None, the seed is updated
        after every inference call.  Otherwise, the seed is fixed.
      name: Optional name for the model.
    """
    super().__init__(initial_seed, name)
    self.num_expectation_samples = num_expectation_samples

  def _expectation(self, function):
    """Default implementation wrapped by `self.expectation`.

    Estimates an expectation value using sample averaging.
    """

    @tf.custom_gradient
    def _inner_expectation():
      """Enables derivatives."""
      samples = tf.stop_gradient(self.sample(self.num_expectation_samples))
      bitstrings, _, counts = utils.unique_bitstrings_with_counts(samples)

      # TODO(#157): try to parameterize the persistence.
      with tf.GradientTape() as values_tape:
        # Adds variables in `self.energy` to `variables` argument of `grad_fn`.
        values_tape.watch(self.energy.trainable_variables)
        values = function(bitstrings)
        average_of_values = tf.nest.map_structure(
            lambda x: utils.weighted_average(counts, x), values)

      def grad_fn(*upstream, variables):
        """See equation A5 in the QHBM paper appendix for details.

        # TODO(#119): confirm equation number.
        """
        function_grads = values_tape.gradient(
            average_of_values,
            variables,
            output_gradients=upstream,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)

        flat_upstream = tf.nest.flatten(upstream)
        flat_values = tf.nest.flatten(values)
        combined_flat = tf.nest.map_structure(lambda x, y: x * y, flat_upstream,
                                              flat_values)
        combined_flat_sum = tf.nest.map_structure(
            lambda x: tf.map_fn(tf.reduce_sum, x), combined_flat)
        combined_sum = tf.reduce_sum(tf.stack(combined_flat_sum), 0)
        average_of_combined_sum = utils.weighted_average(counts, combined_sum)

        # Compute grad E terms.
        with tf.GradientTape() as tape:
          energies = self.energy(bitstrings)
        energies_grads = tape.jacobian(
            energies,
            variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
        average_of_energies_grads = tf.nest.map_structure(
            lambda x: utils.weighted_average(counts, x), energies_grads)

        product_of_averages = tf.nest.map_structure(
            lambda x: x * average_of_combined_sum, average_of_energies_grads)

        products = tf.nest.map_structure(
            lambda x: tf.einsum("i...,i->i...", x, combined_sum),
            energies_grads)
        average_of_products = tf.nest.map_structure(
            lambda x: utils.weighted_average(counts, x), products)

        # Note: upstream gradient is already a coefficient in poa, aop, and fg.
        return tuple(), [
            poa - aop + fg for poa, aop, fg in zip(
                product_of_averages, average_of_products, function_grads)
        ]

      return average_of_values, grad_fn

    return _inner_expectation()

  def _log_partition(self):
    """Default implementation wrapped by `self.log_partition`."""

    @tf.custom_gradient
    def _inner_log_partition():
      """Wraps forward pass computaton."""
      result = self._log_partition_forward_pass()
      # Adds variables in `self.energy` to `variables` argument of `grad_fn`.
      _ = [tf.identity(x) for x in self.energy.trainable_variables]
      grad_fn = self._log_partition_grad_generator()
      return result, grad_fn

    return _inner_log_partition()

  @abc.abstractmethod
  def _log_partition_forward_pass(self):
    """Returns approximation to the log partition function."""
    raise NotImplementedError()

  def _log_partition_grad_generator(self):
    """Returns default estimator for the log partition function derivative."""

    def grad_fn(upstream, variables):
      """See equation C2 in the appendix.  TODO(#119)"""

      def energy_grad(bitstrings):
        """Calculates the derivative with respect to the current variables."""
        with tf.GradientTape() as tape:
          energies = self.energy(bitstrings)
        jac = tape.jacobian(
            energies,
            variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return jac

      energy_grad_expectation_list = self.expectation(energy_grad)
      return tuple(), [
          upstream * (-1.0 * ege) for ege in energy_grad_expectation_list
      ]

    return grad_fn


class AnalyticEnergyInference(EnergyInference):
  """Uses an explicit categorical distribution to implement parent functions."""

  def __init__(self,
               num_bits: int,
               num_expectation_samples: int,
               initial_seed: Union[None, tf.Tensor] = None,
               name: Union[None, str] = None):
    """Initializes an AnalyticEnergyInference.

    Internally, this class saves all possible bitstrings as a tensor, whose
    energies are calculated relative to an input energy function for sampling
    and other inference tasks.

    Args:
      num_bits: Number of bits on which this layer acts.
      num_expectation_samples: Number of samples to draw and use for estimating
        the expectation value.
      initial_seed: PRNG seed; see tfp.random.sanitize_seed for details. This
        seed will be used in the `sample` method.  If None, the seed is updated
        after every inference call.  Otherwise, the seed is fixed.
      name: Optional name for the model.
    """
    super().__init__(num_expectation_samples, initial_seed, name)
    self._all_bitstrings = tf.constant(
        list(itertools.product([0, 1], repeat=num_bits)), dtype=tf.int8)
    self._logits_variable = tf.Variable(
        tf.zeros([tf.shape(self._all_bitstrings)[0]]), trainable=False)
    self._distribution = tfd.Categorical(logits=self._logits_variable)

  @property
  def all_bitstrings(self):
    """Returns every bitstring."""
    return self._all_bitstrings

  @property
  def all_energies(self):
    """Returns the energy of every bitstring."""
    return self.energy(self.all_bitstrings)

  @property
  def distribution(self):
    """Categorical distribution set during last call to `self.infer`."""
    return self._distribution

  def _call(self, inputs, *args, **kwargs):
    """See base class docstring."""
    if inputs is None:
      return self.distribution
    else:
      return self.sample(inputs)

  def _entropy(self):
    """See base class docstring."""
    return self.distribution.entropy()

  def _log_partition_forward_pass(self):
    """See base class docstring."""
    # TODO(#115)
    return tf.reduce_logsumexp(self.distribution.logits_parameter())

  def _sample(self, num_samples: int):
    """See base class docstring."""
    return tf.gather(
        self.all_bitstrings,
        self.distribution.sample(num_samples, seed=self.seed),
        axis=0)

  def infer(self, energy: energy_model.BitstringEnergy):
    """See base class docstring."""
    self._energy = energy
    self._logits_variable.assign(-1.0 * self.all_energies)


class BernoulliEnergyInference(EnergyInference):
  """Manages inference for a Bernoulli defined by spin energies."""

  def __init__(self,
               num_bits: int,
               num_expectation_samples: int,
               initial_seed: Union[None, tf.Tensor] = None,
               name: Union[None, str] = None):
    """Initializes a BernoulliEnergyInference.

    Args:
      num_bits: Number of bits on which this layer acts.
      num_expectation_samples: Number of samples to draw and use for estimating
        the expectation value.
      initial_seed: PRNG seed; see tfp.random.sanitize_seed for details. This
        seed will be used in the `sample` method.  If None, the seed is updated
        after every inference call.  Otherwise, the seed is fixed.
      name: Optional name for the model.
    """
    super().__init__(num_expectation_samples, initial_seed, name)
    self._logits_variable = tf.Variable(tf.zeros([num_bits]), trainable=False)
    self._distribution = tfd.Bernoulli(
        logits=self._logits_variable, dtype=tf.int8)

  @property
  def distribution(self):
    """Bernoulli distribution set during last call to `self.infer`."""
    return self._distribution

  def _call(self, inputs, *args, **kwargs):
    """See base class docstring."""
    if inputs is None:
      return self.distribution
    else:
      return self.sample(inputs)

  def _entropy(self):
    """Returns the exact entropy.

    The total entropy of a set of spins is the sum of each individual spin's
    entropies.
    """
    return tf.reduce_sum(self.distribution.entropy())

  def _log_partition_forward_pass(self):
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

  def _sample(self, num_samples: int):
    """See base class docstring"""
    return self.distribution.sample(num_samples, seed=self.seed)

  def infer(self, energy: energy_model.BitstringEnergy):
    """See base class docstring."""
    self._energy = energy
    self._logits_variable.assign(self.energy.logits)
