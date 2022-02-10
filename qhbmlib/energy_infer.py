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
from qhbmlib import utils


class EnergyInference(tf.keras.layers.Layer, abc.ABC):
  r"""Sets the methods required for inference on BitstringEnergy objects.

  Let $E$ be the energy function defined by a given `BitstringEnergy`, and let
  $X$ be the set of bitstrings in the domain of $E$.  Associated with $E$ is
  a probability distribution
  $$p(x) = \frac{e^{-E(x)}}{\sum_{y\in X} e^{-E(y)}},$$
  which we call the Energy Based Model (EBM) associated with $E$.  Inference
  in this class means estimating quantities of interest relative to the EBM.
  """

  def __init__(self, input_energy: energy_model.BitstringEnergy, name: Union[None, str]=None, initial_seed: Union[None, tf.Tensor]=None):
    """Initializes an EnergyInference.

    Args:
      input_energy: The parameterized energy function which defines this distribution
        via the equations of an energy based model.  This function assumes that
        all parameters of `energy` are `tf.Variable`s and that they are all
        returned by `energy.variables`.
      name: Optional name for the model.
      initial_seed: PRNG seed; see tfp.random.sanitize_seed for details. This
        seed will be used in the `sample` method.  If None, the seed is updated
        after every inference call.  Otherwise, the seed is fixed.
    """
    super().__init__(name=name)
    self._energy = input_energy
    self._energy.build([None, self._energy.num_bits])

    self._tracked_variables = input_energy.variables
    if len(self._tracked_variables) == 0:
      self._checkpoint = False
    else:
      self._tracked_variables_checkpoint = [
          tf.Variable(v.read_value(), trainable=False)
          for v in self._tracked_variables
      ]
      self._checkpoint = True

    if initial_seed is None:
      self._update_seed = tf.Variable(True, trainable=False)
    else:
      self._update_seed = tf.Variable(False, trainable=False)
    self._seed = tf.Variable(tfp.random.sanitize_seed(initial_seed), trainable=False)

    self._checkpoint_variables()
    self._do_first_inference = tf.Variable(False, trainable=False)

  @property
  def seed(self):
    """Current TFP compatible seed controlling sampling behavior."""
    return self._seed
    
  @seed.setter
  def seed(self, initial_seed: Union[None, tf.Tensor]):
    """Sets a new value of the random seed.

    Args:
      initial_seed: PRNG seed; see tfp.random.sanitize_seed for details. This
        seed will be used in the `sample` method.  If None, the seed is updated
        after every inference call.  Otherwise, the seed is fixed.
    """
    if initial_seed is None:
      self._update_seed.assign(True)
    else:
      self._update_seed.assign(False)
    self._seed.assign(tfp.random.sanitize_seed(initial_seed))

  @abc.abstractmethod
  def _ready_inference(self):
    """Performs computations common to all inference methods.

    Contains inference code that must be run first if the variables of
    `self.energy` have been updated since the last time inference was performed.
    """
    raise NotImplementedError()

  def _preface_every_call(f):
    """Wraps given function with things to run before every inference call.

    This decorator wraps the given function to so it performs the following
    check: if the values of the variables in `self.energy` have changed since
    the last checkpoint, call `self._ready_inference` before proceeding.

    As well, this decorator wraps the given function so it changes the seed
    if not set by the user during initialization.

    Args:
      f: The method of `EnergyInference` to wrap.

    Returns:
      wrapper: The wrapped function.
    """
    def wrapper(self, *args, **kwargs):
      if self._do_first_inference:
        self._ready_inference()
        self._do_first_inference.assign(False)
      if self._update_seed:
        new_seed, _ = tfp.random.split_seed(self.seed)
        self._seed.assign(new_seed)
      if self.variables_updated:
        self._checkpoint_variables()
        self._ready_inference()
      return f(self, *args, **kwargs)
    return wrapper

  @_preface_every_call
  def sample(self, n):
    """Returns samples from the EBM corresponding to `self.energy`.

    This can be an approximate sampling.
    """
    return self._sample(n)
  
  @_preface_every_call
  def entropy(self):
    """Returns an estimate of the entropy."""
    return self._entropy()

  @_preface_every_call
  def log_partition(self):
    """Returns an estimate of the log partition function."""
    return self._log_partition()

  @_preface_every_call
  def expectation(self, function, num_samples: int):
    """Returns the expectation value of the given function.

    Args:
      function: Mapping from a 2D tensor of bitstrings to a possibly nested
        structure.  The structure must have atomic elements all of which are
        float tensors with the same batch size as the input bitstrings.
      num_samples: The number of bitstring samples to use when estimating the
        expectation value of `function`.
    """
    return self._expectation(function, num_samples)

  @property
  def variables_updated(self):
    """Returns True if tracked variables do not have the checkpointed values."""
    if self._checkpoint:
      variables_not_equal_list = tf.nest.map_structure(
          lambda v, vc: tf.math.reduce_any(tf.math.not_equal(v, vc)),
          self._tracked_variables,
          self._tracked_variables_checkpoint)
      return tf.math.reduce_any(tf.stack(variables_not_equal_list))
    else:
      return False

  def _checkpoint_variables(self):
    """Checkpoints the currently tracked variables."""
    if self._checkpoint:
      tf.nest.map_structure(
          lambda v, vc: vc.assign(v),
          self._tracked_variables,
          self._tracked_variables_checkpoint)

  @property
  def energy(self):
    """The energy function which sets the probabilities for this EBM."""
    return self._energy
  
  @abc.abstractmethod
  def _sample(self, n):
    """Default implementation wrapped by `self.sample`."""
    raise NotImplementedError()

  @abc.abstractmethod
  def _entropy(self):
    """Default implementation wrapped by `self.entropy`."""
    raise NotImplementedError()

  @abc.abstractmethod
  def _log_partition(self):
    """Default implementation wrapped by `self.log_partition`."""
    raise NotImplementedError()

  def _expectation(self, function, num_samples: int):
    """Default implementation wrapped by `self.expectation`.

    Estimates an expectation value using sample averaging.
    """
    @tf.custom_gradient
    def _inner_expectation():
      """Enables derivatives."""
      samples = tf.stop_gradient(self.sample(num_samples))
      bitstrings, counts = utils.unique_bitstrings_with_counts(samples)

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

  @_preface_every_call
  def call(self, inputs):
    """Returns the number of samples specified in the inputs."""
    return self.sample(inputs)


class AnalyticEnergyInference(EnergyInference):
  """Uses an explicit categorical distribution to implement parent functions."""

  def __init__(self, input_energy: energy_model.BitstringEnergy, name: Union[None, str]=None, initial_seed: Union[None, tf.Tensor]=None):
    """Initializes an AnalyticEnergyInference.

    Internally, this class saves all possible bitstrings as a tensor, whose
    energies are calculated relative to an input energy function for sampling
    and other inference tasks.

    Args:
      input_energy: The parameterized energy function which defines this distribution
        via the equations of an energy based model.  This function assumes that
        all parameters of `energy` are `tf.Variable`s and that they are all
        returned by `energy.variables`.
      name: Optional name for the model.
      initial_seed: PRNG seed; see tfp.random.sanitize_seed for details. This
        seed will be used in the `sample` method.  If None, the seed is updated
        after every inference call.  Otherwise, the seed is fixed.
    """
    super().__init__(input_energy, name, initial_seed)
    self._all_bitstrings = tf.constant(
        list(itertools.product([0, 1], repeat=input_energy.num_bits)), dtype=tf.int8)
    self._logits_variable = tf.Variable(-1.0 * input_energy(self.all_bitstrings))
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
    """Categorical distribution set during `self._ready_inference`."""
    return self._distribution

  def _ready_inference(self):
    """See base class docstring."""
    self._logits_variable.assign(-1.0 * self.all_energies)
    
  def _sample(self, n):
    """See base class docstring"""
    return tf.gather(
        self.all_bitstrings,
        self.distribution.sample(n, seed=self.seed),
        axis=0)

  def _entropy(self):
    """See base class docstring"""
    return self.distribution.entropy()

  def _log_partition(self):
    """See base class docstring"""
    return tf.reduce_logsumexp(self.distribution.logits_parameter())

  def call(self, inputs):
    if inputs is None:
      return self.distribution
    else:
      return self.sample(inputs)


class BernoulliEnergyInference(EnergyInference):
  """Manages inference for a Bernoulli defined by spin energies."""

  def __init__(self, input_energy: energy_model.BernoulliEnergy, name: Union[None, str]=None, initial_seed: Union[None, tf.Tensor]=None):
    """Initializes a BernoulliEnergyInference.

    Args:
      input_energy: The parameterized energy function which defines this distribution
        via the equations of an energy based model.  This function assumes that
        all parameters of `energy` are `tf.Variable`s and that they are all
        returned by `energy.variables`.
      name: Optional name for the model.
      initial_seed: PRNG seed; see tfp.random.sanitize_seed for details. This
        seed will be used in the `sample` method.  If None, the seed is updated
        after every inference call.  Otherwise, the seed is fixed.
    """
    super().__init__(input_energy, name, initial_seed)
    self._logits_variable = tf.Variable(input_energy.logits, trainable=False)
    self._distribution = tfd.Bernoulli(logits=self._logits_variable, dtype=tf.int8)

  @property
  def distribution(self):
    """Bernoulli distribution set during `self._ready_inference`."""
    return self._distribution

  def _ready_inference(self):
    """See base class docstring."""
    self._logits_variable.assign(self.energy.logits)

  def _sample(self, n):
    """See base class docstring"""
    return self.distribution.sample(n, seed=self.seed)

  def _entropy(self):
    """Returns the exact entropy.

    The total entropy of a set of spins is the sum of each individual spin's
    entropies.
    """
    return tf.reduce_sum(self.distribution.entropy())

  def _log_partition(self):
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
    if inputs is None:
      return self.distribution
    else:
      return self.sample(inputs)
