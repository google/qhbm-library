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

from qhbmlib.models import energy
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
               input_energy: energy.BitstringEnergy,
               initial_seed: Union[None, tf.Tensor] = None,
               name: Union[None, str] = None):
    """Initializes an EnergyInferenceBase.

    Args:
      input_energy: The parameterized energy function which defines this
        distribution via the equations of an energy based model.  This class
        assumes that all parameters of `energy` are `tf.Variable`s and that
        they are all returned by `energy.variables`.
      initial_seed: PRNG seed; see tfp.random.sanitize_seed for details. This
        seed will be used in the `sample` method.  If None, the seed is updated
        after every inference call.  Otherwise, the seed is fixed.
      name: Optional name for the model.
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

  @property
  def variables_updated(self):
    """Returns True if tracked variables do not have the checkpointed values."""
    if self._checkpoint:
      variables_not_equal_list = tf.nest.map_structure(
          lambda v, vc: tf.math.reduce_any(tf.math.not_equal(v, vc)),
          self._tracked_variables, self._tracked_variables_checkpoint)
      return tf.math.reduce_any(tf.stack(variables_not_equal_list))
    else:
      return False

  def _checkpoint_variables(self):
    """Checkpoints the currently tracked variables."""
    if self._checkpoint:
      tf.nest.map_structure(lambda v, vc: vc.assign(v), self._tracked_variables,
                            self._tracked_variables_checkpoint)

  def _preface_inference(self):
    """Things all energy inference methods do before proceeding.

    Called by `preface_inference` before the wrapped inference method.
    Currently includes:
      - run `self._ready_inference` if this is first call of a wrapped function
      - change the seed if not set by the user during initialization
      - run `self._ready_inference` if tracked energy parameters changed

    Note: subclasses should take care to call the superclass method.
    """
    if self._first_inference:
      self._checkpoint_variables()
      self._ready_inference()
      self._first_inference.assign(False)
    if self._update_seed:
      new_seed, _ = tfp.random.split_seed(self.seed)
      self._seed.assign(new_seed)
    if self.variables_updated:
      self._checkpoint_variables()
      self._ready_inference()

  @abc.abstractmethod
  def _ready_inference(self):
    """Performs computations common to all inference methods.

    Contains inference code that must be run first if the variables of
    `self.energy` have been updated since the last time inference was performed.
    """

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


class EnergyInference(EnergyInferenceBase):
  """Provides some default method implementations."""

  def __init__(self,
               input_energy: energy.BitstringEnergy,
               num_expectation_samples: int,
               initial_seed: Union[None, tf.Tensor] = None,
               name: Union[None, str] = None):
    """Initializes an EnergyInference.

    Args:
      input_energy: The parameterized energy function which defines this
        distribution via the equations of an energy based model.  This class
        assumes that all parameters of `energy` are `tf.Variable`s and that
        they are all returned by `energy.variables`.
      num_expectation_samples: Number of samples to draw and use for estimating
        the expectation value.
      initial_seed: PRNG seed; see tfp.random.sanitize_seed for details. This
        seed will be used in the `sample` method.  If None, the seed is updated
        after every inference call.  Otherwise, the seed is fixed.
      name: Optional name for the model.
    """
    super().__init__(input_energy, initial_seed, name)
    self.num_expectation_samples = num_expectation_samples

  def _entropy(self):
    """Default implementation wrapped by `self.entropy`."""
    return self.expectation(self.energy) + self.log_partition()

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

  def _log_partition_forward_pass(self):
    """Returns approximation to the log partition function.

    Note this simple estimator is biased.  See the paper appendix for its
    motivation.  TODO(#216)
    """
    samples = self.sample(self.num_expectation_samples)
    unique_samples, _, _ = utils.unique_bitstrings_with_counts(samples)
    return tf.math.reduce_logsumexp(-1.0 * self.energy(unique_samples))

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
               input_energy: energy.BitstringEnergy,
               num_expectation_samples: int,
               initial_seed: Union[None, tf.Tensor] = None,
               name: Union[None, str] = None):
    """Initializes an AnalyticEnergyInference.

    Internally, this class saves all possible bitstrings as a tensor, whose
    energies are calculated relative to an input energy function for sampling
    and other inference tasks.

    Args:
      input_energy: The parameterized energy function which defines this
        distribution via the equations of an energy based model.  This class
        assumes that all parameters of `energy` are `tf.Variable`s and that
        they are all returned by `energy.variables`.
      num_expectation_samples: Number of samples to draw and use for estimating
        the expectation value.
      initial_seed: PRNG seed; see tfp.random.sanitize_seed for details. This
        seed will be used in the `sample` method.  If None, the seed is updated
        after every inference call.  Otherwise, the seed is fixed.
      name: Optional name for the model.
    """
    super().__init__(input_energy, num_expectation_samples, initial_seed, name)
    self._all_bitstrings = tf.constant(
        list(itertools.product([0, 1], repeat=input_energy.num_bits)),
        dtype=tf.int8)
    self._logits_variable = tf.Variable(
        -input_energy(self.all_bitstrings), trainable=False)
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
    self._logits_variable.assign(-self.all_energies)

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


class BernoulliEnergyInference(EnergyInference):
  """Manages inference for a Bernoulli defined by spin energies."""

  def __init__(self,
               input_energy: energy.BernoulliEnergy,
               num_expectation_samples: int,
               initial_seed: Union[None, tf.Tensor] = None,
               name: Union[None, str] = None):
    """Initializes a BernoulliEnergyInference.

    Args:
      input_energy: The parameterized energy function which defines this
        distribution via the equations of an energy based model.  This class
        assumes that all parameters of `energy` are `tf.Variable`s and that
        they are all returned by `energy.variables`.
      num_expectation_samples: Number of samples to draw and use for estimating
        the expectation value.
      initial_seed: PRNG seed; see tfp.random.sanitize_seed for details. This
        seed will be used in the `sample` method.  If None, the seed is updated
        after every inference call.  Otherwise, the seed is fixed.
      name: Optional name for the model.
    """
    super().__init__(input_energy, num_expectation_samples, initial_seed, name)
    self._logits_variable = tf.Variable(input_energy.logits, trainable=False)
    self._distribution = tfd.Bernoulli(
        logits=self._logits_variable, dtype=tf.int8)

  @property
  def distribution(self):
    """Bernoulli distribution set during `self._ready_inference`."""
    return self._distribution

  def _ready_inference(self):
    """See base class docstring."""
    self._logits_variable.assign(self.energy.logits)

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
        tf.math.exp(thetas) + tf.math.exp(-thetas))
    return tf.math.reduce_sum(single_log_partitions)

  def _sample(self, num_samples: int):
    """See base class docstring"""
    return self.distribution.sample(num_samples, seed=self.seed)


class GibbsWithGradientsKernel(tfp.mcmc.TransitionKernel):
  """Implements the Gibbs With Gradients update rule.

  See Algorithm 1 of
  https://arxiv.org/abs/2102.04509v2
  for details.
  """

  def __init__(self, input_energy):
    """Initializes a GibbsWithGradientsKernel.

    Args:
      input_energy: The parameterized energy function which helps define the
        acceptance probabilities of the Markov chain.
    """
    self._energy = input_energy
    self._num_bits = input_energy.num_bits
    self._parameters = dict(input_energy=input_energy)

  def _q_i_of_x(self, x):
    """Sets the distribution written in equation 6 of the paper."""
    x_float = tf.cast(x, tf.float32)
    with tf.GradientTape() as tape:
      tape.watch(x_float)
      current_energy = tf.squeeze(self._energy(tf.expand_dims(x_float, 0)))
    f_grad = tape.gradient(current_energy, x_float)
    d_tilde = -(2 * tf.cast(x, tf.float32) - 1) * f_grad
    i_probs = tf.nn.softmax(d_tilde / 2.0)
    return tfp.distributions.Categorical(probs=i_probs)

  def one_step(self, current_state, previous_kernel_results, seed=None):
    """Takes one step of the TransitionKernel.

    Args:
      current_state: 1D `tf.Tensor` which is the current chain state.
      previous_kernel_results: Required but unused argument.
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

    Returns:
      next_state: 1D `Tensor` which is the next state in the chain.
      kernel_results: Empty list.
    """
    del previous_kernel_results
    q_i_of_x = self._q_i_of_x(current_state)
    proposed_i = q_i_of_x.sample()
    if current_state[proposed_i] == 0:
      x_prime = current_state + tf.one_hot(
          proposed_i, self._num_bits, dtype=tf.int8)
    else:
      x_prime = current_state - tf.one_hot(
          proposed_i, self._num_bits, dtype=tf.int8)
    q_i_of_x_prime = self._q_i_of_x(x_prime)
    q_ratio = q_i_of_x_prime.probs_parameter(
    )[proposed_i] / q_i_of_x.probs_parameter()[proposed_i]
    exp_f = tf.math.exp(-self._energy(tf.expand_dims(x_prime, 0)) +
                        self._energy(tf.expand_dims(current_state, 0)))
    accept_prob = tf.math.minimum(exp_f * q_ratio, 1.0)
    accept = tfp.distributions.Bernoulli(
        probs=accept_prob, dtype=tf.bool).sample()
    if accept:
      next_state = x_prime
    else:
      next_state = current_state
    kernel_results = []
    return next_state, kernel_results

  @property
  def is_calibrated(self):
    """Returns `True` if Markov chain converges to specified distribution."""
    return True

  def bootstrap_results(self, init_state):
    """Returns an object with the same type as returned by `one_step(...)[1]`.

    Args:
      init_state: 1D `tf.Tensor` which is the initial chain state.

    Returns:
      kernel_results: Empty list.
    """
    del init_state
    return []


class GibbsWithGradientsInference(EnergyInference):
  """Manages inference using a Gibbs With Gradients kernel."""

  def __init__(self,
               input_energy: energy.BitstringEnergy,
               num_expectation_samples: int,
               num_burnin_samples: int,
               initial_seed: Union[None, tf.Tensor] = None,
               name: Union[None, str] = None):
    """Initializes a GibbsWithGradientsInference.

    Args:
      input_energy: The parameterized energy function which defines this
        distribution via the equations of an energy based model.  This class
        assumes that all parameters of `energy` are `tf.Variable`s and that
        they are all returned by `energy.variables`.
      num_expectation_samples: Number of samples to draw and use for estimating
        the expectation value.
      num_burnin_samples: Number of samples to discard when letting the chain
        equilibrate after updating the parameters of `input_energy`.
      initial_seed: PRNG seed; see tfp.random.sanitize_seed for details. This
        seed will be used in the `sample` method.  If None, the seed is updated
        after every inference call.  Otherwise, the seed is fixed.
      name: Optional name for the model.
    """
    super().__init__(input_energy, num_expectation_samples, initial_seed, name)
    self._num_burnin_samples = num_burnin_samples
    self._kernel = GibbsWithGradientsKernel(input_energy)
    self._current_state = tf.Variable(
        tfp.distributions.Bernoulli(
            probs=[0.5] * self.energy.num_bits, dtype=tf.int8).sample(),
        trainable=False)
    self._traced_one_step = tf.function(
        self._kernel.one_step).get_concrete_function(self._current_state, [])

  @property
  def num_burnin_samples(self):
    """Returns the number of burnin samples discarded by this class.

    Number of samples to discard when letting the chain equilibrate after
    updating the parameters of `self.energy`.
    """
    return self._num_burnin_samples

  def _wrapped_one_step(self):
    """Returns the next state."""
    next_state, _ = self._traced_one_step(self._current_state, [])
    self._current_state.assign(next_state)
    return next_state

  def _ready_inference(self):
    """See base class docstring.

    Runs the chain for a number of steps without saving the results, in order
    to better reach equilibrium before recording samples.
    """
    for _ in tf.range(self.num_burnin_samples):
      _ = self._wrapped_one_step()

  def _call(self, inputs, *args, **kwargs):
    """See base class docstring."""
    return self.sample(inputs)

  def _sample(self, num_samples: int):
    """See base class docstring.

    The kernel is repeatedly called to traverse a chain of samples.
    """
    ta = tf.TensorArray(tf.int8, size=num_samples)
    for i in tf.range(num_samples):
      ta = ta.write(i, self._wrapped_one_step())
    return ta.stack()
