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
"""Module for defining and sampling from EBMs."""

import abc
import collections
import itertools
from absl import logging

import cirq
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_quantum as tfq

from tensorflow_probability.python.mcmc.internal import util as mcmc_util


class EnergyFunction(tf.keras.Model, abc.ABC):

  def __init__(self, name=None):
    super().__init__(name=name)

  @property
  @abc.abstractmethod
  def num_bits(self):
    raise NotImplementedError()

  @property
  def has_operator(self):
    return False

  @abc.abstractmethod
  def energy(self, bitstrings):
    raise NotImplementedError()

  def operator_shards(self, qubits):
    raise NotImplementedError()

  def operator_expectation(self, expectations):
    raise NotImplementedError()


class KOBE(EnergyFunction):

  def __init__(self,
               num_bits,
               order,
               initializer=tf.keras.initializers.RandomUniform(),
               name=None):
    super().__init__(name=name)
    self._num_bits = num_bits
    self._order = order
    indices_list = []
    for i in range(1, order + 1):
      combos = itertools.combinations(range(num_bits), i)
      indices_list.extend(list(combos))
    self._indices = tf.ragged.stack(indices_list)
    self._num_variables = tf.constant(len(indices_list))
    self._variables = self.add_weight(
        name=f'{self.name}_variables',
        shape=[self._num_variables],
        initializer=initializer)

  @property
  def num_bits(self):
    return self._num_bits

  @property
  def has_operator(self):
    return True

  @property
  def order(self):
    return self._order

  def copy(self):
    kobe = KOBE(self.num_bits, self.order, name=self.name)
    kobe._variables.assign(self._variables)
    return kobe

  @tf.function
  def energy(self, bitstrings):
    spins = 1 - 2 * bitstrings
    parities_t = tf.zeros(
        [self._num_variables, tf.shape(bitstrings)[0]], dtype=tf.float32)
    for i in tf.range(self._num_variables):
      parity = tf.reduce_prod(tf.gather(spins, self._indices[i], axis=-1), -1)
      parities_t = tf.tensor_scatter_nd_update(parities_t, [[i]], [parity])
    return tf.reduce_sum(tf.transpose(parities_t) * self._variables, -1)

  def operator_shards(self, qubits):
    ops = []
    for i in range(self._num_variables):
      string_factors = []
      for loc in self._indices[i]:
        string_factors.append(cirq.Z(qubits[loc]))
      string = cirq.PauliString(string_factors)
      ops.append(cirq.PauliSum.from_pauli_strings(string))
    return ops

  def operator_expectation(self, expectations):
    return tf.reduce_sum(expectations * self._variables)


class MLP(EnergyFunction):

  def __init__(self, num_bits, units, activations, name=None):
    super().__init__(name=name)
    self._num_bits = num_bits
    self._hidden_layers = [
        tf.keras.layers.Dense(u, activation=a)
        for u, a in zip(units, activations)
    ]
    self._energy_layer = tf.keras.layers.Dense(1)
    self.build([1, num_bits])

  @property
  def num_bits(self):
    return self._num_bits

  def copy(self):
    mlp = tf.keras.models.clone_model(self)
    for i in tf.range(len(mlp.trainable_variables)):
      mlp.trainable_variables[i].assign(self.trainable_variables[i])
    return mlp

  @tf.function
  def call(self, bitstrings):
    x = bitstrings
    for hidden_layer in self._hidden_layers:
      x = hidden_layer(x)
    x = self._energy_layer(x)
    return tf.squeeze(x, -1)

  @tf.function
  def energy(self, bitstrings):
    return self(bitstrings)


class EnergySampler(abc.ABC):

  @property
  @abc.abstractmethod
  def energy_function(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def sample(self, num_samples, unique=True):
    raise NotImplementedError()


@tf.function
def unique_bitstrings_with_counts(bitstrings):
  """Extract the unique bitstrings in the given bitstring tensor.
    Works by converting each bitstring to a 64 bit integer, then using built-in
    `tf.unique_with_counts` on this 1-D array, then mapping these integers back
    to
    bitstrings. The inputs and outputs are to be related by the same invariants
    as
    those of `tf.unique_with_counts`,
    y[idx[i]] = input_bitstrings[i] for i in [0, 1,...,rank(input_bitstrings) -
    1]
    TODO(zaqqwerty): the signature and return values are designed to be similar
    to those of tf.unique_with_counts.  This function is needed because
    `tf.unique_with_counts` does not work on 2-D tensors.  When it begins to
    work
    on 2-D tensors, then this function will be deprecated.
    Args:
      input_bitstrings: 2-D `tf.Tensor` of dtype `int8`.  This tensor is
        interpreted as a list of bitstrings.  Bitstrings are required to be 64
        bits or fewer.
      out_idx: An optional `tf.DType` from: `tf.int32`, `tf.int64`. Defaults to
        `tf.int32`.  Specified type of idx and count outputs.
    Returns:
      y: 2-D `tf.Tensor` of dtype `int8` containing the unique 0-axis entries of
        `input_bitstrings`.
      idx: 1-D `tf.Tensor` of dtype `out_idx` such that `idx[i]` is the index in
        `y` containing the value `input_bitstrings[i]`.
      count: 1-D `tf.Tensor` of dtype `out_idx` such that `count[i]` is the
      number
        of occurences of `y[i]` in `input_bitstrings`.
  """
  # Convert bitstrings to integers and uniquify those integers.
  input_shape = tf.shape(bitstrings)
  mask = tf.cast(bitstrings, tf.int64)
  base = tf.bitwise.left_shift(
      mask, tf.range(tf.cast(input_shape[1], tf.int64), dtype=tf.int64))
  ints_equiv = tf.reduce_sum(base, 1)
  _, idx, counts = tf.unique_with_counts(ints_equiv)

  # Convert unique integers to corresponding unique bitstrings.
  unique_bitstrings = tf.zeros((tf.shape(counts)[0], input_shape[1]),
                               dtype=tf.int8)
  unique_bitstrings = tf.tensor_scatter_nd_update(unique_bitstrings,
                                                  tf.expand_dims(idx, axis=1),
                                                  bitstrings)

  return unique_bitstrings, counts


class EnergyKernel(tfp.mcmc.TransitionKernel, abc.ABC):

  @property
  @abc.abstractmethod
  def energy_function(self):
    raise NotImplementedError()


class UncalibratedGWGResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple('UncalibratedGWGResults',
                           ['target_log_prob', 'log_acceptance_correction'])):
  __slots__ = ()


class UncalibratedGWG(EnergyKernel):

  def __init__(self,
               energy_function,
               gradient=True,
               temperature=2.0,
               num_samples=1,
               name=None):
    self._parameters = dict(
        energy_function=energy_function,
        gradient=gradient,
        temperature=temperature,
        num_samples=num_samples,
        name=name)
    self._diff_function = self._grad_diff_function if gradient else self._exact_diff_function

  @property
  def energy_function(self):
    return self._parameters['energy_function']

  @property
  def gradient(self):
    return self._parameters['gradient']

  @property
  def temperature(self):
    return self._parameters['temperature']

  @property
  def num_samples(self):
    return self._parameters['num_samples']

  @property
  def name(self):
    return self._parameters['name']

  @property
  def parameters(self):
    return self._parameters

  @property
  def is_calibrated(self):
    return False

  def copy(self):
    return UncalibratedGWG(
        self.energy_function.copy(),
        gradient=self.gradient,
        temperature=self.temperature,
        num_samples=self.num_samples,
        name=self.name)

  @tf.function
  def _exact_diff_function(self, current_state):
    current_state_t = tf.transpose(current_state)
    diff_t = tf.zeros_like(current_state_t, dtype=tf.float32)
    current_energy = self.energy_function.energy(current_state)

    for i in tf.range(tf.shape(current_state)[-1]):
      pert_state = tf.transpose(
          tf.tensor_scatter_nd_update(current_state_t, [[i]],
                                      tf.expand_dims(1 - current_state_t[i],
                                                     0)))
      diff_t = tf.tensor_scatter_nd_update(
          diff_t, [[i]],
          tf.expand_dims(
              current_energy - self.energy_function.energy(pert_state), 0))
    return tf.transpose(diff_t)

  @tf.function
  def _grad_diff_function(self, current_state):
    current_state = tf.cast(current_state, tf.float32)
    with tf.GradientTape() as tape:
      tape.watch(current_state)
      energy = self.energy_function.energy(current_state)
    grad = tape.gradient(energy, current_state)
    return (2 * current_state - 1) * grad

  @tf.function
  def one_step(self, current_state, previous_kernel_results):
    forward_diff = self._diff_function(current_state)
    forward_dist = tfp.distributions.OneHotCategorical(
        logits=forward_diff / self.temperature, dtype=tf.int8)
    all_changes = forward_dist.sample(self.num_samples)
    total_change = tf.cast(tf.reduce_sum(all_changes, 0) > 0, tf.int8)
    next_state = tf.bitwise.bitwise_xor(total_change, current_state)

    target_log_prob = -1 * self.energy_function.energy(next_state)

    forward_log_prob = tf.reduce_sum(forward_dist.log_prob(all_changes), 0)
    backward_diff = self._diff_function(next_state)
    backward_dist = tfp.distributions.OneHotCategorical(
        logits=backward_diff / self.temperature, dtype=tf.int8)
    backward_log_prob = tf.reduce_sum(backward_dist.log_prob(all_changes), 0)
    log_acceptance_correction = backward_log_prob - forward_log_prob

    kernel_results = UncalibratedGWGResults(
        target_log_prob=target_log_prob,
        log_acceptance_correction=log_acceptance_correction)

    return next_state, kernel_results

  @tf.function
  def bootstrap_results(self, init_state):
    target_log_prob = -1 * self.energy_function.energy(init_state)
    kernel_results = UncalibratedGWGResults(
        target_log_prob=target_log_prob,
        log_acceptance_correction=tf.zeros_like(target_log_prob))
    return kernel_results


class MetropolisHastings(tfp.mcmc.MetropolisHastings, EnergyKernel):

  def __init__(self, inner_kernel, name=None):
    super().__init__(inner_kernel, name=name)

  @property
  def energy_function(self):
    return self.inner_kernel.energy_function

  def copy(self):
    return MetropolisHastings(self.inner_kernel.copy())


class GWG(MetropolisHastings):

  def __init__(self,
               energy_function,
               gradient=True,
               temperature=2.0,
               num_samples=1,
               name=None):
    self._impl = MetropolisHastings(
        UncalibratedGWG(
            energy_function,
            gradient=gradient,
            temperature=temperature,
            num_samples=num_samples,
            name=name))
    self._parameters = self._impl.inner_kernel.parameters.copy()

  @property
  def energy_function(self):
    return self._impl.inner_kernel.energy_function

  @property
  def gradient(self):
    return self._impl.inner_kernel.gradient

  @property
  def temperature(self):
    return self._impl.inner_kernel.temperature

  @property
  def num_samples(self):
    return self._impl.inner_kernel.num_samples

  @property
  def name(self):
    return self._impl.inner_kernel.name

  @property
  def parameters(self):
    return self._impl.inner_kernel.parameters

  @property
  def is_calibrated(self):
    return True

  def copy(self):
    return GWG(
        self.energy_function.copy(),
        gradient=self.gradient,
        temperature=self.temperature,
        num_samples=self.num_samples,
        name=self.name)

  @tf.function
  def one_step(self, current_state, previous_kernel_results):
    return self._impl.one_step(current_state, previous_kernel_results)

  @tf.function
  def bootstrap_results(self, init_state):
    return self._impl.bootstrap_results(init_state)


class MCMC(EnergySampler):

  def __init__(self,
               kernel,
               num_chains=1,
               buffer_capacity=1000,
               buffer_probability=1,
               num_burnin_steps=0,
               num_steps_between_results=0,
               parallel_iterations=10,
               name=None):
    self._kernel = kernel
    self._num_chains = num_chains
    self._num_bits = kernel.energy_function.num_bits
    self._buffer_capacity = buffer_capacity
    self._buffer_probability = buffer_probability
    self._num_burnin_steps = num_burnin_steps
    self._num_steps_between_results = num_steps_between_results
    self._parallel_iterations = parallel_iterations
    self._name = name

    self._buffer = tf.queue.RandomShuffleQueue(
        buffer_capacity, 0, [tf.int8], shapes=[self._num_bits])

  @property
  def energy_function(self):
    return self.kernel.energy_function

  @property
  def kernel(self):
    return self._kernel

  @property
  def num_chains(self):
    return self._num_chains

  @property
  def buffer_capacity(self):
    return self._buffer_capacity

  @property
  def buffer_probability(self):
    return self._buffer_probability

  @property
  def num_burnin_steps(self):
    return self._num_burnin_steps

  @property
  def num_steps_between_results(self):
    return self._num_steps_between_results

  @property
  def parallel_iterations(self):
    return self._parallel_iterations

  @property
  def name(self):
    return self._name

  def copy(self):
    mcmc = MCMC(
        self.kernel.copy(),
        num_chains=self.num_chains,
        buffer_capacity=self.buffer_capacity,
        buffer_probability=self.buffer_probability,
        num_burnin_steps=self.num_burnin_steps,
        num_steps_between_results=self.num_steps_between_results,
        parallel_iterations=self.parallel_iterations,
        name=self.name)
    mcmc._buffer = tf.queue.QueueBase.from_list(tf.constant(0), [self._buffer])
    return mcmc

  @tf.function
  def sample(self, num_samples, unique=True):
    num_results = tf.cast(tf.math.ceil(num_samples / self.num_chains), tf.int32)

    if tf.random.uniform(()) > self.buffer_probability:
      init_state = tf.cast(
          tf.random.uniform([self.num_chains, self._num_bits],
                            maxval=2,
                            dtype=tf.int32), tf.int8)
    else:
      init_state = self._buffer.dequeue_many(
          tf.math.minimum(self.num_chains, self._buffer.size()))
      init_state = tf.concat([
          init_state,
          tf.cast(
              tf.random.uniform(
                  [self.num_chains - tf.shape(init_state)[0], self._num_bits],
                  maxval=2,
                  dtype=tf.int32), tf.int8)
      ], 0)

    previous_kernel_results = self.kernel.bootstrap_results(init_state)

    samples = tfp.mcmc.sample_chain(
        num_results,
        init_state,
        previous_kernel_results=previous_kernel_results,
        kernel=self.kernel,
        num_burnin_steps=self.num_burnin_steps,
        num_steps_between_results=self.num_steps_between_results,
        trace_fn=None,
        parallel_iterations=self.parallel_iterations,
        name=self.name)

    sampled_states = tf.reshape(samples, [-1, self._num_bits])
    if tf.shape(sampled_states)[0] > self.buffer_capacity - self._buffer.size():
      self._buffer.dequeue_many(
          tf.math.minimum(
              tf.shape(sampled_states)[0] -
              (self.buffer_capacity - self._buffer.size()),
              self._buffer.size()))
    self._buffer.enqueue_many(sampled_states[:tf.math.minimum(
        tf.shape(sampled_states)[0], self.buffer_capacity)])

    sampled_states = sampled_states[:num_samples]
    if unique:
      return unique_bitstrings_with_counts(sampled_states)
    return sampled_states


class EBM(tf.keras.Model):

  def __init__(self,
               energy_function,
               energy_sampler,
               is_analytic=False,
               name=None):
    super().__init__(name=name)
    self._energy_function = energy_function
    self._energy_sampler = energy_sampler
    self._is_analytic = is_analytic
    if is_analytic:
      self._all_bitstrings = tf.constant(
          list(itertools.product([0, 1], repeat=energy_function.num_bits)),
          dtype=tf.int8)

  @property
  def num_bits(self):
    return self._energy_function.num_bits

  @property
  def has_operator(self):
    return self._energy_function.has_operator

  @property
  def is_analytic(self):
    return self._is_analytic

  def copy(self):
    energy_sampler = self._energy_sampler.copy()
    return EBM(
        energy_sampler.energy_function,
        energy_sampler,
        is_analytic=self.is_analytic,
        name=self.name)

  @tf.function
  def energy(self, bitstrings):
    return self._energy_function.energy(bitstrings)

  def operator_expectation(self, expectations):
    return self._energy_function.operator_expectation(expectations)

  def operator_shards(self, qubits):
    return self._energy_function.operator_shards(qubits)

  @tf.function
  def sample(self, num_samples, unique=True):
    if self.is_analytic and self._energy_sampler is None:
      samples = tf.gather(
          self._all_bitstrings,
          tfp.distributions.Categorical(logits=-1 *
                                        self.energies()).sample(num_samples))
      if unique:
        return unique_bitstrings_with_counts(samples)
      return samples
    else:
      return self._energy_sampler.sample(num_samples, unique=unique)

  @tf.function
  def energies(self):
    if self.is_analytic:
      return self.energy(self._all_bitstrings)
    raise NotImplementedError()

  @tf.function
  def probabilities(self):
    if self.is_analytic:
      return tf.exp(-self.ebm.energies()) / tf.exp(
          self.log_partition_function())
    raise NotImplementedError()

  @tf.function
  def log_partition_function(self):
    if self.is_analytic:
      return tf.reduce_logsumexp(-1 * self.energies())
    raise NotImplementedError()

  @tf.function
  def entropy(self):
    if self.is_analytic:
      return tfp.distributions.Categorical(logits=-1 *
                                           self.energies()).entropy()
    raise NotImplementedError()


class Bernoulli(EBM):

  def __init__(self,
               num_bits,
               initializer=tf.keras.initializers.RandomUniform(),
               is_analytic=False,
               name=None):
    tf.keras.Model.__init__(self, name=name)
    self._num_bits = num_bits
    self._variables = self.add_weight(
        name=f'{self.name}_variables',
        shape=[self.num_bits],
        initializer=initializer)
    self._is_analytic = is_analytic
    if is_analytic:
      self._all_bitstrings = tf.constant(
          list(itertools.product([0, 1], repeat=num_bits)), dtype=tf.int8)

  @property
  def num_bits(self):
    return self._num_bits

  @property
  def has_operator(self):
    return True

  @property
  def is_analytic(self):
    return self._is_analytic

  def copy(self):
    bernoulli = Bernoulli(self.num_bits, name=self.name)
    bernoulli._variables.assign(self._variables)
    return bernoulli

  @tf.function
  def energy(self, bitstrings):
    return tf.reduce_sum(
        tf.cast(1 - 2 * bitstrings, tf.float32) * self._variables, -1)

  def operator_shards(self, qubits):
    return [
        cirq.PauliSum.from_pauli_strings(cirq.Z(qubits[i]))
        for i in range(self.num_bits)
    ]

  def operator_expectation(self, expectations):
    return tf.reduce_sum(expectations * self._variables)

  @tf.function
  def sample(self, num_samples, unique=True):
    r"""Fairly samples from the EBM defined by `energy`.

        For Bernoulli distribution, let $p$ be the probability of bit being `1`.
        In this case, we have $p = \frac{e^{theta}}{{e^{theta}+e^{-theta}}}$.
        Therefore, each independent logit is:
          $$logit = \log\frac{p}{1-p} = \log\frac{e^{theta}}{e^{-theta}}
                 = \log{e^{2*theta}} = 2*theta$$
        """
    samples = tfp.distributions.Bernoulli(
        logits=2 * self._variables, dtype=tf.int8).sample(num_samples)
    if unique:
      return unique_bitstrings_with_counts(samples)
    return samples

  @tf.function
  def energies(self):
    if self.is_analytic:
      return self.energy(self._all_bitstrings)
    raise NotImplementedError()

  @tf.function
  def probabilities(self):
    if self.is_analytic:
      return tf.exp(-self.energies()) / tf.exp(self.log_partition_function())
    raise NotImplementedError()

  @tf.function
  def log_partition_function(self):
    if self.is_analytic:
      return tf.reduce_logsumexp(-1 * self.energies())
    raise NotImplementedError()

  @tf.function
  def entropy(self):
    return tf.reduce_sum(
        tfp.distributions.Bernoulli(logits=2 * self._variables).entropy())

  
def probability_to_logit(probability):
  p = tf.cast(probability, tf.dtypes.float32)
  return tf.math.log(p) - tf.math.log(1 - p)


def logit_to_probability(logit_in):
  logit = tf.cast(logit_in, tf.dtypes.float32)
  return tf.math.divide(tf.math.exp(logit), 1 + tf.math.exp(logit))
