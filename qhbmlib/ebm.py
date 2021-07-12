# Copyright 2021 The QHBM Library Authors.
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

import collections
import itertools
import abc

import cirq
import tensorflow as tf
import tensorflow_probability as tfp


class EBM(abc.ABC):

  def __init__(self, num_bits, analytic=False):
    self._num_bits = num_bits
    self.analytic = analytic
    if analytic:
      self.all_bitstrings = tf.constant(
          list(itertools.product([0, 1], repeat=num_bits)), dtype=tf.int8)
    self.sampler = None

  @property
  def num_bits(self):
    return self._num_bits

  @property
  @abc.abstractmethod
  def thetas(self):
    raise NotImplementedError()

  @property
  def sampler(self):
    return self._sampler

  @sampler.setter
  def sampler(self, sampler):
    self._sampler = sampler

  @tf.function
  def sample(self, num_samples):
    if self.sampler is not None:
      return self.sampler.sample(num_samples)
    if self.analytic:
      all_energies = self.energy(self.all_bitstrings)
      dist = tfp.distributions.Categorical(
          logits=-1 * all_energies, dtype=tf.int8)
      return tf.gather(self.all_bitstrings, dist.sample(num_samples))
    raise NotImplementedError()

  @abc.abstractmethod
  def energy(self, bitstrings):
    raise NotImplementedError()

  @tf.function
  def log_partition_function(self):
    if self.analytic:
      all_energies = self.energy(self.all_bitstrings)
      return tf.reduce_logsumexp(-1 * all_energies)
    raise NotImplementedError()

  @tf.function
  def entropy(self):
    if self.analytic:
      all_energies = self.energy(self.all_bitstrings)
      dist = tfp.distributions.Categorical(logits=-1 * all_energies)
      return dist.entropy()
    raise NotImplementedError()

  def operators(self, qubits):
    raise NotImplementedError()


class Bernoulli(EBM):

  def __init__(self, num_bits, initializer, analytic=False):
    super().__init__(num_bits, analytic)
    self._thetas = tf.Variable(initializer([num_bits]))

  @property
  def thetas(self):
    return self._thetas

  @tf.function
  def energy(self, bitstrings):
    dist = tfp.distributions.Bernoulli(logits=2 * self.thetas, dtype=tf.int8)
    return -1 * tf.reduce_sum(dist.log_prob(bitstrings), -1)

  @tf.function
  def sample(self, num_samples):
    dist = tfp.distributions.Bernoulli(logits=2 * self.thetas, dtype=tf.int8)
    return dist.sample(num_samples)

  @tf.function
  def log_partition_function(self):
    return tf.constant(0.0)

  @tf.function
  def entropy(self):
    dist = tfp.distributions.Bernoulli(logits=2 * self.thetas, dtype=tf.int8)
    return tf.reduce_sum(dist.entropy())

  def operators(self, qubits):
    return [cirq.Z(qubits[i]) for i in range(self.num_bits)]


class HOBE(EBM):

  def __init__(self, num_bits, order, initializer, analytic=False):
    super().__init__(num_bits, analytic)
    self.order = order
    self.indices = []
    for i in range(1, order + 1):
      combos = itertools.combinations(range(order), i)
      self.indices.extend([tf.constant(c) for c in combos])
    self._thetas = tf.Variable(initializer([len(self.indices)]))

  @property
  def thetas(self):
    return self._thetas

  @tf.function
  def energy(self, bitstrings):
    spins = 1 - 2 * bitstrings
    parities_t = tf.zeros(
        [len(self.indices), tf.shape(bitstrings)[0]], dtype=tf.float32)
    for i in range(len(self.indices)):
      parity = tf.reduce_prod(tf.gather(spins, self.indices[i], axis=-1), -1)
      parities_t = tf.tensor_scatter_nd_update(parities_t, [[i]], [parity])
    return tf.reduce_sum(tf.transpose(parities_t) * self.thetas, -1)

  def operators(self, qubits):
    operators = []
    for i in range(len(self.indices)):
      operators.append(
          cirq.PauliString(
              cirq.Z(qubits[self.indices[i][j]])
              for j in range(len(self.indices[i]))))
    return operators


class FFN(EBM):

  def __init__(self, num_bits, units, activations, analytic=False):
    super().__init__(num_bits, analytic)
    layers = [tf.keras.layers.InputLayer([num_bits])] + [
        tf.keras.layers.Dense(u, activation=a)
        for u, a in zip(units, activations)
    ] + [tf.keras.layers.Dense(1)]
    self.model = tf.keras.Sequential(layers)
    self.sampler = None

  @property
  def thetas(self):
    return self.model.trainable_variables

  @tf.function
  def energy(self, bitstrings):
    return tf.squeeze(self.model(bitstrings), -1)


class EBMSampler(abc.ABC):

  @property
  @abc.abstractmethod
  def ebm(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def sample(self, num_samples):
    raise NotImplementedError()


GWGKernelResults = collections.namedtuple(
    "GWGKernelResults", ["target_log_prob", "log_acceptance_correction"])


class GWGKernel(tfp.mcmc.TransitionKernel):

  def __init__(self, ebm, grad=True, temp=2.0, num_samples=1):
    super().__init__()
    self.ebm = ebm
    self.grad = grad
    self.diff_function = self.grad_diff_function if grad else self.exact_diff_function
    self.temp = temp
    self.num_samples = num_samples

  @tf.function
  def exact_diff_function(self, current_state):
    current_state_t = tf.transpose(current_state)
    diff_t = tf.zeros_like(current_state_t, dtype=tf.float32)
    current_energy = self.ebm.energy(current_state)

    for i in range(tf.shape(current_state)[-1]):
      pert_state = tf.transpose(
          tf.tensor_scatter_nd_update(current_state_t, [[i]],
                                      tf.expand_dims(1 - current_state_t[i],
                                                     0)))
      diff_t = tf.tensor_scatter_nd_update(
          diff_t, [[i]],
          tf.expand_dims(current_energy - self.ebm.energy(pert_state), 0))
    return tf.transpose(diff_t)

  @tf.function
  def grad_diff_function(self, current_state):
    current_state = tf.cast(current_state, tf.float32)
    with tf.GradientTape() as tape:
      tape.watch(current_state)
      energy = self.ebm.energy(current_state)
    grad = tape.gradient(energy, current_state)
    return (2 * current_state - 1) * grad

  @tf.function
  def one_step(self, current_state, previous_kernel_results):
    forward_diff = self.diff_function(current_state)
    forward_dist = tfp.distributions.OneHotCategorical(
        logits=forward_diff / self.temp, dtype=tf.int8)
    all_changes = forward_dist.sample(self.num_samples)
    total_change = tf.cast(tf.reduce_sum(all_changes, 0) > 0, dtype=tf.int8)
    next_state = tf.bitwise.bitwise_xor(total_change, current_state)

    target_log_prob = -1 * self.ebm.energy(next_state)

    forward_log_prob = tf.reduce_sum(forward_dist.log_prob(all_changes), 0)
    backward_diff = self.diff_function(next_state)
    backward_dist = tfp.distributions.OneHotCategorical(
        logits=backward_diff / self.temp, dtype=tf.int8)
    backward_log_prob = tf.reduce_sum(backward_dist.log_prob(all_changes), 0)
    log_acceptance_correction = backward_log_prob - forward_log_prob

    kernel_results = GWGKernelResults(
        target_log_prob=target_log_prob,
        log_acceptance_correction=log_acceptance_correction)

    return next_state, kernel_results

  @property
  def is_calibrated(self):
    return False

  def bootstrap_results(self, init_state):
    target_log_prob = -1 * self.ebm.energy(init_state)
    kernel_results = GWGKernelResults(
        target_log_prob=target_log_prob,
        log_acceptance_correction=tf.zeros_like(target_log_prob))
    return kernel_results


class MCMC:

  def __init__(self,
               kernel,
               num_chains,
               num_bits,
               buffer_size,
               buffer_prob,
               num_burnin_steps=0,
               num_steps_between_results=0,
               parallel_iterations=10):
    self.kernel = kernel
    self.num_chains = num_chains
    self.num_bits = num_bits
    self.buffer_size = buffer_size
    self.buffer_prob = buffer_prob
    self.num_burnin_steps = num_burnin_steps
    self.num_steps_between_results = num_steps_between_results
    self.parallel_iterations = parallel_iterations

    self.buffer = tf.queue.RandomShuffleQueue(
        buffer_size, 0, [tf.int8], shapes=[num_bits])

  @tf.function
  def sample(self, num_samples):
    num_results = tf.cast(tf.math.ceil(num_samples / self.num_chains), tf.int32)

    if self.buffer.size() == 0 or tf.random.uniform(()) > self.buffer_prob:
      init_state = tf.cast(
          tf.random.uniform([self.num_chains, self.num_bits],
                            maxval=2,
                            dtype=tf.int32), tf.int8)
    else:
      init_state = self.buffer.dequeue_many(self.num_chains)

    previous_kernel_results = self.kernel.bootstrap_results(init_state)

    samples = tfp.mcmc.sample_chain(
        num_results,
        init_state,
        previous_kernel_results=previous_kernel_results,
        kernel=self.kernel,
        num_burnin_steps=self.num_burnin_steps,
        num_steps_between_results=self.num_steps_between_results,
        parallel_iterations=self.parallel_iterations,
        trace_fn=None)
    sampled_states = tf.reshape(samples, [-1, self.num_bits])

    projected_buffer_size = self.buffer.size() + tf.shape(sampled_states)[0]
    if projected_buffer_size > self.buffer_size:
      self.buffer.dequeue_many(projected_buffer_size - self.buffer_size)
    self.buffer.enqueue_many(sampled_states)

    return sampled_states[:num_samples]


class GWG(EBMSampler):

  def __init__(self,
               ebm,
               num_chains,
               buffer_size,
               buffer_prob,
               grad=True,
               temp=2.0,
               num_samples=1,
               num_burnin_steps=0,
               num_steps_between_results=0,
               parallel_iterations=10):
    self._ebm = ebm
    inner_kernel = GWGKernel(
        ebm, grad=grad, temp=temp, num_samples=num_samples)
    kernel = tfp.mcmc.MetropolisHastings(inner_kernel)
    self.mcmc = MCMC(
        kernel,
        num_chains,
        ebm.num_bits,
        buffer_size,
        buffer_prob,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=num_steps_between_results,
        parallel_iterations=parallel_iterations)

  @property
  def ebm(self):
    return self._ebm

  @tf.function
  def sample(self, num_samples):
    return self.mcmc.sample(num_samples)