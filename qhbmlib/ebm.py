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

import abc
import collections
import itertools
from absl import logging

import cirq
import scipy
import tensorflow as tf
import tensorflow_probability as tfp


@tf.function
def probability_to_logit(probability):
  logging.info("retracing: probability_to_logit")
  p = tf.cast(probability, tf.dtypes.float32)
  return tf.math.log(p) - tf.math.log(1 - p)


@tf.function
def logit_to_probability(logit_in):
  logging.info("retracing: logit_to_probability")
  logit = tf.cast(logit_in, tf.dtypes.float32)
  return tf.math.divide(tf.math.exp(logit), 1 + tf.math.exp(logit))


def build_bernoulli(num_nodes, identifier):

  @tf.function
  def energy_bernoulli(logits, bitstring):
    """Calculate the energy of a bitstring against a product of Bernoullis.

    Args:
      logits: 1-D tf.Tensor of dtype float32 containing the logits for each
        Bernoulli factor.
      bitstring: 1-D tf.Tensor of dtype int8 of the form [x_0, ..., x_n-1]. Must
        be the same shape as thetas.

    Returns:
      energy: 0-D tf.Tensor of dtype float32 containing the
        energy of the bitstring calculated as
        sum_i[ln(1+exp(logits_i)) - x_i*logits_i].
    """
    logging.info("retracing: energy_bernoulli_{}".format(identifier))
    bitstring = tf.cast(bitstring, dtype=tf.float32)
    return tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(bitstring, logits))

  @tf.function
  def sampler_bernoulli(thetas, num_samples):
    """Sample bitstrings from a product of Bernoullis.

    Args:
      thetas: 1 dimensional `tensor` of dtype `float32` containing the logits
        for each Bernoulli factor.
      bitstring: 1 dimensional `tensor` of dtype `int8` of the form [x_0, ...,
        x_n-1] so that x is a bitstring.

    Returns:
      bitstrings: `tensor` of dtype `int8` and shape [num_samples, bits]
        where bitstrings are sampled according to
        p(bitstring | thetas) ~ exp(-energy(bitstring | thetas))
    """
    logging.info("retracing: sampler_bernoulli_{}".format(identifier))
    return tfp.distributions.Bernoulli(
        logits=thetas, dtype=tf.int8).sample(num_samples)

  @tf.function
  def log_partition_bernoulli(thetas):
    logging.info("retracing: log_partition_bernoulli_{}".format(identifier))
    # The result is always zero given our definition of the energy.
    return tf.constant(0.0)

  @tf.function
  def entropy_bernoulli(thetas):
    """Calculate the entropy of a product of Bernoullis.

    Args:
        thetas: 1 dimensional `tensor` of dtype `float32` containing the logits
          for each Bernoulli factor.

    Returns:
      entropy: 0 dimensional `tensor` of dtype `float32` containing the
        entropy (in nats) of the distribution.
    """
    logging.info("retracing: entropy_bernoulli_{}".format(identifier))
    return tf.reduce_sum(
        tfp.distributions.Bernoulli(logits=thetas,
                                    dtype=tf.dtypes.int8).entropy())

  return (energy_bernoulli, sampler_bernoulli, log_partition_bernoulli,
          entropy_bernoulli, num_nodes)


def build_boltzmann(num_nodes, identifier):

  if num_nodes > 30:
    raise ValueError("Analytic Boltzmann sampling methods fail past 30 bits.")

  def get_all_boltzmann_sub(num_nodes, identifier):
    flat_spins_mask = tf.cast(
        tf.reshape(
            tf.linalg.band_part(tf.ones([num_nodes, num_nodes]), 0, -1) -
            tf.linalg.diag(tf.ones(num_nodes)), num_nodes * num_nodes), tf.bool)

    @tf.function
    def boltzmann_bits_to_spins(x):
      logging.info("retracing: boltzmann_bits_to_spins_{}".format(identifier))
      return 1 - 2 * x

    @tf.function
    def energy_boltzmann(thetas, x_in):
      logging.info("retracing: energy_boltzmann_{}".format(identifier))
      spins = tf.cast(boltzmann_bits_to_spins(x_in), tf.float32)
      bias_term = tf.reduce_sum(thetas[:num_nodes] * spins)
      w_slice = thetas[num_nodes:]
      spins_outer = tf.matmul(
          tf.transpose(tf.expand_dims(spins, 0)), tf.expand_dims(spins, 0))
      spins_flat = tf.reshape(spins_outer, [num_nodes * num_nodes])
      interaction_spins = tf.boolean_mask(spins_flat, flat_spins_mask)
      interaction_term = tf.reduce_sum(w_slice * interaction_spins)
      return bias_term + interaction_term

    all_strings = tf.constant(
        list(itertools.product([0, 1], repeat=num_nodes)), dtype=tf.int8)

    @tf.function
    def all_energies(thetas):
      logging.info("retracing: all_energies_{}".format(identifier))
      return tf.map_fn(
          lambda x: energy_boltzmann(thetas, x),
          all_strings,
          fn_output_signature=tf.float32)

    @tf.function
    def all_exponentials(thetas):
      logging.info("retracing: all_exponentials_{}".format(identifier))
      return tf.math.exp(
          tf.multiply(tf.constant(-1, dtype=tf.float32), all_energies(thetas)))

    @tf.function
    def partition_boltzmann(thetas):
      logging.info("retracing: partition_boltzmann_{}".format(identifier))
      return tf.reduce_sum(all_exponentials(thetas))

    @tf.function
    def log_partition_boltzmann(thetas):
      logging.info("retracing: log_partition_boltzmann_{}".format(identifier))
      return tf.math.log(partition_boltzmann(thetas))

    @tf.function
    def all_probabilities(thetas):
      logging.info("retracing: all_probabilities_{}".format(identifier))
      return all_exponentials(thetas) / partition_boltzmann(thetas)

    @tf.function
    def sampler_boltzmann(thetas, num_samples):
      logging.info("retracing: sampler_boltzmann_{}".format(identifier))
      Z = partition_boltzmann(thetas)
      exponentials = all_exponentials(thetas)
      raw_samples = tfp.distributions.Categorical(
          logits=tf.multiply(
              tf.constant(-1, dtype=tf.float32), all_energies(thetas)),
          dtype=tf.int32).sample(num_samples)
      return tf.gather(all_strings, raw_samples)

    @tf.function
    def entropy_boltzmann(thetas):
      logging.info("retracing: entropy_boltzmann_{}".format(identifier))
      these_probs = all_probabilities(thetas)
      these_logs = tf.math.log(these_probs)
      return -1.0 * tf.reduce_sum(these_probs * these_logs)

    return (energy_boltzmann, sampler_boltzmann, log_partition_boltzmann,
            entropy_boltzmann, ((num_nodes**2 - num_nodes) // 2) + num_nodes)

  return get_all_boltzmann_sub(num_nodes, identifier)


# ============================================================================ #
# K-local EBM tools.
# ============================================================================ #


@tf.function
def bits_to_spins(x, n_bits):
  logging.info("retracing: bits_to_spins")
  return 1 - 2 * x


def get_parity_index_list(n_bits, k):
  if k < 1:
    raise ValueError("The locality of interactions must be at least 1.")
  if k > n_bits:
    raise ValueError("The locality cannot be greater than the number of bits.")
  index_list = list(range(n_bits))
  return tf.constant(list(itertools.combinations(index_list, k)))


def get_single_locality_parities(n_bits, k):
  indices = get_parity_index_list(n_bits, k)

  @tf.function
  def single_locality_parities(spins):
    logging.info("retracing: single_locality_parities")
    return tf.math.reduce_prod(tf.gather(spins, indices), axis=1)

  return single_locality_parities


def get_single_locality_operators(qubits, k):
  index_list = get_parity_index_list(len(qubits), k)
  op_list = []
  for indices in index_list:
    this_op = cirq.PauliSum().from_pauli_strings(1.0 * cirq.I(qubits[0]))
    for i in indices:
      this_op *= cirq.Z(qubits[i])
    op_list.append(this_op)
  return op_list


def get_all_operators(qubits, max_k):
  """Operators corresponding to `get_klocal_energy_function`"""
  op_list = []
  for k in range(1, max_k + 1):
    op_list += get_single_locality_operators(qubits, k)
  return op_list


def get_all_parities(n_bits, max_k):
  func_list = []
  for k in range(1, max_k + 1):
    func_list.append(get_single_locality_parities(n_bits, k))

  @tf.function
  def all_parities(spins, func_list=func_list):
    logging.info("retracing: all_parities")
    return tf.concat([f(spins) for f in func_list], axis=0)

  return all_parities


def get_klocal_energy_function_num_values(n_bits, max_k):
  n_vals = 0
  for i in range(1, max_k + 1):
    n_vals += scipy.special.comb(n_bits, i, exact=True)
  return n_vals


def get_klocal_energy_function(n_bits, max_k):
  all_parities = get_all_parities(n_bits, max_k)

  @tf.function
  def klocal_energy_function(thetas, x):
    logging.info("retracing: klocal_energy_function")
    spins = bits_to_spins(x, n_bits)
    parities = all_parities(spins)
    return tf.reduce_sum(
        tf.math.multiply(thetas, tf.cast(parities, tf.float32)))

  return klocal_energy_function


# ============================================================================ #
# Swish neural network tools.
# ============================================================================ #


def get_swish_net_hidden_width(num_bits):
  return num_bits + 1 + 2


def get_initial_layer(num_bits):
  """Linear initial layer."""
  w_in = num_bits
  w_out = get_swish_net_hidden_width(num_bits)

  @tf.function
  def initial_layer(thetas, x):
    logging.info("retracing: initial_layer")
    mat = tf.reshape(thetas[:w_in * w_out], [w_out, w_in])
    bias = thetas[w_in * w_out:w_in * w_out + w_out]
    return tf.linalg.matvec(mat, x) + bias

  return initial_layer


def get_hidden_layer(num_bits, i):
  """Swish hidden unit."""
  w = get_swish_net_hidden_width(num_bits)

  @tf.function
  def hidden_layer(thetas, x):
    logging.info("retracing: hidden_layer_{}".format(i))
    mat = tf.reshape(thetas[:w**2], [w, w])
    bias = thetas[w**2:w**2 + w]
    return tf.nn.swish(tf.linalg.matvec(mat, x) + bias)

  return hidden_layer


def get_final_layer(num_bits):
  """Linear final layer."""
  w_in = get_swish_net_hidden_width(num_bits)
  w_out = 1

  @tf.function
  def final_layer(thetas, x):
    logging.info("retracing: final_layer")
    mat = tf.reshape(thetas[:w_in * w_out], [w_out, w_in])
    bias = thetas[w_in * w_out:w_in * w_out + w_out]
    return tf.reduce_sum(tf.linalg.matvec(mat, x) + bias)

  return final_layer


def get_swish_num_values(num_bits, num_layers):
  h_w = get_swish_net_hidden_width(num_bits)
  n_init_params = num_bits * h_w + h_w
  n_hidden_params = h_w**2 + h_w
  n_hidden_params_total = n_hidden_params * num_layers
  n_final_params = h_w + 1
  return n_init_params + n_hidden_params_total + n_final_params


def get_swish_network(num_bits, num_layers):
  """Any function mapping [0,1]^n to R^m can be approximated by

  a Swish network with hidden layer width n+m+2.
  """
  h_w = get_swish_net_hidden_width(num_bits)
  n_init_params = num_bits * h_w + h_w
  n_hidden_params = h_w**2 + h_w
  n_hidden_params_total = n_hidden_params * num_layers
  n_final_params = h_w + 1

  this_initial_layer = get_initial_layer(num_bits)

  def identity(thetas, x):
    return x

  hidden_func = identity

  def get_hidden_stack_inner(previous_func, i):

    def current_hidden_stack(thetas, x):
      cropped_thetas = thetas[i * n_hidden_params:(i + 1) * n_hidden_params]
      return get_hidden_layer(num_bits, i)(cropped_thetas,
                                           previous_func(thetas, x))

    return current_hidden_stack

  for i in range(num_layers):
    hidden_func = get_hidden_stack_inner(hidden_func, i)

  this_final_layer = get_final_layer(num_bits)

  @tf.function
  def swish_network(thetas, x):
    logging.info("retracing: swish_network")
    x = tf.cast(x, tf.float32)
    return this_final_layer(
        thetas[n_init_params + n_hidden_params_total:],
        hidden_func(thetas[n_init_params:n_init_params + n_hidden_params_total],
                    this_initial_layer(thetas[:n_init_params], x)))

  return swish_network


# ============================================================================ #
# Tools for analytic sampling from small energy functions.
# ============================================================================ #


def get_ebm_functions(num_bits, energy_function, ident):
  """Gets functions for exact calculations on energy based models over bits.

  Energy based models (EBMs) are defined by a parameterized energy function,
  E_theta(b), which maps bitstrings to real numbers.  This energy function
  corresponds to a probability distribution
  p(b) = exp(-1.0 * E_theta(b)) / sum_b exp(-1.0 * E_theta(b))

  Args:
    num_bits: number of bits in the samples from the ebm.
    energy_function: function accepting a 1-D `tf.Tensor` of floats and a 1-D
      `tf.Tensor` of ints.  The floats are parameters of an energy calculation,
      and the ints are the bitstring whose energy is calculated.
    ident: Python `str` used to identify functions during tracing.

  Returns:
    sampler_function: function for getting samples from the EBM.
    log_partition_function: function to calculate the natural logarithm of the
      partition function of the EBM.
    entropy_function: function for calculating the entropy of the EBM.
  """
  all_strings = tf.constant(
      list(itertools.product([0, 1], repeat=num_bits)), dtype=tf.int8)

  @tf.function
  def all_energies(thetas):
    """Given the EBM parameters, returns the energy of every bitstring."""
    logging.info("retracing: all_energies_{}".format(ident))
    # TODO(zaqqwerty): get code to be nearly as fast but with less memory
    # overhead.  tf.map_fn seems to get too CPU fragmented.
    return tf.vectorized_map(lambda x: energy_function(thetas, x), all_strings)

  @tf.function
  def sampler_function(thetas, num_samples):
    """Samples from the EBM.

    Args:
      thetas: `tf.Tensor` of DType `tf.float32` which are the parameters of the
        EBM calculation.
      num_samples: Scalar `tf.Tensor` of DType `tf.int32` which is the number of
        samples to draw from the EBM.

    Returns:
      `tf.Tensor` of DType `tf.int8` of shape [num_samples, num_bits] which is
        a list of samples from the EBM.
    """
    logging.info("retracing: sampler_function_{}".format(ident))
    negative_energies = -1.0 * all_energies(thetas)
    raw_samples = tfp.distributions.Categorical(
        logits=negative_energies, dtype=tf.int32).sample(num_samples)
    return tf.gather(all_strings, raw_samples)

  @tf.function
  def log_partition_function(thetas):
    """Calculates the logarithm of the partition function of the EBM.

    Args:
      thetas: `tf.Tensor` of DType `tf.float32` which are the parameters of the
        EBM calculation.

    Returns:
      Scalar `tf.Tensor` of DType `tf.float32` which is the logarithm of the
        partition function of the EBM.
    """
    logging.info("retracing: log_partition_function_{}".format(ident))
    negative_energies = -1.0 * all_energies(thetas)
    return tf.reduce_logsumexp(negative_energies)

  @tf.function
  def entropy_function(thetas):
    """Calculates the entropy of the EBM.

    Args:
      thetas: `tf.Tensor` of DType `tf.float32` which are the parameters of the
        EBM calculation.

    Returns:
      Scalar `tf.Tensor` of DType `tf.float32` which is the entropy of the EBM.
    """
    logging.info("retracing: entropy_function_{}".format(ident))
    negative_energies = -1.0 * all_energies(thetas)
    return tfp.distributions.Categorical(logits=negative_energies).entropy()

  return sampler_function, log_partition_function, entropy_function


# ============================================================================ #
# Tools for MCMC sampling from arbitrary energy functions.
# ============================================================================ #


def get_batched_energy_function(energy_function):
  """Converts a given energy function that takes only a single bitstring as an

  argument to one which can accept batches of bitstrings.
  """

  @tf.function
  def batched_energy_function(energy_function_params, x):
    logging.info("retracing: batched_energy_function")
    if tf.rank(x) == 1:
      return energy_function(energy_function_params, x)
    shape = tf.shape(x)
    return tf.reshape(
        tf.vectorized_map(lambda x: energy_function(energy_function_params, x),
                          tf.reshape(x, [-1, shape[-1]])), shape[:-1])

  return batched_energy_function


BernoulliProposalResults = collections.namedtuple(
    "BernoulliProposalResults",
    ["target_log_prob", "log_acceptance_correction"])


class BernoulliProposal(tfp.mcmc.TransitionKernel):
  """Proposes the next bitstring by flipping the current bitstring according to

  a Bernoulli distribution.
  """

  def __init__(self, energy_function, flip_prob, num_bits):
    super().__init__()
    self.energy_function = get_batched_energy_function(energy_function)
    self.energy_function_params = None
    self.flip_prob = flip_prob
    self.num_bits = num_bits
    self.dist = tfp.distributions.Bernoulli(
        probs=[flip_prob] * num_bits, dtype=tf.int8)

  def set_energy_function_params(self, energy_function_params):
    if self.energy_function_params is None:
      self.energy_function_params = tf.Variable(energy_function_params)
    else:
      self.energy_function_params.assign(energy_function_params)

  @tf.function
  def one_step(self, current_state, previous_kernel_results):
    logging.info("retracing: one_step")
    next_state = tf.bitwise.bitwise_xor(
        self.dist.sample(tf.shape(current_state)[0]), current_state)
    target_log_prob = -1.0 * self.energy_function(self.energy_function_params,
                                                  next_state)
    kernel_results = BernoulliProposalResults(
        target_log_prob=target_log_prob,
        log_acceptance_correction=tf.zeros_like(target_log_prob))
    return next_state, kernel_results

  @property
  def is_calibrated(self):
    return False

  def bootstrap_results(self, init_state):
    target_log_prob = -1.0 * self.energy_function(self.energy_function_params,
                                                  init_state)
    kernel_results = BernoulliProposalResults(
        target_log_prob=target_log_prob,
        log_acceptance_correction=tf.zeros_like(target_log_prob))
    return kernel_results


GibbsWithGradientsProposalResults = collections.namedtuple(
    "GibbsWithGradientsProposalResults",
    ["target_log_prob", "log_acceptance_correction"])


class GibbsWithGradientsProposal(tfp.mcmc.TransitionKernel):
  """Proposes the next bitstring using Gibbs with Gradients."""

  def __init__(self, energy_function, gradient=True, temp=2.0, num_samples=1):
    super().__init__()
    self.energy_function = get_batched_energy_function(energy_function)
    self.energy_function_params = None
    self.gradient = gradient
    self.difference_function = self.gradient_difference_function if gradient else self.exact_difference_function
    self.temp = temp
    self.num_samples = num_samples

  def set_energy_function_params(self, energy_function_params):
    if self.energy_function_params is None:
      self.energy_function_params = tf.Variable(energy_function_params)
    else:
      self.energy_function_params.assign(energy_function_params)

  @tf.function
  def exact_difference_function(self, current_state):
    logging.info("retracing: exact_difference_function")
    current_state_transpose = tf.transpose(current_state)
    diff = tf.zeros_like(current_state_transpose, dtype=tf.float32)
    current_energy = self.energy_function(self.energy_function_params,
                                          current_state)
    for i in range(tf.shape(current_state)[-1]):
      pert_state = tf.transpose(
          tf.tensor_scatter_nd_update(
              current_state_transpose, [[i]],
              tf.expand_dims(1 - current_state_transpose[i], 0)))
      diff = tf.tensor_scatter_nd_update(
          diff, [[i]],
          tf.expand_dims(
              current_energy -
              self.energy_function(self.energy_function_params, pert_state), 0))
    return tf.transpose(diff)

  @tf.function
  def gradient_difference_function(self, current_state):
    logging.info("retracing: gradient_difference_function")
    current_state = tf.cast(current_state, tf.float32)
    with tf.GradientTape() as tape:
      tape.watch(current_state)
      energy = self.energy_function(self.energy_function_params, current_state)
    grad = tape.gradient(energy, current_state)
    return (2 * current_state - 1) * grad

  @tf.function
  def one_step(self, current_state, previous_kernel_results):
    logging.info("retracing: one_step")
    forward_diff = self.difference_function(current_state)
    forward_dist = tfp.distributions.OneHotCategorical(
        logits=forward_diff / self.temp, dtype=tf.int8)
    all_changes = forward_dist.sample(self.num_samples)
    change = tf.cast(tf.reduce_sum(all_changes, 0) > 0, dtype=tf.int8)
    next_state = tf.bitwise.bitwise_xor(change, current_state)

    target_log_prob = -1.0 * self.energy_function(self.energy_function_params,
                                                  next_state)

    forward_log_prob = tf.reduce_sum(forward_dist.log_prob(all_changes), 0)
    backward_diff = self.difference_function(next_state)
    backward_dist = tfp.distributions.OneHotCategorical(
        logits=backward_diff / self.temp, dtype=tf.int8)
    backward_log_prob = tf.reduce_sum(backward_dist.log_prob(all_changes), 0)
    log_acceptance_correction = backward_log_prob - forward_log_prob

    kernel_results = GibbsWithGradientsProposalResults(
        target_log_prob=target_log_prob,
        log_acceptance_correction=log_acceptance_correction)

    return next_state, kernel_results

  @property
  def is_calibrated(self):
    return False

  def bootstrap_results(self, init_state):
    target_log_prob = -1.0 * self.energy_function(self.energy_function_params,
                                                  init_state)
    kernel_results = GibbsWithGradientsProposalResults(
        target_log_prob=target_log_prob,
        log_acceptance_correction=tf.zeros_like(target_log_prob))
    return kernel_results


class MetropolisHastingsMCMC(tfp.mcmc.MetropolisHastings):
  """Wrapper around tpc.mcmc.MetropolisHastings that sets the energy function

  parameters of the inner kernel's energy function.
  """

  def __init__(self, inner_kernel):
    super().__init__(inner_kernel)

  def set_energy_function_params(self, energy_function_params):
    logging.info("retracing: set_energy_function_params")
    self.inner_kernel.set_energy_function_params(energy_function_params)


class MCMCSampler:
  """Samples from an MCMC kernel."""

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
  def __call__(self, energy_function_params, num_samples):
    logging.info("retracing: MCMCSampler")
    self.kernel.set_energy_function_params(energy_function_params)
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


# ============================================================================ #
# General discrete EBM classes.
# ============================================================================ #


class EnergyFunction(abc.ABC):
  """Class for working with discrete energy based models."""

  @property
  @abc.abstractmethod
  def bitwidth(self):
    """Returns the number of bits in samples from this EBM."""

  @property
  @abc.abstractmethod
  def thetas(self):
    """List of trainable variables in this EBM."""

  @abc.abstractmethod
  def energy(self, bitstring):
    """Returns the energy associated to a bitstring.

    Args:
      bitstring: `tf.tensor` of dtype `tf.int8` whose entries are bits.
    
    Returns:
      `tf.tensor` dtype `tf.float32` which is the energy of `bitstring`.
    """


class EBM(EnergyFunction):
  """Class for defining sampling routines for EBMs."""
  
  @abc.abstractmethod
  def sample(self, n_samples):
    """Returns bitstring samples from this EBM.

    Args:
      n_samples: Number of samples to draw from the distribution.
    
    Returns:
      unique_samples: 2D tensor of dtype `tf.int8` whose entries are
        bits. For each `i`, `unique_samples[i]` is some bitstring which
        was sampled from this EBM, and each `unique_samples[i]` is unique.
      counts: 1D tensor of dtype `tf.int32` such that `counts[i]` is the
        number of times `unique_samples[i]` was sampled. `sum(counts)` is
        equal to `n_samples`.
    """

class AnalyticEBM(EBM):
  """Extends EBM to include calculations requiring the normalizing constant."""

  @abc.abstractmethod
  def log_partition_function(self):
    """Returns the logarithm of the partition function of this EBM.
    
    Returns:
      Scalar tensor of dtype `tf.float32`.
    """

  @abc.abstractmethod
  def entropy_function(self):
    """Returns the entropy of this EBM.
    
    Returns:
      Scalar tensor of dtype `tf.float32`.
    """

    
class GeneralAnalyticEBM(AnalyticEBM):
  
  def __init__(self, energy: EnergyFunction):
    self.energy_function = energy
    self.all_bitstrings = tf.constant(
          list(itertools.product([0, 1], repeat=num_bits)), dtype=tf.int8)

  @property
  def bitwidth(self):
    return self.energy.bitwidth

  @property
  def thetas(self):
    return self.energy.thetas

  def energy(self, bitstring):
    return self.energy.energy

  def sample(self):
    all_energies = self.energy(self.all_bitstrings)
    dist = tfp.distributions.Categorical(
      logits=-1 * all_energies, dtype=tf.int8)
    return tf.gather(self.all_bitstrings, dist.sample(num_samples))

  def log_partition_function(self):
    all_energies = self.energy(self.all_bitstrings)
    return tf.reduce_logsumexp(-1 * all_energies)

  def entropy_function(self):
    all_energies = self.energy(self.all_bitstrings)
    dist = tfp.distributions.Categorical(logits=-1 * all_energies)
    return dist.entropy()


class Bernoulli(AnalyticEBM):
  def __init__(self, num_bits, initializer, analytic=False):
    self._num_bits = num_bits
    self._thetas = tf.Variable(initializer([num_bits]))

  @property
  def bitwidth(self):
    return self._num_bits

  @property
  def thetas(self):
    return self._thetas

  def energy(self, bitstrings):
    dist = tfp.distributions.Bernoulli(logits=2 * self.thetas, dtype=tf.int8)
    return -1 * tf.reduce_sum(dist.log_prob(bitstrings), -1)

  def sample(self, num_samples):
    dist = tfp.distributions.Bernoulli(logits=2 * self.thetas, dtype=tf.int8)
    return dist.sample(num_samples)

  def log_partition_function(self):
    return tf.constant(0.0)

  def entropy(self):
    dist = tfp.distributions.Bernoulli(logits=2 * self.thetas, dtype=tf.int8)
    return tf.reduce_sum(dist.entropy())
