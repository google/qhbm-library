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

import cirq
import scipy
import tensorflow as tf
import tensorflow_probability as tfp


@tf.function
def probability_to_logit(probability):
    print("retracing: probability_to_logit")
    p = tf.cast(probability, tf.dtypes.float32)
    return tf.math.log(p) - tf.math.log(1 - p)


@tf.function
def logit_to_probability(logit_in):
    print("retracing: logit_to_probability")
    logit = tf.cast(logit_in, tf.dtypes.float32)
    return tf.math.divide(tf.math.exp(logit), 1 + tf.math.exp(logit))


def build_bernoulli(num_nodes, identifier):
    @tf.function
    def energy_bernoulli(logits, bitstring):
        """Calculate the energy of a bitstring against a product of Bernoullis.

        Args:
          logits: 1-D tf.Tensor of dtype float32 containing the
            logits for each Bernoulli factor.
          bitstring: 1-D tf.Tensor of dtype bool of the
            form [x_0, ..., x_n-1].  Must be the same shape as thetas.

        Returns:
          energy: 0-D tf.Tensor of dtype float32 containing the
            energy of the bitstring calculated as
            sum_i[ln(1+exp(logits_i)) - x_i*logits_i].
        """
        print("retracing: energy_bernoulli_{}".format(identifier))
        bitstring = tf.cast(bitstring, dtype=tf.float32)
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(bitstring, logits))

    @tf.function
    def sampler_bernoulli(thetas, num_samples):
        """Sample bitstrings from a product of Bernoullis.

        Args:
          thetas: 1 dimensional `tensor` of dtype `float32` containing the
            logits for each Bernoulli factor.
          bitstring: 1 dimensional `tensor` of dtype `bool` of the form
            [x_0, ..., x_n-1] so that x is a bitstring.

        Returns:
          bitstrings: `tensor` of dtype `bool` and shape [num_samples, bits]
            where bitstrings are sampled according to
            p(bitstring | thetas) ~ exp(-energy(bitstring | thetas))
        """
        print("retracing: sampler_bernoulli_{}".format(identifier))
        return tfp.distributions.Bernoulli(logits=thetas, dtype=tf.dtypes.int8).sample(
            num_samples
        )

    @tf.function
    def log_partition_bernoulli(thetas):
        print("retracing: log_partition_bernoulli_{}".format(identifier))
        # The result is always zero given our definition of the energy.
        return tf.constant(0.0)

    @tf.function
    def entropy_bernoulli(thetas):
        """Calculate the entropy of a product of Bernoullis.

        Args:
            thetas: 1 dimensional `tensor` of dtype `float32` containing the
              logits for each Bernoulli factor.

        Returns:
          entropy: 0 dimensional `tensor` of dtype `float32` containing the
            entropy (in nats) of the distribution.
        """
        print("retracing: entropy_bernoulli_{}".format(identifier))
        return tf.reduce_sum(
            tfp.distributions.Bernoulli(logits=thetas, dtype=tf.dtypes.int8).entropy()
        )

    return (
        energy_bernoulli,
        sampler_bernoulli,
        log_partition_bernoulli,
        entropy_bernoulli,
        num_nodes,
    )


def build_boltzmann(num_nodes, identifier):

    if num_nodes > 30:
        raise ValueError("Analytic Boltzmann sampling methods fail past 30 bits.")

    def get_all_boltzmann_sub(num_nodes, identifier):
        flat_spins_mask = tf.cast(
            tf.reshape(
                tf.linalg.band_part(tf.ones([num_nodes, num_nodes]), 0, -1)
                - tf.linalg.diag(tf.ones(num_nodes)),
                num_nodes * num_nodes,
            ),
            tf.bool,
        )

        @tf.function
        def boltzmann_bits_to_spins(x):
            print("retracing: boltzmann_bits_to_spins_{}".format(identifier))
            return tf.subtract(
                tf.ones(num_nodes, dtype=tf.int8),
                tf.constant(2, dtype=tf.int8) * tf.cast(x, tf.int8),
            )

        @tf.function
        def energy_boltzmann(thetas, x_in):
            print("retracing: energy_boltzmann_{}".format(identifier))
            spins = tf.cast(boltzmann_bits_to_spins(x_in), tf.float32)
            bias_term = tf.reduce_sum(thetas[:num_nodes] * spins)
            w_slice = thetas[num_nodes:]
            spins_outer = tf.matmul(
                tf.transpose(tf.expand_dims(spins, 0)), tf.expand_dims(spins, 0)
            )
            spins_flat = tf.reshape(spins_outer, [num_nodes * num_nodes])
            interaction_spins = tf.boolean_mask(spins_flat, flat_spins_mask)
            interaction_term = tf.reduce_sum(w_slice * interaction_spins)
            return bias_term + interaction_term

        all_strings = tf.constant(
            list(itertools.product([0, 1], repeat=num_nodes)), dtype=tf.int8
        )

        @tf.function
        def all_energies(thetas):
            print("retracing: all_energies_{}".format(identifier))
            return tf.map_fn(
                lambda x: energy_boltzmann(thetas, x),
                all_strings,
                fn_output_signature=tf.float32,
            )

        @tf.function
        def all_exponentials(thetas):
            print("retracing: all_exponentials_{}".format(identifier))
            return tf.math.exp(
                tf.multiply(tf.constant(-1, dtype=tf.float32), all_energies(thetas))
            )

        @tf.function
        def partition_boltzmann(thetas):
            print("retracing: partition_boltzmann_{}".format(identifier))
            return tf.reduce_sum(all_exponentials(thetas))

        @tf.function
        def log_partition_boltzmann(thetas):
            print("retracing: log_partition_boltzmann_{}".format(identifier))
            return tf.math.log(partition_boltzmann(thetas))

        @tf.function
        def all_probabilities(thetas):
            print("retracing: all_probabilities_{}".format(identifier))
            return all_exponentials(thetas) / partition_boltzmann(thetas)

        @tf.function
        def sampler_boltzmann(thetas, num_samples):
            print("retracing: sampler_boltzmann_{}".format(identifier))
            raw_samples = tfp.distributions.Categorical(
                logits=tf.multiply(
                    tf.constant(-1, dtype=tf.float32), all_energies(thetas)
                ),
                dtype=tf.int32,
            ).sample(num_samples)
            return tf.gather(all_strings, raw_samples)

        @tf.function
        def entropy_boltzmann(thetas):
            print("retracing: entropy_boltzmann_{}".format(identifier))
            these_probs = all_probabilities(thetas)
            these_logs = tf.math.log(these_probs)
            return -1.0 * tf.reduce_sum(these_probs * these_logs)

        return (
            energy_boltzmann,
            sampler_boltzmann,
            log_partition_boltzmann,
            entropy_boltzmann,
            ((num_nodes ** 2 - num_nodes) // 2) + num_nodes,
        )

    return get_all_boltzmann_sub(num_nodes, identifier)


# ============================================================================ #
# K-local EBM tools.
# ============================================================================ #


@tf.function
def bits_to_spins(x, n_bits):
    print("retracing: bits_to_spins")
    return tf.subtract(
        tf.ones(n_bits, dtype=tf.int8), tf.constant(2, dtype=tf.int8) * x
    )


def get_parity_index_list(n_bits, k):
    if k < 1:
        raise ValueError("The locality of interactions must be at least 1.")
    if k > n_bits:
        raise ValueError("The locality cannot be greater than the number of bits.")
    index_list = [i for i in range(n_bits)]
    return tf.constant(list(itertools.combinations(index_list, k)))


def get_single_locality_parities(n_bits, k):
    indices = get_parity_index_list(n_bits, k)

    @tf.function
    def single_locality_parities(spins):
        print("retracing: single_locality_parities")
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
        print("retracing: all_parities")
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
        print("retracing: klocal_energy_function")
        spins = bits_to_spins(x, n_bits)
        parities = all_parities(spins)
        return tf.reduce_sum(tf.math.multiply(thetas, tf.cast(parities, tf.float32)))

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
        print("retracing: initial_layer")
        mat = tf.reshape(thetas[: w_in * w_out], [w_out, w_in])
        bias = thetas[w_in * w_out : w_in * w_out + w_out]
        return tf.linalg.matvec(mat, x) + bias

    return initial_layer


def get_hidden_layer(num_bits, i):
    """Swish hidden unit."""
    w = get_swish_net_hidden_width(num_bits)

    @tf.function
    def hidden_layer(thetas, x):
        print("retracing: hidden_layer_{}".format(i))
        mat = tf.reshape(thetas[: w ** 2], [w, w])
        bias = thetas[w ** 2 : w ** 2 + w]
        return tf.nn.swish(tf.linalg.matvec(mat, x) + bias)

    return hidden_layer


def get_final_layer(num_bits):
    """Linear final layer."""
    w_in = get_swish_net_hidden_width(num_bits)
    w_out = 1

    @tf.function
    def final_layer(thetas, x):
        print("retracing: final_layer")
        mat = tf.reshape(thetas[: w_in * w_out], [w_out, w_in])
        bias = thetas[w_in * w_out : w_in * w_out + w_out]
        return tf.reduce_sum(tf.linalg.matvec(mat, x) + bias)

    return final_layer


def get_swish_num_values(num_bits, num_layers):
    h_w = get_swish_net_hidden_width(num_bits)
    n_init_params = num_bits * h_w + h_w
    n_hidden_params = h_w ** 2 + h_w
    n_hidden_params_total = n_hidden_params * num_layers
    n_final_params = h_w + 1
    return n_init_params + n_hidden_params_total + n_final_params


def get_swish_network(num_bits, num_layers):
    """Any function mapping [0,1]^n to R^m can be approximated by
    a Swish network with hidden layer width n+m+2."""
    h_w = get_swish_net_hidden_width(num_bits)
    n_init_params = num_bits * h_w + h_w
    n_hidden_params = h_w ** 2 + h_w
    n_hidden_params_total = n_hidden_params * num_layers

    this_initial_layer = get_initial_layer(num_bits)

    def identity(thetas, x):
        return x

    hidden_func = identity

    def get_hidden_stack_inner(previous_func, i):
        def current_hidden_stack(thetas, x):
            cropped_thetas = thetas[i * n_hidden_params : (i + 1) * n_hidden_params]
            return get_hidden_layer(num_bits, i)(
                cropped_thetas, previous_func(thetas, x)
            )

        return current_hidden_stack

    for i in range(num_layers):
        hidden_func = get_hidden_stack_inner(hidden_func, i)

    this_final_layer = get_final_layer(num_bits)

    @tf.function
    def swish_network(thetas, x):
        print("retracing: swish_network")
        x = tf.cast(x, tf.float32)
        return this_final_layer(
            thetas[n_init_params + n_hidden_params_total :],
            hidden_func(
                thetas[n_init_params : n_init_params + n_hidden_params_total],
                this_initial_layer(thetas[:n_init_params], x),
            ),
        )

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
      energy_function: function accepting a 1-D `tf.Tensor` of floats and
        a 1-D `tf.Tensor` of bools.  The floats are parameters of an energy
        calculation, and the bools are the bitstring whose energy is calculated.
      ident: Python `str` used to identify functions during tracing.

    Returns:
      sampler_function: function for getting samples from the EBM.
      log_partition_function: function to calculate the natural logarithm of the
        partition function of the EBM.
      entropy_function: function for calculating the entropy of the EBM.
    """
    all_strings = tf.constant(
        list(itertools.product([0, 1], repeat=num_bits)), dtype=tf.int8
    )

    @tf.function
    def all_energies(thetas):
        """Given the EBM parameters, returns the energy of every bitstring."""
        print("retracing: all_energies_{}".format(ident))
        # TODO(zaqqwerty): get code to be nearly as fast but with less memory
        # overhead.  tf.map_fn seems to get too CPU fragmented.
        return tf.vectorized_map(lambda x: energy_function(thetas, x), all_strings)

    @tf.function
    def sampler_function(thetas, num_samples):
        """Samples from the EBM.

        Args:
          thetas: `tf.Tensor` of DType `tf.float32` which are the parameters of
            the EBM calculation.
          num_samples: Scalar `tf.Tensor` of DType `tf.int32` which is the number
            of samples to draw from the EBM.

        Returns:
          `tf.Tensor` of DType `tf.int8` of shape [num_samples, num_bits] which is
            a list of samples from the EBM.
        """
        print("retracing: sampler_function_{}".format(ident))
        negative_energies = -1.0 * all_energies(thetas)
        raw_samples = tfp.distributions.Categorical(
            logits=negative_energies, dtype=tf.int32
        ).sample(num_samples)
        return tf.gather(all_strings, raw_samples)

    @tf.function
    def log_partition_function(thetas):
        """Calculates the logarithm of the partition function of the EBM.

        Args:
          thetas: `tf.Tensor` of DType `tf.float32` which are the parameters of
            the EBM calculation.

        Returns:
          Scalar `tf.Tensor` of DType `tf.float32` which is the logarithm of the
            partition function of the EBM.
        """
        print("retracing: log_partition_function_{}".format(ident))
        negative_energies = -1.0 * all_energies(thetas)
        return tf.reduce_logsumexp(negative_energies)

    @tf.function
    def entropy_function(thetas):
        """Calculates the entropy of the EBM.

        Args:
          thetas: `tf.Tensor` of DType `tf.float32` which are the parameters of
            the EBM calculation.

        Returns:
          Scalar `tf.Tensor` of DType `tf.float32` which is the entropy of the EBM.
        """
        print("retracing: entropy_function_{}".format(ident))
        negative_energies = -1.0 * all_energies(thetas)
        return tfp.distributions.Categorical(logits=negative_energies).entropy()

    return sampler_function, log_partition_function, entropy_function


# ============================================================================ #
# Tools for MCMC sampling from arbitrary energy functions.
# ============================================================================ #


BitstringKernelResults = collections.namedtuple(
    "BitstringKernelResults",
    ["target_log_prob", "log_acceptance_correction", "energy_function_params"],
)


class BitstringKernel(tfp.mcmc.TransitionKernel):
    def __init__(self, proposal_function, energy_function):
        super().__init__()
        self.proposal_function = proposal_function
        self.energy_function = energy_function

    @tf.function
    def one_step(self, current_state, previous_kernel_results):
        print("retracing: one_step")
        next_state = self.proposal_function(current_state)
        kernel_results = BitstringKernelResults(
            target_log_prob=-1.0
            * self.energy_function(
                previous_kernel_results.energy_function_params, next_state
            ),
            log_acceptance_correction=tf.constant(0.0),
            energy_function_params=previous_kernel_results.energy_function_params,
        )
        return next_state, kernel_results

    def bootstrap_results(self, init_state, energy_function_params):
        kernel_results = BitstringKernelResults(
            target_log_prob=-1.0
            * self.energy_function(energy_function_params, init_state),
            log_acceptance_correction=tf.constant(0.0),
            energy_function_params=energy_function_params,
        )
        return kernel_results

    def is_calibrated(self):
        return False


MCMCBitstringKernelResults = collections.namedtuple(
    "MCMCBitstringKernelResults",
    [
        "accepted_results",
        "is_accepted",
        "log_accept_ratio",
        "proposed_state",
        "proposed_results",
    ],
)


class MCMCBitstringKernel(tfp.mcmc.TransitionKernel):
    """Adapted from the TFP MetropolisHastings kernel."""

    def __init__(self, inner_kernel):
        super().__init__()
        self.inner_kernel = inner_kernel

    @tf.function
    def one_step(self, current_state, previous_kernel_results):
        print("retracing: one_step")
        [proposed_state, proposed_results] = self.inner_kernel.one_step(
            current_state, previous_kernel_results.accepted_results
        )
        log_accept_ratio = (
            proposed_results.target_log_prob
            - previous_kernel_results.accepted_results.target_log_prob
            + proposed_results.log_acceptance_correction
        )
        # If proposed state reduces likelihood: randomly accept.
        # If proposed state increases likelihood: always accept.
        # I.e., u < min(1, accept_ratio),  where u ~ Uniform[0,1)
        #       ==> log(u) < log_accept_ratio
        log_uniform = tf.math.log(
            tf.random.uniform(
                tf.shape(proposed_results.target_log_prob),
                minval=0,
                maxval=None,
                dtype=proposed_results.target_log_prob.dtype,
            )
        )
        is_accepted = log_uniform < log_accept_ratio
        next_state = tf.where(is_accepted, proposed_state, current_state)
        accepted_results = tf.cond(
            is_accepted,
            lambda: proposed_results,
            lambda: previous_kernel_results.accepted_results,
        )
        kernel_results = MCMCBitstringKernelResults(
            accepted_results=accepted_results,
            is_accepted=is_accepted,
            log_accept_ratio=log_accept_ratio,
            proposed_state=proposed_state,
            proposed_results=proposed_results,
        )
        return next_state, kernel_results

    def bootstrap_results(self, init_state, energy_function_params):
        pkr = self.inner_kernel.bootstrap_results(init_state, energy_function_params)
        x = pkr.target_log_prob
        return MCMCBitstringKernelResults(
            accepted_results=pkr,
            is_accepted=tf.ones_like(x, dtype=tf.bool),
            log_accept_ratio=tf.zeros_like(x),
            proposed_state=init_state,
            proposed_results=pkr,
        )

    def is_calibrated(self):
        return True


def get_bernoulli_proposal_function(flip_prob, num_qubits):
    dist = tfp.distributions.Bernoulli(probs=[flip_prob] * num_qubits, dtype=tf.bool)

    @tf.function
    def bernoulli_proposal_function(current_state):
        print("retracing: bernoulli_proposal_function")
        mask = dist.sample(1)[0]
        return tf.math.logical_xor(current_state, mask)

    return bernoulli_proposal_function


def get_chain(
    proposal_function,
    energy_function,
    num_burnin_steps,
    num_steps_between_results,
    parallel_iterations,
):
    inner_kernel = BitstringKernel(proposal_function, energy_function)
    mcmc_kernel = MCMCBitstringKernel(inner_kernel)

    @tf.function
    def run_chain(last_state, energy_function_params, num_steps):
        print("retracing: run_chain")
        last_state = tf.cast(last_state, tf.bool)
        return tf.cast(
            tfp.mcmc.sample_chain(
                num_results=num_steps,
                num_burnin_steps=num_burnin_steps,
                num_steps_between_results=num_steps_between_results,
                parallel_iterations=parallel_iterations,
                current_state=last_state,
                kernel=mcmc_kernel,
                previous_kernel_results=mcmc_kernel.bootstrap_results(
                    last_state, energy_function_params
                ),
                trace_fn=None,
            ),
            tf.int8,
        )

    return run_chain
