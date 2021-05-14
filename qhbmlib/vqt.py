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
"""Impementations of the VQT loss and its derivatives."""

import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib import qhbm_base


# ============================================================================ #
# Sample-based VQT.
# ============================================================================ #


@tf.function
def vqt_loss(
    qhbm: qhbm_base.QHBM,
    num_samples: tf.Tensor,
    beta: tf.Tensor,
    hamiltonian: tf.Tensor,
):
    """Estimates the VQT loss function between a QHBM and a Hamiltonian.

    The two step process follows the VQT loss estimator described in the paper.
    First, samples are drawn from the classical probability distribution of the
    input QHBM.  Then, two quantities are estimated from these samples: the
    entropy of the distribution is calculated by a frequency estimator, and the
    energy is calculated via the Monte-Carlo estimator defined in the paper.

    Args:
      qhbm: The QHBM against which the loss is to be calculated.
      num_samples: The number of samples to draw from the classical probability
        distribution of the QHBM.
      beta: Inverse temperature of the target thermal state.
      hamiltonian: Hamiltonian of the target thermal state.
        It is a 2-D tensor of strings, the result of calling
        `tfq.convert_to_tensor` on a list of lists of cirq.PauliSum which has only
        one term, `[[op]]`.

    Returns:
      Estimate of the VQT loss between the input QHBM and target thermal state.
    """
    samples, counts = qhbm.sample_bitstrings(num_samples)
    prob_terms = tf.cast(counts, tf.float32) / tf.cast(num_samples, tf.float32)
    log_prob_terms = tf.math.log(prob_terms)
    minus_entropy_approx = tf.reduce_sum(prob_terms * log_prob_terms)
    num_circuits = tf.shape(counts)[0]
    state_circuits = qhbm.state_circuits(samples)
    tiled_hamiltonian = tf.tile(hamiltonian, [num_circuits, 1])
    expectations = tf.reshape(
        tfq.layers.SampledExpectation()(
            state_circuits,
            symbol_names=tf.constant([], dtype=tf.string),
            symbol_values=tf.tile(
                tf.constant([[]], dtype=tf.float32), [num_circuits, 1]
            ),
            operators=tiled_hamiltonian,
            repetitions=tf.expand_dims(counts, 1),
        ),
        [num_circuits],
    )
    energy_approx = tf.reduce_sum(expectations * prob_terms)
    return minus_entropy_approx + beta * energy_approx


@tf.function
def vqt_loss_thetas_grad(
    qhbm: qhbm_base.QHBM,
    num_samples: tf.Tensor,
    beta: tf.Tensor,
    hamiltonian: tf.Tensor,
):
    """Sample-based estimate of classical param gradient of VQT loss."""
    print("retracing: vqt_loss_thetas_grad")
    # Sample circuits to average the gradient over.
    circuits, bitstrings, counts = qhbm.sample_state_circuits(num_samples)
    num_samples = tf.cast(num_samples, tf.float32)
    num_circuits = tf.shape(circuits)[0]
    counts = tf.cast(counts, tf.float32)

    # Compute the per bitstring difference between beta H and energy
    tiled_hamiltonian = tf.tile(hamiltonian, [num_circuits, 1])
    expectations = tf.reshape(
        tfq.layers.SampledExpectation()(
            circuits,
            symbol_names=tf.constant([], dtype=tf.string),
            symbol_values=tf.tile(
                tf.constant([[]], dtype=tf.float32), [num_circuits, 1]
            ),
            operators=tiled_hamiltonian,
            repetitions=tf.expand_dims(counts, 1),
        ),
        [num_circuits],
    )
    beta_h = beta * expectations
    energies, energy_grads = tf.map_fn(
        qhbm.energy_and_energy_grad,
        bitstrings,
        fn_output_signature=(tf.float32, tf.float32),
    )
    diff_list = beta_h - energies

    part_a = tf.divide(tf.reduce_sum(counts * diff_list), num_samples)
    part_b = tf.divide(
        tf.reduce_sum(tf.expand_dims(counts, 1) * energy_grads, 0), num_samples
    )
    prod_list = tf.expand_dims(diff_list, 1) * energy_grads
    part_c = tf.divide(
        tf.reduce_sum(tf.expand_dims(counts, 1) * prod_list, 0), num_samples
    )

    return part_a * part_b - part_c


@tf.function
def vqt_loss_phis_grad(
    qhbm: qhbm_base.QHBM,
    num_samples: tf.Tensor,
    beta: tf.Tensor,
    hamiltonian: tf.Tensor,
):
    """Sample-based estimate of quantum circuit param gradient of VQT loss."""
    print("retracing: vqt_loss_phis_grad")
    circuits, _, counts = qhbm.sample_unresolved_state_circuits(num_samples)
    num_circuits = tf.shape(counts)[0]
    new_dup_phis = tf.identity(qhbm.phis)
    with tf.GradientTape() as tape:
        tape.watch(new_dup_phis)
        sub_energy_list = tf.reshape(
            tfq.layers.Expectation()(
                circuits,
                symbol_names=qhbm.phis_symbols,
                symbol_values=tf.tile(
                    tf.expand_dims(new_dup_phis, 0), [num_circuits, 1]
                ),
                operators=tf.tile(hamiltonian, [num_circuits, 1]),
            ),
            [num_circuits],
        )
        e_avg = tf.divide(
            tf.reduce_sum(tf.cast(counts, tf.float32) * sub_energy_list),
            tf.cast(num_samples, tf.float32),
        )
    this_grad = tape.gradient(e_avg, new_dup_phis)
    return beta * this_grad


# ============================================================================ #
# Exact VQT.
# ============================================================================ #


def _tiled_expectation(circuits: tf.Tensor, hamiltonian: tf.Tensor):
    """Calculates the expectation value for every circuit.

    Args:
      circuits: 1-D tensor of strings which are TFQ serialized circuits with
        no free parameters.
      hamiltonian: 2-D tensor of strings, the result of calling
        `tfq.convert_to_tensor` on a list of lists of cirq.PauliSum which has only
        one term, `[[op]]`.  Will be tiled along the 0th dimension, to be measured
        against each entry of `circuits`.

    Returns:
      1-D tensor of floats which are the expectation values of the single op in
        `hamiltonian` measured against each circuit in `circuits`.
    """
    num_circuits = tf.shape(circuits)[0]
    measurements = tf.tile(hamiltonian, [num_circuits, 1])
    return tf.reshape(
        tfq.layers.Expectation()(
            circuits,
            symbol_names=tf.constant([], dtype=tf.string),
            symbol_values=tf.tile(
                tf.constant([[]], dtype=tf.float32), [tf.shape(circuits)[0], 1]
            ),
            operators=measurements,
        ),
        [num_circuits],
    )


@tf.function
def exact_vqt_loss(
    qhbm: qhbm_base.ExactQHBM,
    num_samples: tf.Tensor,
    beta: tf.Tensor,
    hamiltonian: tf.Tensor,
):
    """Calculates the VQT loss function between an ExactQHBM and a hamiltonian."""
    print("retracing: exact_vqt_loss")
    circuits, _, counts = qhbm.sample_state_circuits(num_samples)
    sub_energy_list = _tiled_expectation(circuits, hamiltonian)
    weighted_energy_list = tf.cast(counts, tf.float32) * sub_energy_list
    e_avg = tf.divide(
        tf.reduce_sum(weighted_energy_list), tf.cast(num_samples, tf.float32)
    )
    return beta * e_avg - qhbm.entropy_function()


@tf.function
def exact_vqt_loss_thetas_grad(
    qhbm: qhbm_base.ExactQHBM,
    num_samples: tf.Tensor,
    beta: tf.Tensor,
    hamiltonian: tf.Tensor,
):
    """Gradient of exact VQT loss function with respect to classical params."""
    print("retracing: exact_vqt_loss_thetas_grad")
    # Sample circuits to average the gradient over.
    circuits, bitstrings, counts = qhbm.sample_state_circuits(num_samples)
    num_samples = tf.cast(num_samples, tf.float32)
    counts = tf.cast(counts, tf.float32)

    # Compute the per bitstring difference between beta H and energy
    beta_h = beta * _tiled_expectation(circuits, hamiltonian)
    energies, energy_grads = tf.map_fn(
        qhbm.energy_and_energy_grad,
        bitstrings,
        fn_output_signature=(tf.float32, tf.float32),
    )
    diff_list = beta_h - energies

    part_a = tf.divide(tf.reduce_sum(counts * diff_list), num_samples)
    part_b = tf.divide(
        tf.reduce_sum(tf.expand_dims(counts, 1) * energy_grads, 0), num_samples
    )
    prod_list = tf.expand_dims(diff_list, 1) * energy_grads
    part_c = tf.divide(
        tf.reduce_sum(tf.expand_dims(counts, 1) * prod_list, 0), num_samples
    )

    return part_a * part_b - part_c


@tf.function
def exact_vqt_loss_phis_grad(
    qhbm: qhbm_base.ExactQHBM,
    num_samples: tf.Tensor,
    beta: tf.Tensor,
    hamiltonian: tf.Tensor,
):
    """Gradient of exact VQT loss function with respect to circuit params."""
    print("retracing: exact_vqt_loss_phis_grad")
    circuits, _, counts = qhbm.sample_unresolved_state_circuits(num_samples)
    num_circuits = tf.shape(counts)[0]
    new_dup_phis = tf.identity(qhbm.phis)
    with tf.GradientTape() as tape:
        tape.watch(new_dup_phis)
        sub_energy_list = tf.reshape(
            tfq.layers.Expectation()(
                circuits,
                symbol_names=qhbm.phis_symbols,
                symbol_values=tf.tile(
                    tf.expand_dims(new_dup_phis, 0), [num_circuits, 1]
                ),
                operators=tf.tile(hamiltonian, [num_circuits, 1]),
            ),
            [num_circuits],
        )
        e_avg = tf.divide(
            tf.reduce_sum(tf.cast(counts, tf.float32) * sub_energy_list),
            tf.cast(num_samples, tf.float32),
        )
    this_grad = tape.gradient(e_avg, new_dup_phis)
    return beta * this_grad


# ============================================================================ #
# Version of VQT that takes a list of QHBMs as the target Hamiltonian.
# ============================================================================ #

# TODO(#14)
# @tf.function
# def get_avg_sub_energy(
#     full_y_samples_j, qhbm_energy_function, qhbm_thetas_j, bit_sub_indices_j
# ):
#     print("retracing: get_avg_sub_energy")
#     sub_samples = tf.gather(full_y_samples_j, bit_sub_indices_j, axis=1)
#     unique_sub_samples, _, counts = qhbm_base.unique_with_counts(sub_samples)
#     counts = tf.cast(counts, tf.float32)
#     unique_sub_energies, _ = tf.map_fn(
#         lambda x: qhbm.energy_and_energy_grad(qhbm_energy_function, qhbm_thetas_j, x),
#         unique_sub_samples,
#         fn_output_signature=(tf.float32, tf.float32),
#     )
#     sub_energies = tf.multiply(counts, unique_sub_energies)
#     return tf.divide(tf.reduce_sum(sub_energies), tf.reduce_sum(counts))


# def build_sub_term_energy_func(qhbm_vqt, qhbm_hamiltonian_list):
#     bit_sub_indices_list = [
#         util.get_bit_sub_indices(qhbm_vqt.qubits, this_qhbm.qubits)
#         for this_qhbm in qhbm_hamiltonian_list
#     ]
#     qhbm_thetas = [this_qhbm.thetas for this_qhbm in qhbm_hamiltonian_list]
#     qhbm_u_dagger = [this_qhbm.u_dagger for this_qhbm in qhbm_hamiltonian_list]
#     qhbm_phis_symbols=[this_qhbm.phis_symbols for this_qhbm in qhbm_hamiltonian_list]
#     qhbm_phis = [this_qhbm.phis for this_qhbm in qhbm_hamiltonian_list]
#     qhbm_energy_list = [
#         this_qhbm.energy_function for this_qhbm in qhbm_hamiltonian_list
#     ]

#     def inner_build_sub_term_energy_func(
#         energy_list, thetas, sub_indices, daggers, phis_s, phis
#     ):
#         list_of_get_avg_energy_lists = []

#         def inner_jth_sampled_circuits_energy_list(
#             j, i_energy_list, i_thetas, i_sub_indices, i_daggers, i_phis_s, i_phis
#         ):
#             @tf.function
#             def jth_sampled_circuits_energy_list(circuits, counts):
#                 print("retracing: jth_sampled_circuits_energy_list_{}".format(j))
#                 return tf.map_fn(
#                     lambda x: get_avg_sub_energy(
#                         x.to_tensor(), energy_list[j], thetas[j], sub_indices[j]
#                     ),
#                     qhbm.sample_pulled_back_bitstrings(
#                         daggers[j], phis_s[j], phis[j], circuits, counts
#                     ),
#                     fn_output_signature=tf.float32,
#                 )

#             return jth_sampled_circuits_energy_list

#         for j, _ in enumerate(energy_list):
#             list_of_get_avg_energy_lists.append(
#                 inner_jth_sampled_circuits_energy_list(
#                     j, energy_list, thetas, sub_indices, daggers, phis_s, phis
#                 )
#             )

#         @tf.function
#         def sub_term_energy_func(
#             vqt_circuits, vqt_circuits_counts, le=list_of_get_avg_energy_lists
#         ):
#             print("retracing: sub_term_energy_func")
#             avg_energy_lists = tf.stack(
#                 [f(vqt_circuits, vqt_circuits_counts) for f in le]
#             )
#             return tf.reduce_sum(avg_energy_lists, 0)

#         return sub_term_energy_func

#     return inner_build_sub_term_energy_func(
#         qhbm_energy_list,
#         qhbm_thetas,
#         bit_sub_indices_list,
#         qhbm_u_dagger,
#         qhbm_phis_symbols,
#         qhbm_phis,
#     )


# @tf.function
# def vqt_qhbm_loss(qhbm_vqt, num_vqt_samples, sub_term_energy_func):
#     print("retracing: vqt_qhbm_loss")
#     vqt_circuits, _, vqt_circuits_counts = qhbm_vqt.sample_state_circuits(
#         num_vqt_samples,
#     )
#     sub_energy_list = sub_term_energy_func(vqt_circuits, vqt_circuits_counts)
#     e_avg = tf.divide(
#         tf.reduce_sum(tf.cast(vqt_circuits_counts, tf.float32) * sub_energy_list),
#         num_vqt_samples,
#     )
#     return e_avg - qhbm_vqt.entropy_function()


# @tf.function
# def vqt_qhbm_loss_thetas_grad(qhbm_vqt, num_vqt_samples, sub_term_energy_func):
#     print("retracing: vqt_qhbm_loss_thetas_grad")
#     # Build components of the gradient
#     (
#         vqt_circuits,
#         vqt_bitstrings,
#         vqt_circuits_counts,
#     ) = qhbm_vqt.sample_state_circuits(
#         num_vqt_samples,
#     )
#     vqt_expanded_circuits_counts = tf.cast(
#         tf.expand_dims(vqt_circuits_counts, 1), tf.dtypes.float32
#     )
#     vqt_energies, vqt_energy_grads = tf.map_fn(
#         lambda x: qhbm_vqt.energy_and_energy_grad(x),
#         vqt_bitstrings,
#         fn_output_signature=(tf.float32, tf.float32),
#     )

#     # Compute the per vqt bitstring diff list
#     sub_energy_list = sub_term_energy_func(vqt_circuits, vqt_circuits_counts)
#     diff_list = sub_energy_list - vqt_energies
#     expanded_diff_list = tf.expand_dims(diff_list, 1)

#     # Compute the positive term in the gradient
#     diff_list_avg = tf.divide(
#         tf.reduce_sum(tf.cast(vqt_circuits_counts, tf.float32) * diff_list),
#         num_vqt_samples,
#     )
#     vqt_energy_grads_avg = tf.divide(
#         tf.reduce_sum(vqt_expanded_circuits_counts * vqt_energy_grads, 0),
#         num_vqt_samples,
#     )
#     positive = vqt_energy_grads_avg * diff_list_avg

#     # Compute the negative term in the gradient
#     prod_list = vqt_energy_grads * expanded_diff_list
#     negative = tf.divide(
#         tf.reduce_sum(vqt_expanded_circuits_counts * prod_list, 0), num_vqt_samples
#     )

#     return positive - negative


# @tf.function
# def phis_grad_sub_func(
#     i,
#     qhbm,
#     num_vqt_samples,
#     D,
#     eps,
#     sub_term_energy_func,
# ):
#     p_axis = tf.one_hot(i, D, dtype=tf.float32)
#     perturbation = p_axis * eps

#     # Forward
#     original_phis = qhbm.phis.read_value()
#     qhbm.phis.assign_add(perturbation)
#     (vqt_circuits_forward,_,vqt_circuits_counts_forward) = qhbm.sample_state_circuits(
#         num_vqt_samples,
#     )
#     sub_energy_list = sub_term_energy_func(
#         vqt_circuits_forward, vqt_circuits_counts_forward
#     )
#     forward = tf.divide(
#         tf.reduce_sum(
#             tf.cast(vqt_circuits_counts_forward, tf.float32) * sub_energy_list
#         ),
#         num_vqt_samples,
#     )

#     # Backward
#     qhbm.phis.assign_add(-2.0 * perturbation)
#     (
#         vqt_circuits_backward,
#         _,
#         vqt_circuits_counts_backward,
#     ) = qhbm.sample_state_circuits(
#         num_vqt_samples,
#     )
#     sub_energy_list = sub_term_energy_func(
#         vqt_circuits_backward, vqt_circuits_counts_backward
#     )
#     backward = tf.divide(
#         tf.reduce_sum(
#             tf.cast(vqt_circuits_counts_backward, tf.float32) * sub_energy_list
#         ),
#         num_vqt_samples,
#     )
#     qhbm.phis.assign(original_phis)

#     return tf.divide(forward - backward, (2.0 * eps))


# @tf.function
# def vqt_qhbm_loss_phis_grad(qhbm_vqt, num_vqt_samples, sub_term_energy_func, eps=0.1):
#     print("retracing: vqt_qhbm_loss_phis_grad")
#     D = tf.shape(qhbm_vqt.phis)[0]
#     return tf.map_fn(
#         lambda x: phis_grad_sub_func(
#             x,
#             qhbm_vqt,
#             num_vqt_samples,
#             D,
#             eps,
#             sub_term_energy_func,
#         ),
#         tf.range(D),
#         fn_output_signature=tf.float32,
#     )
