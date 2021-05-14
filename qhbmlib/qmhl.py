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
"""Impementations of the QMHL loss and its derivatives."""

import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib import qhbm_base


# ============================================================================ #
# Sample based QMHL TODO(zaqqwerty).
# ============================================================================ #


# ============================================================================ #
# Exact QMHL.
# ============================================================================ #


@tf.function
def exact_qmhl_loss(
    qhbm_model: qhbm_base.ExactQHBM,
    target_circuits: tf.Tensor,
    target_counts: tf.Tensor,
):
    """Calculate the QMHL loss of the model against the target.

    Args:
      qhbm_model: Parameterized model density operator.
      target_circuits: 1-D tensor of strings which are serialized circuits.  These
        circuits represent samples from the data density matrix.
      target_counts: 1-D tensor of integers which are the number of samples to
        draw from the data density matrix: `target_counts[i]` is the number of
        samples to draw from `target_circuits[i]`.

    Returns:
      loss: Quantum cross entropy between the target and model.
    """
    print("retracing: qmhl_loss")
    expected = qhbm_model.pulled_back_energy_expectation(target_circuits, target_counts)
    log_partition = qhbm_model.log_partition_function()
    return expected + log_partition


@tf.function
def exact_qmhl_loss_thetas_grad(
    qhbm_model: qhbm_base.ExactQHBM,
    num_model_samples: tf.Tensor,
    target_circuits: tf.Tensor,
    target_counts: tf.Tensor,
):
    """Calculate thetas gradient of the QMHL loss of the model against the target.

    Args:
      qhbm_model: `QHBM` which is the parameterized model density operator.
      num_model_samples: Scalar integer tensor which is the number of bitstrings
        sampled from the classical distribution of `qhbm_model` to average over
        when estimating the model density operator.
      target_circuits: 1-D tensor of strings which are serialized circuits.  These
        circuits represent samples from the data density matrix.
      target_counts: 1-D tensor of integers which are the number of samples to
        draw from the data density matrix: `target_counts[i]` is the number of
        samples to draw from `target_circuits[i]`.

    Returns:
      Stochastic estimate of the gradient of the QMHL loss with respect to the
        classical model parameters.
    """
    print("retracing: exact_qmhl_loss_thetas_grad")

    # Get energy gradients for the classical bitstrings
    unique_samples_c, counts_c = qhbm_model.sample_bitstrings(num_model_samples)
    expanded_counts_c = tf.cast(
        tf.tile(tf.expand_dims(counts_c, 1), [1, tf.shape(qhbm_model.thetas)[0]]),
        tf.dtypes.float32,
    )
    _, e_grad_list_c = tf.map_fn(
        qhbm_model.energy_and_energy_grad,
        unique_samples_c,
        fn_output_signature=(tf.float32, tf.float32),
    )

    # Get energy gradients for the pulled-back data bitstrings
    ragged_samples_pb = qhbm_model.sample_pulled_back_bitstrings(
        target_circuits, target_counts
    )
    # safe when all circuits have the same number of qubits
    all_samples_pb = ragged_samples_pb.values.to_tensor()
    unique_samples_pb, _, counts_pb = qhbm_base.unique_with_counts(all_samples_pb)
    expanded_counts_pb = tf.cast(
        tf.tile(tf.expand_dims(counts_pb, 1), [1, tf.shape(qhbm_model.thetas)[0]]),
        tf.dtypes.float32,
    )
    _, e_grad_list_pb = tf.map_fn(
        qhbm_model.energy_and_energy_grad,
        unique_samples_pb,
        fn_output_signature=(tf.float32, tf.float32),
    )

    # Build theta gradients after reweighting
    e_grad_c_avg = tf.divide(
        tf.reduce_sum(expanded_counts_c * e_grad_list_c, 0),
        tf.cast(tf.reduce_sum(counts_c), tf.float32),
    )
    e_grad_pb_avg = tf.divide(
        tf.reduce_sum(expanded_counts_pb * e_grad_list_pb, 0),
        tf.cast(tf.reduce_sum(counts_pb), tf.float32),
    )
    return tf.math.subtract(e_grad_pb_avg, e_grad_c_avg)


@tf.function
def exact_qmhl_loss_phis_grad(
    qhbm: qhbm_base.ExactQHBM,
    op_tensor: tf.Tensor,
    target_circuits: tf.Tensor,
    target_count: tf.Tensor,
):
    """Calculate phis gradient of the QMHL loss of the model against the target.

    Args:
      qhbm: Parameterized model density operator.
      op_tensor: Result of calling `tfq.convert_to_tensor` on a list of Cirq
        PauliSums of the form `[[op0, op1, ..., opN-1]]`.  Each op should be
        diagonal in the computational basis.  The assumption made is that
        `qhbm.thetas[i]` is the weight of operator `op_tensor[0][i]` in the
        latent modular Hamiltonian.
      target_circuits: 1-D tensor of strings which are serialized circuits.  These
        circuits represent samples from the data density matrix.
      target_counts: 1-D tensor of integers which are the number of samples to
        draw from the data density matrix: `target_counts[i]` is the number of
        samples to draw from `target_circuits[i]`.

    Returns:
      Stochastic estimate of the gradient of the QMHL loss with respect to the
        unitary model parameters.
    """
    print("retracing: exact_qmhl_loss_phis_grad")
    num_circuits = tf.shape(target_circuits)[0]
    circuits_pb = tfq.append_circuit(
        target_circuits, tf.tile(qhbm.u_dagger, [num_circuits])
    )
    new_dup_phis = tf.identity(qhbm.phis)
    tiled_thetas = tf.tile(tf.expand_dims(qhbm.thetas, 0), [num_circuits, 1])
    with tf.GradientTape() as tape:
        tape.watch(new_dup_phis)
        sub_energy_list = tfq.layers.Expectation()(
            circuits_pb,
            symbol_names=qhbm.phis_symbols,
            symbol_values=tf.tile(tf.expand_dims(new_dup_phis, 0), [num_circuits, 1]),
            operators=tf.tile(op_tensor, [num_circuits, 1]),
        )
        # Weight each operator by the corresponding classical model parameter.
        scaled_sub_energy_list = tiled_thetas * sub_energy_list
        # Get the total latent modular Hamiltonian energy for each input circuit.
        pre_e_avg = tf.reduce_sum(scaled_sub_energy_list, 1)
        e_avg = tf.divide(
            tf.reduce_sum(tf.cast(target_count, tf.float32) * pre_e_avg),
            tf.reduce_sum(tf.cast(target_count, tf.float32)),
        )
    return tape.gradient(e_avg, new_dup_phis)


# ============================================================================ #
# QMHL for non-BUDA QHBMs.
# ============================================================================ #


# TODO(#18)
# @tf.function
# def qmhl_loss_thetas_grad(qhbm_model, num_model_samples, target_density):
#     """Calculate thetas gradient of the QMHL loss of the model against the target.

#     Args:
#       qhbm_model: `QHBM` which is the parameterized model density operator.
#       num_model_samples: number of bitstrings sampled from the classical
#         distribution of `qhbm_model` to average over when estimating the model
#         density operator.
#       target_density: `DensityOperator` which is the distribution whose logarithm
#         we are trying to learn.

#     Returns:
#       Stochastic estimate of the gradient of the QMHL loss with respect to the
#         classical model parameters.
#     """
#     print("retracing: qmhl_loss_thetas_grad")

#     # Get energy gradients for the classical bitstrings
#     unique_samples_c, counts_c = qhbm.sample_bitstrings(
#         qhbm_model.sampler_function, qhbm_model.thetas, num_model_samples
#     )
#     expanded_counts_c = tf.cast(
#         tf.tile(tf.expand_dims(counts_c, 1), [1, tf.shape(qhbm_model.thetas)[0]]),
#         tf.dtypes.float32,
#     )

#     def loop(x):
#         return qhbm.energy_and_energy_grad(
#             qhbm_model.energy_function, qhbm_model.thetas, x
#         )

#     _, e_grad_list_c = tf.map_fn(
#         loop, unique_samples_c, fn_output_signature=(tf.float32, tf.float32)
#     )

#     # Get energy gradients for the pulled-back data bitstrings
#     ragged_samples_pb = qhbm.sample_pulled_back_bitstrings(
#         qhbm_model.u_dagger,
#         qhbm_model.phis_symbols,
#         qhbm_model.phis,
#         target_density.circuits,
#         target_density.counts,
#     )
#     # safe when all circuits have the same number of qubits
#     all_samples_pb = ragged_samples_pb.values.to_tensor()
#     unique_samples_pb, _, counts_pb = qhbm_base.unique_with_counts(all_samples_pb)
#     expanded_counts_pb = tf.cast(
#         tf.tile(tf.expand_dims(counts_pb, 1), [1, tf.shape(qhbm_model.thetas)[0]]),
#         tf.dtypes.float32,
#     )
#     _, e_grad_list_pb = tf.map_fn(
#         loop, unique_samples_pb, fn_output_signature=(tf.float32, tf.float32)
#     )

#     # Build theta gradients after reweighting
#     e_grad_c_avg = tf.divide(
#         tf.reduce_sum(expanded_counts_c * e_grad_list_c, 0),
#         tf.cast(tf.reduce_sum(counts_c), tf.float32),
#     )
#     e_grad_pb_avg = tf.divide(
#         tf.reduce_sum(expanded_counts_pb * e_grad_list_pb, 0),
#         tf.cast(tf.reduce_sum(counts_pb), tf.float32),
#     )
#     return tf.math.subtract(e_grad_pb_avg, e_grad_c_avg)


# @tf.function
# def phis_grad_sub_func(
#     i,
#     qhbm_energy_function,
#     qhbm_thetas,
#     qhbm_u_dagger,
#     qhbm_phis_symbols,
#     qhbm_phis,
#     circuit_samples,
#     circuit_counts,
#     num_phis,
#     eps,
# ):
#     """phis_grad_sub_func."""
#     print("retracing: phis_grad_sub_func")
#     p_axis = tf.one_hot(i, num_phis, dtype=tf.float32)
#     perturbation = p_axis * eps
#     forward = qhbm.pulled_back_energy_expectation(
#         qhbm_energy_function,
#         qhbm_thetas,
#         qhbm_u_dagger,
#         qhbm_phis_symbols,
#         qhbm_phis + perturbation,
#         circuit_samples,
#         circuit_counts,
#     )
#     backward = qhbm.pulled_back_energy_expectation(
#         qhbm_energy_function,
#         qhbm_thetas,
#         qhbm_u_dagger,
#         qhbm_phis_symbols,
#         qhbm_phis - perturbation,
#         circuit_samples,
#         circuit_counts,
#     )
#     return tf.divide(forward - backward, (2.0 * eps))


# @tf.function
# def qmhl_loss_phis_grad(qhbm_model, target_density, eps=0.1):
#     """Calculate phis gradient of the QMHL loss of the model against the target.

#     Args:
#       qhbm_model: `QHBM` which is the parameterized model density operator.
#       target_density: `DensityOperator` which is the distribution whose logarithm
#         we are trying to learn.
#       eps: the size of the finite difference step.

#     Returns:
#       Stochastic estimate of the gradient of the QMHL loss with respect to the
#         unitary model parameters.
#     """
#     print("retracing: qmhl_loss_phis_grad")
#     num_phis = tf.shape(qhbm_model.phis)[0]

#     def loop(x):
#         phis_grad_sub_func(
#             x,
#             qhbm_model.energy_function,
#             qhbm_model.thetas,
#             qhbm_model.u_dagger,
#             qhbm_model.phis_symbols,
#             qhbm_model.phis,
#             target_density.circuits,
#             target_density.counts,
#             num_phis,
#             eps,
#         )

#     return tf.map_fn(loop, tf.range(num_phis), fn_output_signature=tf.float32)
