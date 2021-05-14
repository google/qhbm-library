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
"""Utility functions."""

import numpy as np

import tensorflow as tf
import tensorflow_quantum as tfq


# ============================================================================ #
# Density matrix utilities.
# ============================================================================ #


@tf.function
def pure_state_tensor_to_density_matrix(pure_states, counts):
    """Returns the uniform mixture of the given tensor of pure states.

    This function first treats every state as a column vector, then takes the
    outer product of each state with its adjoint.  Then the sum over all the outer
    products is taken, and the resulting matrix normalized.

    Args:
      pure_states: 2-D `tf.Tensor` of dtype `complex64` and shape
        [num_states, 2**num_qubits] containing a pure state decomposition of a
        density operator.
    """
    expanded_s = tf.expand_dims(pure_states, 1)
    col_s = tf.transpose(expanded_s, [0, 2, 1])
    adj_s = tf.linalg.adjoint(col_s)
    prod = tf.linalg.matmul(col_s, adj_s)
    thermal_state = tf.reduce_sum(
        prod * tf.cast(tf.expand_dims(tf.expand_dims(counts, 1), 2), tf.complex64), 0
    )
    return tf.math.divide(thermal_state, tf.cast(tf.reduce_sum(counts), tf.complex64))


@tf.function
def circuits_and_counts_to_density_matrix(circuits, counts):
    print("retracing: circuits_and_counts_to_density_matrix")
    pure_states = tfq.layers.State()(circuits).to_tensor()
    return pure_state_tensor_to_density_matrix(pure_states, counts)


@tf.function
def fidelity(rho, sigma):
    """Calculate the fidelity between the two given density matrices.

    Args:
      rho: 2-D `tf.Tensor` of dtype `complex64` representing the left density
        matrix in the fidelity calculation.
      sigma: 2-D `tf.Tensor` of dtype `complex64` representing the right density
        matrix in the fidelity calculation.

    Returns:
      A tf.Tensor float64 fidelity scalar between the two given density matrices.
    """
    print("retracing: fidelity")
    e_rho, v_rho = tf.linalg.eigh(tf.cast(rho, tf.complex128))
    rho_sqrt = tf.linalg.matmul(
        tf.linalg.matmul(v_rho, tf.linalg.diag(tf.math.sqrt(e_rho))),
        tf.linalg.adjoint(v_rho),
    )
    omega = tf.linalg.matmul(
        tf.linalg.matmul(rho_sqrt, tf.cast(sigma, tf.complex128)), rho_sqrt
    )
    e_omega = tf.linalg.eigvals(omega)
    return tf.math.abs(tf.math.reduce_sum(tf.math.sqrt(e_omega))) ** 2


@tf.function
def fast_fidelity(rho, sigma):
    """Calculate the 10x faster fidelity between the two given density matrices.

    Args:
      rho: 2-D `tf.Tensor` of dtype `complex64` representing the left density
        matrix in the fidelity calculation.
      sigma: 2-D `tf.Tensor` of dtype `complex64` representing the right density
        matrix in the fidelity calculation.

    Returns:
      A tf.Tensor float64 fidelity scalar between the two given density matrices.
    """
    print("retracing: fast_fidelity")
    e_rho, v_rho = tf.linalg.eigh(tf.cast(rho, tf.complex128))
    # Optimization
    # 1) tf.matmul(a, tf.linalg.diag(b))
    # -> tf.multiply(a, tf.tile(tf.expand_dims(b, axis=0), [tf.shape(a)[0], 1]))
    # minor portion, but 10x
    sqrt_e_rho = tf.sqrt(e_rho)
    v_rho_sqrt_e_rho = tf.multiply(
        v_rho, tf.tile(tf.expand_dims(sqrt_e_rho, axis=0), (tf.shape(v_rho)[0], 1))
    )
    rho_sqrt = tf.linalg.matmul(v_rho_sqrt_e_rho, tf.linalg.adjoint(v_rho))
    omega = tf.linalg.matmul(
        tf.linalg.matmul(rho_sqrt, tf.cast(sigma, tf.complex128)), rho_sqrt
    )
    # 2) Optimization eigvals -> eigh (10x), with numerical errors in 1e-4.
    e_omega, _ = tf.linalg.eigh(omega)
    return tf.math.abs(tf.math.reduce_sum(tf.math.sqrt(e_omega))) ** 2


@tf.function
def _np_fidelity_internal(e_rho, v_rho, sigma):
    """Internal np fidelity logic for preventing retracing graphs."""
    sqrt_e_rho = tf.cast(tf.sqrt(e_rho), dtype=tf.complex128)
    v_rho_sqrt_e_rho = tf.cast(
        tf.multiply(
            v_rho, tf.tile(tf.expand_dims(sqrt_e_rho, axis=0), (tf.shape(v_rho)[0], 1))
        ),
        tf.complex128,
    )
    rho_sqrt = tf.linalg.matmul(v_rho_sqrt_e_rho, tf.linalg.adjoint(v_rho))
    omega = tf.linalg.matmul(
        tf.linalg.matmul(rho_sqrt, tf.cast(sigma, tf.complex128)), rho_sqrt
    )
    return omega


def np_fidelity(rho, sigma):
    """Calculate the numpy-based fidelity between the two given density matrices.

    tf.linalg.eigh() currently shows memory usage growth,
    so it's slower & requires larger memory than np.linalg.eigh().
    To enable the experiments, we need this function.

    For example, 4x3 HEA VQT or 3x3 QAIA VQT can't run fidelity calculation on
    64 GiB due to memory exceeds.

    Args:
      rho: 2-D `tf.Tensor` of dtype `complex64` representing the left density
        matrix in the fidelity calculation.
      sigma: 2-D `tf.Tensor` of dtype `complex64` representing the right density
        matrix in the fidelity calculation.

    Returns:
      A tf.Tensor float64 fidelity scalar between the two given density matrices.
    """
    e_rho, v_rho = np.linalg.eigh(tf.cast(rho, dtype=tf.complex128))
    omega = _np_fidelity_internal(e_rho, v_rho, sigma)
    e_omega, _ = np.linalg.eigh(omega)
    return tf.math.abs(tf.math.reduce_sum(tf.math.sqrt(e_omega))) ** 2


@tf.function
def optimized_trace_matmul(rho, sigma):
    """Returns optimized version of tf.linalg.trace(tf.matmul(rho, sigma)).

    Assuming the both have the same shape.

    Args:
      rho: 2-D `tf.Tensor` of dtype `complex64` representing the left density
        matrix in the trace-matmul calculation.
      sigma: 2-D `tf.Tensor` of dtype `complex64` representing the right density
        matrix in the trace-matmul calculation.

    Returns:
      A tf.Tensor float64 trace value between the two given density matrices.
    """
    return tf.reduce_sum(
        tf.multiply(
            tf.cast(rho, tf.complex128), tf.transpose(tf.cast(sigma, tf.complex128))
        )
    )


@tf.function
def relative_entropy(rho, sigma):
    """Calculate the relative entropy between the two given density matrices.

    D(rho||sigma) = Tr[rho(log(rho) - log(sigma))]
                  = tf.linalg.trace(
                        tf.matmul(rho,
                                  tf.linalg.logm(rho) - tf.linalg.logm(sigma)))

    Args:
      rho: 2-D `tf.Tensor` of dtype `complex64` representing the left density
        matrix in the fidelity calculation.
      sigma: 2-D `tf.Tensor` of dtype `complex64` representing the right density
        matrix in the fidelity calculation.

    Returns:
      A tf.Tensor float64 fidelity scalar between the two given density matrices.
    """
    print("retracing: relative_entropy")
    log_rho = tf.linalg.logm(tf.cast(rho, tf.complex128))
    log_sigma = tf.linalg.logm(tf.cast(sigma, tf.complex128))
    return optimized_trace_matmul(rho, tf.subtract(log_rho, log_sigma))


@tf.function
def get_thermal_state(beta, h_num):
    """Computes the thermal state.

    Compute exp(-beta*h_num)/Tr[exp(-beta*h_num)].  Uses the log-sum-exp trick to
    avoid numerical instability.  When x is a vector with elements
    x_1, x_2, ..., x_N, define the log-sum-exp function as
    LSE(x) = ln(sum_i exp(x_i)).  Then, consider the entries of the derivative of
    LSE, [d/dx LSE(x)]_i = exp(x_i)/(sum_j exp(x_j)).  When h_u is the unitary
    diagonalizing h_num and h_diag is the corresponding matrix of eigenvalues,
    the thermal state can be rewritten as
    exp(-beta*h_num)/Tr[exp(-beta*h_num)]
        = exp(-beta*h_u@h_diag@h_u_dag)/Tr[exp(-beta*h_u@h_diag@h_u_dag)]
        = h_u @ exp(-beta* h_diag) @ h_u_dag /Tr[exp(-beta*h_u@h_diag@h_u_dag)]
          # Due to matrix exponential identity
          # https://en.wikipedia.org/wiki/Matrix_exponential#Diagonalizable_case
        = h_u @ exp(-beta* h_diag) @ h_u_dag /Tr[exp(-beta*h_diag)]
          # Due to cyclicity of trace
        = h_u @ exp(-beta* h_diag) @ h_u_dag / (sum_i <i|exp(-beta*h_diag)|i>)
          # Definition of trace
        = h_u @ exp(-beta* h_diag) @ h_u_dag / (sum_i exp(-beta*h_diag[i, i]))
          # Definition of diagonal matrix
        = h_u @ (exp(-beta* h_diag) / sum_i exp(-beta*h_diag[i, i])) @ h_u_dag
          # Scalar
        = h_u @ sum_i (exp(x_i) / sum_j exp(x_j))|i><i| @ h_u_dag
          # Let x_i = - beta * h_diag[i, i]
        = h_u @ sum_i [d/dx LSE(x)]_i|i><i| @ h_u_dag
          # Substitute LSE derivative

    Args:
      beta: Scalar `tf.Tensor` of dtype `float64` which is the inverse temperature
        at which to calculate the thermal state.
      h_num: Square, Hermitian `tf.Tensor` of dtype `complex128` which is the
        Hamiltonian whose thermal state is to be calculated.

    Returns:
      `tf.Tensor` of dtype `complex128` which is the density matrix of the thermal
          state of `h_num` at inverse temperature `beta`.
    """
    print("retracing: get_thermal_state")
    e_raw, v_raw = tf.linalg.eigh(h_num)
    e = tf.cast(e_raw, tf.float64)
    x = -1.0 * beta * e
    with tf.GradientTape() as g:
        g.watch(x)
        lse = tf.reduce_logsumexp(x)
    lse_grad = g.gradient(lse, x)
    tiled_lse_grad = tf.cast(
        tf.tile(tf.expand_dims(lse_grad, 0), [tf.shape(h_num)[0], 1]), tf.complex128
    )
    return tf.linalg.matmul(
        tf.math.multiply(v_raw, tiled_lse_grad), tf.linalg.adjoint(v_raw)
    )


@tf.function
def _np_get_thermal_state_internal(beta, e, h_shape):
    x = -1.0 * beta * e
    with tf.GradientTape() as g:
        g.watch(x)
        lse = tf.reduce_logsumexp(x)
    lse_grad = g.gradient(lse, x)
    tiled_lse_grad = tf.cast(
        tf.tile(tf.expand_dims(lse_grad, 0), [h_shape[0], 1]), tf.complex128
    )
    return tiled_lse_grad


def np_get_thermal_state(beta, h_num):
    """Computes the thermal state with numpy eigh().

    For details, please refer to get_thermal_state().
    Since tf.linalg.eigh() shows huge memory usage growth, we can't use it
    for larger grid size.

    Args:
      beta: Scalar `tf.Tensor` of dtype `float64` which is the inverse temperature
        at which to calculate the thermal state.
      h_num: Square, Hermitian `tf.Tensor` of dtype `complex128` which is the
        Hamiltonian whose thermal state is to be calculated.

    Returns:
      `tf.Tensor` of dtype `complex128` which is the density matrix of the thermal
          state of `h_num` at inverse temperature `beta`.
    """
    print("retracing: get_thermal_state")
    h_num = tf.cast(h_num, tf.complex128)
    e_raw, v_raw = np.linalg.eigh(h_num)
    tiled_lse_grad = _np_get_thermal_state_internal(
        tf.cast(beta, tf.float64), tf.cast(e_raw, tf.float64), tf.shape(h_num)
    )
    return tf.linalg.matmul(
        tf.math.multiply(v_raw, tiled_lse_grad), tf.linalg.adjoint(v_raw)
    )


# @tf.function
def log_partition_function(beta, h_num):
    """Computes the logarithm of the partition function.

    The log partition function is ln(Tr[expm(-beta * h_num)])

    Args:
      beta: Scalar `tf.Tensor` of dtype `float64` which is the inverse temperature
        at which to calculate the thermal state.
      h_num: Square, Hermitian `tf.Tensor` of dtype `complex128` which is the
        Hamiltonian appearing in the partition function.

    Returns:
      Scalar `tf.Tensor` of dtype `float64` which is the logarithm of the
        partition function.
    """
    print("retracing: log_partition_function")
    # Use eigh instead of eigvalsh to ease backpropagation, see docs on eigvalsh.
    h_eigs, _ = np.linalg.eigh(
        tf.cast(h_num, tf.complex128)
    )  # h_eigs are real since h_num is Hermitian
    return tf.reduce_logsumexp(-tf.cast(beta, tf.float64) * tf.cast(h_eigs, tf.float64))


@tf.function
def entropy(rho):
    """Computes the von Neumann entropy of the given density matrix.

    The von Neumann entropy is -Tr[rho ln[rho]].  Note that the entropy is then in
    units of nats here instead of bits since we use natural logarithm.  Simplify:
    -Tr[rho ln[rho]]
        = -Tr[U D U_dag ln[U D U_dag]]
        = -Tr[U D U_dag U ln[D] U_dag]
        = -Tr[D ln[D]]
        = -sum_i D_ii ln D_ii

    Args:
      rho: Square, Hermitian `tf.Tensor` of dtype `complex128` which is the
        density matrix.

    Returns:
      Scalar `tf.Tensor` of dtype `float64` which is the von Neumann entropy (in
        units of nats) of `rho`.
    """
    # Use eigh instead of eigvalsh to ease backpropagation, see docs on eigvalsh.
    rho_eigs, _ = tf.linalg.eigh(rho)
    # limit as x->0 of x ln x is 0, so replace nans from any 0 ln 0 with 0
    rho_prod = tf.math.multiply_no_nan(tf.math.log(rho_eigs), rho_eigs)
    return -tf.math.reduce_sum(rho_prod)


@tf.function
def density_matrix_to_image(dm):
    """Convert multi-qubit density matrix into an RGB image."""
    print("retracing: density_matrix_to_image")
    max_qubits = 9
    total_edge = 2 ** max_qubits
    dm_len = tf.shape(dm)[0]
    superpixel = tf.linalg.LinearOperatorFullMatrix(
        tf.ones((total_edge // dm_len, total_edge // dm_len), dtype=tf.float32)
    )
    my_zeros = tf.zeros([total_edge, total_edge, 1])
    dm_real = tf.math.abs(tf.math.real(dm))
    dm_imag = tf.math.abs(tf.math.imag(dm))
    max_dm_real = tf.math.reduce_max(dm_real)
    max_dm_imag = tf.math.reduce_max(dm_imag)
    max_dm = tf.math.maximum(max_dm_real, max_dm_imag)
    dm_real = tf.cast(dm_real / max_dm, tf.float32)
    dm_imag = tf.cast(dm_imag / max_dm, tf.float32)
    dm_real = tf.linalg.LinearOperatorKronecker(
        [tf.linalg.LinearOperatorFullMatrix(dm_real), superpixel]
    ).to_dense()
    dm_imag = tf.linalg.LinearOperatorKronecker(
        [tf.linalg.LinearOperatorFullMatrix(dm_imag), superpixel]
    ).to_dense()
    dm_real = tf.expand_dims(dm_real, 2)
    dm_imag = tf.expand_dims(dm_imag, 2)
    return tf.reshape(
        tf.concat([dm_real, my_zeros, dm_imag], 2), (1, total_edge, total_edge, 3)
    )


# ============================================================================ #
# Bitstring utilities.
# ============================================================================ #


@tf.function
def get_bit_sub_indices(qubits_vqt, qubits_j):
    """Assumes qubits_j is a subset of qubits_vqt."""
    print("retracing: get_bit_sub_indices")
    qubits_vqt_tiled = tf.tile(
        tf.expand_dims(qubits_vqt, 0), [tf.shape(qubits_j)[0], 1, 1]
    )
    qubits_j_expanded = tf.expand_dims(qubits_j, 1)
    expanded_where = tf.equal(qubits_vqt_tiled, qubits_j_expanded)
    reduced_where = tf.reduce_all(expanded_where, 2)
    all_wheres = tf.where(reduced_where)
    this_ones = tf.expand_dims(tf.ones(tf.shape(qubits_j)[0], dtype=tf.int32), 1)
    indices_j = tf.expand_dims(tf.range(tf.shape(qubits_j)[0]), 1)
    this_gather_inds = tf.concat([indices_j, this_ones], 1)
    return tf.gather_nd(all_wheres, this_gather_inds)
