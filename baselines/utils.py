# pylint: skip-file
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
"""Utility functions."""

import numpy as np

import tensorflow as tf
import tensorflow_quantum as tfq


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
          tf.cast(rho, tf.complex128),
          tf.transpose(tf.cast(sigma, tf.complex128))))


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
      A tf.Tensor float64 fidelity scalar between the two given density
      matrices.
    """
  log_rho = tf.linalg.logm(tf.cast(rho, tf.complex128))
  log_sigma = tf.linalg.logm(tf.cast(sigma, tf.complex128))
  return optimized_trace_matmul(rho, tf.subtract(log_rho, log_sigma))


def _get_thermal_state_internal(beta, e, h_shape):
  x = -1.0 * beta * e
  with tf.GradientTape() as g:
    g.watch(x)
    lse = tf.reduce_logsumexp(x)
  lse_grad = g.gradient(lse, x)
  tiled_lse_grad = tf.cast(
      tf.tile(tf.expand_dims(lse_grad, 0), [h_shape[0], 1]), tf.complex128)
  return tiled_lse_grad


def get_thermal_state(beta, h_num):
  """Computes the thermal state.
    Compute exp(-beta*h_num)/Tr[exp(-beta*h_num)].  Uses the log-sum-exp trick
    to
    avoid numerical instability.  When x is a vector with elements
    x_1, x_2, ..., x_N, define the log-sum-exp function as
    LSE(x) = ln(sum_i exp(x_i)).  Then, consider the entries of the derivative
    of
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
      beta: Scalar `tf.Tensor` of dtype `float64` which is the inverse
        temperature at which to calculate the thermal state.
      h_num: Square, Hermitian `tf.Tensor` of dtype `complex128` which is the
        Hamiltonian whose thermal state is to be calculated.
    Returns:
      `tf.Tensor` of dtype `complex128` which is the density matrix of the
      thermal
          state of `h_num` at inverse temperature `beta`.
    """
  h_num = tf.cast(h_num, tf.complex128)
  e_raw, v_raw = np.linalg.eigh(h_num)
  tiled_lse_grad = _get_thermal_state_internal(
      tf.cast(beta, tf.float64), tf.cast(e_raw, tf.float64), tf.shape(h_num))
  return tf.linalg.matmul(
      tf.math.multiply(v_raw, tiled_lse_grad), tf.linalg.adjoint(v_raw))


def log_partition_function(beta, h_num):
  """Computes the logarithm of the partition function.
    The log partition function is ln(Tr[expm(-beta * h_num)])
    Args:
      beta: Scalar `tf.Tensor` of dtype `float64` which is the inverse
        temperature at which to calculate the thermal state.
      h_num: Square, Hermitian `tf.Tensor` of dtype `complex128` which is the
        Hamiltonian appearing in the partition function.
    Returns:
      Scalar `tf.Tensor` of dtype `float64` which is the logarithm of the
        partition function.
    """
  # Use eigh instead of eigvalsh to ease backpropagation, see docs on eigvalsh.
  h_eigs, _ = np.linalg.eigh(tf.cast(
      h_num, tf.complex128))  # h_eigs are real since h_num is Hermitian
  return tf.reduce_logsumexp(-tf.cast(beta, tf.float64) *
                             tf.cast(h_eigs, tf.float64))


def density_matrix_to_image(dm):
  """Convert multi-qubit density matrix into an RGB image."""
  max_qubits = 9
  total_edge = 2**max_qubits
  dm_len = tf.shape(dm)[0]
  superpixel = tf.linalg.LinearOperatorFullMatrix(
      tf.ones((total_edge // dm_len, total_edge // dm_len), dtype=tf.float32))
  my_zeros = tf.zeros([total_edge, total_edge, 1])
  dm_real = tf.math.abs(tf.math.real(dm))
  dm_imag = tf.math.abs(tf.math.imag(dm))
  max_dm_real = tf.math.reduce_max(dm_real)
  max_dm_imag = tf.math.reduce_max(dm_imag)
  max_dm = tf.math.maximum(max_dm_real, max_dm_imag)
  dm_real = tf.cast(dm_real / max_dm, tf.float32)
  dm_imag = tf.cast(dm_imag / max_dm, tf.float32)
  dm_real = tf.linalg.LinearOperatorKronecker(
      [tf.linalg.LinearOperatorFullMatrix(dm_real), superpixel]).to_dense()
  dm_imag = tf.linalg.LinearOperatorKronecker(
      [tf.linalg.LinearOperatorFullMatrix(dm_imag), superpixel]).to_dense()
  dm_real = tf.expand_dims(dm_real, 2)
  dm_imag = tf.expand_dims(dm_imag, 2)
  return tf.reshape(
      tf.concat([dm_real, my_zeros, dm_imag], 2),
      (1, total_edge, total_edge, 3))
