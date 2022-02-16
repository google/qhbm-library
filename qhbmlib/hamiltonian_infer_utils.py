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
"""Utilities for metrics on Hamiltonian."""

import tensorflow as tf

from qhbmlib import circuit_infer_utils
from qhbmlib import energy_infer_utils
from qhbmlib import hamiltonian_model


def trace_matmul(matrix_a, matrix_b):
  r"""Returns value of tf.linalg.trace(tf.matmul(matrix_a, matrix_b)).

  Naive matrix multiplication takes O(n^3) operators, while elementwise
  multiplication takes O(n^2).  To take advantage of this speedup, we transform
  the chained calculation as follows:

  \begin{align*}
    \text{tr}[AB] &= \sum_x \langle x |AB| x \rangle
    \\&= \sum_{x,i,j,k}\langle x| i\rangle A_{ik}B_{kj}\langle j |x\rangle
    \\&= \sum_{x,k} A_{xk}B_{kx}
    \\&= \sum_{x,k} A_{xk}B^T_{xk}
  \end{align*}

  Args:
    matrix_a: 2D tensor which is the left matrix in the calculation.
    matrix_b: 2D tensor which is the right matrix in the calculation.
      Must have the same dtype as `matix_a`.
  """
  return tf.reduce_sum(tf.multiply(matrix_a, tf.transpose(matrix_b)))


def density_matrix(model: hamiltonian_model.Hamiltonian):
  r"""Returns the thermal state corresponding to a modular Hamiltonian.

  Given a modular Hamiltonian $K_{\theta\phi} = U_\phi K_\theta U_\phi^\dagger$,
  the corresponding thermal state is

  \begin{align*}
    \rho &= (\text{tr}[e^{-K_{\theta\phi}}])^{-1} e^{-K_{\theta\phi}}
    \\&= Z_\theta^{-1} U_\phi e^{-K_\theta} U_\phi^\dagger
    \\&= U_\phi P_\theta U_\phi^\dagger
  \end{align*}
  where we defined the diagonal matrix $P_\theta$ as
  $$
    \langle x|P_\theta|y\rangle = \begin{cases}
                           p_\theta(x), & \text{if}\ x = y \\
                           0, & \text{otherwise}
                    \end{cases}
  $$
  Continuing, using the definition of matrix multiplication, we have
  \begin{align*}
    \rho &= U_\phi \sum_{i,k,j} |i\rangle\langle i|P_\theta|k\rangle
            \langle k| U_\phi^\dagger|j\rangle\langle j|
    \\&= U_\phi \sum_{k,j} p_\theta(k)|k\rangle
            \langle k| U_\phi^\dagger|j\rangle\langle j|
    \\&= \sum_{i,k,j} p_\theta(k)|i\rangle\langle i|U_\phi|k\rangle
            \langle k| U_\phi^\dagger|j\rangle\langle j|
  \end{align*}

  Args:
    model: Modular Hamiltonian whose corresponding thermal state is the density
      matrix to be calculated.
  """
  probabilities = tf.cast(
      energy_infer_utils.probabilities(model.energy), tf.complex64)
  unitary_matrix = circuit_infer_utils.unitary(model.circuit)
  return tf.einsum("k,ik,kj->ij", probabilities, unitary_matrix, tf.linalg.adjoint(unitary_matrix))


def fidelity(model: hamiltonian_model.Hamiltonian, sigma: tf.Tensor):
  """Calculate the fidelity between a QHBM and a density matrix.

  Args:
    model: Modular Hamiltonian whose corresponding thermal state is to be
      compared to `sigma`, as the left density matrix in fidelity.
    sigma: 2-D `tf.Tensor` of dtype `complex64` representing the right
      density matrix in the fidelity calculation.

  Returns:
    A scalar `tf.Tensor` which is the fidelity between the density matrix
      represented by this QHBM and `sigma`.
  """
  e_rho = tf.cast(energy_infer_utils.probabilities(model.energy), tf.complex64)
  v_rho = circuit_infer_utils.unitary(model.circuit)
  sqrt_e_rho = tf.sqrt(e_rho)
  v_rho_sqrt_e_rho = tf.multiply(
      v_rho, tf.tile(tf.expand_dims(sqrt_e_rho, 0), (tf.shape(v_rho)[0], 1)))
  rho_sqrt = tf.matmul(v_rho_sqrt_e_rho, tf.linalg.adjoint(v_rho))
  omega = tf.matmul(tf.matmul(rho_sqrt, tf.cast(sigma, tf.complex64)), rho_sqrt)
  # TODO(zaqqwerty): find convincing proof that omega is hermitian,
  # in order to go back to eigvalsh.
  e_omega = tf.linalg.eigvals(omega)
  return tf.cast(
      tf.math.abs(tf.math.reduce_sum(tf.math.sqrt(e_omega)))**2, tf.float32)
