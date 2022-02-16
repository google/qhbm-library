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
  return tf.einsum("k,ik,kj->ij", probabilities, unitary_matrix,
                   tf.linalg.adjoint(unitary_matrix))


def fidelity(model: hamiltonian_model.Hamiltonian, sigma: tf.Tensor):
  r"""Calculate the fidelity between a QHBM and a density matrix.

  Definition of the fidelity between two quantum states $\rho$ and $\sigma$ is
  $$
  F(\rho, \sigma) = \left(\text{tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}\right)^2.
  $$
  When the first argument is a QHBM, we can write
  $$
    F(\rho, \sigma) = \left(\text{tr}\sqrt{
        U_\phi\sqrt{K_\theta}U_\phi^\dagger
        \sigma U_\phi\sqrt{K_\theta}U_\phi^\dagger}\right)^2.
  $$
  By the definition of matrix functions, we can pull the unitaries outside
  the square root, and use cyclicity of trace to remove them.  Then we have
  $$
    F(\rho, \sigma) = \left(\text{tr}\sqrt{
        \sqrt{K_\theta}U_\phi^\dagger\sigma U_\phi\sqrt{K_\theta}}\right)^2.
  $$
  Let $\omega = \sqrt{K_\theta}U_\phi^\dagger\sigma U_\phi\sqrt{K_\theta}$,
  and let $WD W^\dagger$ be a unitary diagonalization of $\omega$.  Note that
  $U_\phi^\dagger\sigma U_\phi$ is Hermitian since it is a unitary
  conjugation of a Hermitian matrix.  Also, for Hermitian matrices A and B,
  we have
  $$
    (ABA)^\dagger = (BA)^\dagger A^\dagger
                  = A^\dagger B^\dagger A^\dagger
                  = ABA.
  $$
  Therefore $ABA$ is also Hermitian.  Thus $\omega$ is Hermitian, which
  allows the use of faster eigenvalue finding routines.  Then we have
  $$
    F(\rho, \sigma) = \left(\text{tr}\sqrt{D}\right)^2
                    = \left(\sum_i\sqrt{D_{ii}}\right)^2.
  $$

  Args:
    model: Modular Hamiltonian whose corresponding thermal state is to be
      compared to `sigma`, as the left density matrix in fidelity.
    sigma: 2-D `tf.Tensor` of dtype `complex64` representing the right
      density matrix in the fidelity calculation.

  Returns:
    A scalar `tf.Tensor` which is the fidelity between the density matrix
      represented by this QHBM and `sigma`.
  """
  k_theta = tf.cast(
      energy_infer_utils.probabilities(model.energy), tf.complex64)
  u_phi = circuit_infer_utils.unitary(model.circuit)
  u_phi_dagger = tf.linalg.adjoint(u_phi)
  sqrt_k_theta = tf.sqrt(k_theta)
  omega = tf.einsum("a,ab,bc,cd,d->ad", sqrt_k_theta, u_phi_dagger, sigma,
                    u_phi, sqrt_k_theta)
  d_omega = tf.linalg.eigvalsh(omega)
  return tf.math.reduce_sum(tf.math.sqrt(d_omega))**2
