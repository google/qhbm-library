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
  """Returns the normalized exponential of the hamililtonian.

  Args:
    model: Modular Hamiltonian whose corresponding thermal state is the density
      matrix to be calculated.
  """
  probabilities = tf.cast(
      energy_infer_utils.probabilities(model.energy), tf.complex64)
  unitary_matrix = circuit_infer_utils.unitary(model.circuit)
  unitary_probs = tf.multiply(
      unitary_matrix,
      tf.tile(
          tf.expand_dims(probabilities, 0), [tf.shape(unitary_matrix)[0], 1]))
  return tf.matmul(unitary_probs, tf.linalg.adjoint(unitary_matrix))


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
  omega = tf.matmul(
      tf.matmul(rho_sqrt, tf.cast(sigma, tf.complex64)), rho_sqrt)
  # TODO(zaqqwerty): find convincing proof that omega is hermitian,
  # in order to go back to eigvalsh.
  e_omega = tf.linalg.eigvals(omega)
  return tf.cast(
      tf.math.abs(tf.math.reduce_sum(tf.math.sqrt(e_omega)))**2, tf.float32)
