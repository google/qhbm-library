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
"""Impementation of the VQT loss function."""

from typing import Union

import tensorflow as tf

from qhbmlib import energy_model
from qhbmlib import hamiltonian_infer
from qhbmlib import hamiltonian_model


def vqt(qhbm_infer: hamiltonian_infer.QHBM,
        model: hamiltonian_model.Hamiltonian,
        hamiltonian: Union[tf.Tensor,
                           hamiltonian_model.Hamiltonian], beta: tf.Tensor):
  """Computes the VQT loss of a given QHBM and Hamiltonian.

  This function is differentiable within a `tf.GradientTape` scope.

  Args:
    qhbm_infer: Inference methods for the model.
    model: The modular Hamiltonian being trained to model the thermal state.
    hamiltonian: The Hamiltonian whose thermal state is to be learned.  If
      it is a `tf.Tensor`, it is of type `tf.string` with shape [1], result of
      calling `tfq.convert_to_tensor` on a list of `cirq.PauliSum`, `[op]`.
      Otherwise, a Hamiltonian.
    beta: A scalar `tf.Tensor` which is the inverse temperature at which the
      loss is calculated.

  Returns:
    The VQT loss.
  """

  # See equations B4 and B5 in appendix.  TODO(#119): confirm equation number.
  def f_vqt(bitstrings):
    if isinstance(hamiltonian, tf.Tensor):
      h_expectations = tf.squeeze(
          qhbm_infer.q_inference.expectation(model.circuit, bitstrings,
                                             hamiltonian), 1)
    elif isinstance(hamiltonian.energy, energy_model.PauliMixin):
      u_dagger_u = model.circuit + hamiltonian.circuit_dagger
      expectation_shards = qhbm_infer.q_inference.expectation(
          u_dagger_u, bitstrings, hamiltonian.operator_shards)
      h_expectations = hamiltonian.energy.operator_expectation(
          expectation_shards)
    else:
      raise NotImplementedError(
          "General `BitstringEnergy` hamiltonians not yet supported.")
    beta_h_expectations = beta * h_expectations
    energies = tf.stop_gradient(model.energy(bitstrings))
    return beta_h_expectations - energies

  qhbm_infer.e_inference.infer(model.energy)
  average_expectation = qhbm_infer.e_inference.expectation(f_vqt)
  current_partition = tf.stop_gradient(qhbm_infer.e_inference.log_partition())
  return average_expectation - current_partition
