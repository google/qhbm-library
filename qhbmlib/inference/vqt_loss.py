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

from qhbmlib.inference import qhbm
from qhbmlib.models import hamiltonian


def vqt(input_qhbm: qhbm.QHBM,
        target_hamiltonian: Union[tf.Tensor,
                                  hamiltonian.Hamiltonian], beta: tf.Tensor):
  """Computes the VQT loss of a given QHBM and Hamiltonian.

  This function is differentiable within a `tf.GradientTape` scope.

  Args:
    input_qhbm: Inference methods for the model.
    target_hamiltonian: The Hamiltonian whose thermal state is to be learned. If
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
    h_expectations = tf.squeeze(
        input_qhbm.q_inference.expectation(bitstrings, target_hamiltonian), 1)
    beta_h_expectations = beta * h_expectations
    energies = tf.stop_gradient(input_qhbm.e_inference.energy(bitstrings))
    return beta_h_expectations - energies

  average_expectation = input_qhbm.e_inference.expectation(f_vqt)
  current_partition = tf.stop_gradient(input_qhbm.e_inference.log_partition())
  return average_expectation - current_partition
