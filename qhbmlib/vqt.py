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
"""Impementations of the VQT loss and its derivatives."""

import tensorflow as tf


def vqt(qhbm, num_samples, hamiltonian, beta):
  """Computes the VQT loss of a given QHBM against given thermal state params.

  This function is differentiable within a `tf.GradientTape` scope.

  Args:
    qhbm: A `qhbm.QHBM` which is the model whose loss is to be calculated.
    num_samples: A scalar `tf.Tensor` specifying the number of samples to draw
      from the EBM of `qhbm` when estimating the loss and its gradients.
    hamiltonian: 1D tensor of strings with one entry, the result of calling
      `tfq.convert_to_tensor` on a list containing one cirq.PauliSum, `[op]`.
      Here, `op` is the Hamiltonian against which the loss is calculated.
    beta: A scalar `tf.Tensor` which is the inverse temperature at which the
      loss is calculated.

  Returns:
    The VQT loss.
  """

  @tf.custom_gradient
  def loss(variables):
    bitstrings, counts = qhbm.ebm.sample(num_samples)
    probs = tf.cast(counts, tf.float32) / tf.cast(num_samples, tf.float32)
    expectation = tf.squeeze(
        qhbm.qnn.expectation(bitstrings, counts, hamiltonian), -1)
    if qhbm.is_analytic:
      entropy = qhbm.entropy()
    else:
      entropy = -tf.reduce_sum(probs * tf.math.log(probs))

    def grad(grad_y, variables=None):
      with tf.GradientTape() as qnn_tape:
        beta_expectations = beta * tf.squeeze(
            qhbm.qnn.expectation(bitstrings, counts, hamiltonian, reduce=False),
            -1)
        beta_expectation = tf.reduce_sum(probs * beta_expectations)
      grad_qnn = qnn_tape.gradient(beta_expectation,
                                   qhbm.qnn.trainable_variables)
      grad_qnn = [grad_y * grad for grad in grad_qnn]

      with tf.GradientTape() as ebm_tape:
        energies = qhbm.ebm.energy(bitstrings)
      energy_gradients = ebm_tape.jacobian(energies,
                                           qhbm.ebm.trainable_variables)
      probs_diffs = probs * (beta_expectations - energies)
      avg_diff = tf.reduce_sum(probs_diffs)
      grad_ebm = [
          grad_y *
          (avg_diff *
           tf.reduce_sum(tf.transpose(probs * tf.transpose(grad)), 0) -
           tf.reduce_sum(tf.transpose(probs_diffs * tf.transpose(grad)), 0))
          for grad in energy_gradients
      ]
      grad_vars = grad_ebm + grad_qnn
      return grad_vars

    return beta * expectation - entropy, grad

  return loss(qhbm.trainable_variables)
