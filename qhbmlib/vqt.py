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


def vqt(model, num_samples, hamiltonian, beta):
  """Computes the VQT loss of a given QHBM against given thermal state params.

  This function is differentiable within a `tf.GradientTape` scope.

  Args:
    model: A `qhbm.QHBM` which is the model whose loss is to be calculated.
    num_samples: A scalar `tf.Tensor` specifying the number of samples to draw
      from the EBM of `model` when estimating the loss and its gradients.
    hamiltonian: 1D tensor of strings with one entry, the result of calling
      `tfq.convert_to_tensor` on a list containing one cirq.PauliSum, `[op]`.
      Here, `op` is the Hamiltonian against which the loss is calculated.
    beta: A scalar `tf.Tensor` which is the inverse temperature at which the
      loss is calculated.

  Returns:
    The VQT loss.
  """

  @tf.custom_gradient
  def loss(trainable_variables):
    # We use `model.qnn.trainable_variables` / `model.ebm.trainable_variables`
    # instead
    del trainable_variables
    bitstrings, counts = model.ebm.sample(num_samples)
    probs = tf.cast(counts, tf.float32) / tf.cast(num_samples, tf.float32)
    expectation = tf.squeeze(
        model.qnn.expectation(bitstrings, counts, hamiltonian), -1)
    if model.is_analytic:
      entropy = model.entropy()
    else:
      entropy = -tf.reduce_sum(probs * tf.math.log(probs))

    def grad(grad_y, variables=None):
      with tf.GradientTape() as tape:
        tape.watch(model.qnn.trainable_variables)
        beta_expectations = beta * tf.squeeze(
            model.qnn.expectation(
                bitstrings, counts, hamiltonian, reduce=False), -1)
        beta_expectation = tf.reduce_sum(probs * beta_expectations)
      grad_qnn = tape.gradient(beta_expectation, model.qnn.trainable_variables)
      grad_qnn = [grad_y * g for g in grad_qnn]

      with tf.GradientTape() as tape:
        tape.watch(model.ebm.trainable_variables)
        energies = model.ebm.energy(bitstrings)
      energy_jac = tape.jacobian(energies, model.ebm.trainable_variables)
      probs_diffs = probs * (beta_expectations - energies)
      avg_diff = tf.reduce_sum(probs_diffs)
      grad_ebm = [
          grad_y *
          (avg_diff * tf.reduce_sum(tf.transpose(probs * tf.transpose(g)), 0) -
           tf.reduce_sum(tf.transpose(probs_diffs * tf.transpose(g)), 0))
          for g in energy_jac
      ]
      grad_qhbm = grad_ebm + grad_qnn
      if variables is None:
        return grad_qhbm
      return grad_qhbm, [tf.zeros_like(g) for g in grad_qhbm]

    return beta * expectation - entropy, grad

  return loss(model.trainable_variables)
