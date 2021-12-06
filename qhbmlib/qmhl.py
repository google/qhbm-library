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
"""Impementations of the QMHL loss and its derivatives."""

import tensorflow as tf


def qmhl(model_qhbm, target_circuits=None, target_counts=None, target_qhbm=None, num_samples=None):
  """Calculate the QMHL loss of the qhbm model against the target.

  This loss is differentiable with respect to the trainable variables of the
    model.

  Args:
    qhbm_model: Parameterized model density operator.
    target_circuits: 1-D tensor of strings which are serialized circuits.
      These circuits represent samples from the data density matrix.
    target_counts: 1-D tensor of integers which are the number of samples to
      draw from the data density matrix: `target_counts[i]` is the number of
        samples to draw from `target_circuits[i]`.

  Returns:
    loss: Quantum cross entropy between the target and model.
  """

  if target_qhbm is not None and num_samples is not None:
    target_circuits, target_counts = target_qhbm.circuits(num_samples)

  @tf.custom_gradient
  def loss(trainable_variables):
    # log_partition estimate

    if model_qhbm.ebm.is_analytic:
      log_partition_function = model_qhbm.log_partition_function()
    else:
      bitstrings, _ = model_qhbm.ebm.sample(tf.reduce_sum(target_counts))
      energies = model_qhbm.ebm.energy(bitstrings)
      log_partition_function = tf.math.reduce_logsumexp(-1 * energies)

    # pulled back expectation of energy operator
    qnn_bitstrings, qnn_counts = model_qhbm.qnn.pulled_back_sample(
        target_circuits, target_counts)
    qnn_probs = tf.cast(qnn_counts, tf.float32) / tf.cast(
        tf.reduce_sum(qnn_counts), tf.float32)
    energies = model_qhbm.ebm.energy(qnn_bitstrings)
    avg_energy = tf.reduce_sum(qnn_probs * energies)

    def grad(grad_y, variables=None):
      """Gradients are computed using estimators from the QHBM paper."""
      # Thetas derivative.
      ebm_bitstrings, ebm_counts = model_qhbm.ebm.sample(
          tf.reduce_sum(target_counts))
      ebm_probs = tf.cast(ebm_counts, tf.float32) / tf.cast(
          tf.reduce_sum(ebm_counts), tf.float32)
      with tf.GradientTape() as tape:
        tape.watch(model_qhbm.ebm.trainable_variables)
        qnn_energies = model_qhbm.ebm.energy(qnn_bitstrings)
      # jacobian is a list over thetas, with ith entry a tensor of shape
      # [tf.shape(qnn_energies)[0], tf.shape(thetas[i])[0]]
      qnn_energy_jac = tape.jacobian(qnn_energies,
                                     model_qhbm.ebm.trainable_variables)

      with tf.GradientTape() as tape:
        tape.watch(model_qhbm.ebm.trainable_variables)
        ebm_energies = model_qhbm.ebm.energy(ebm_bitstrings)
      ebm_energy_jac = tape.jacobian(ebm_energies,
                                     model_qhbm.ebm.trainable_variables)

      # contract over bitstring weights
      grad_ebm = [
          grad_y *
          (tf.reduce_sum(
              tf.transpose(qnn_probs * tf.transpose(qnn_energy_grad)), 0) -
           tf.reduce_sum(
               tf.transpose(ebm_probs * tf.transpose(ebm_energy_grad)), 0))
          for qnn_energy_grad, ebm_energy_grad in zip(qnn_energy_jac,
                                                      ebm_energy_jac)
      ]

      # Phis derivative.
      if model_qhbm.ebm.has_operator:
        with tf.GradientTape() as tape:
          tape.watch(model_qhbm.qnn.trainable_variables)
          energy_shards = model_qhbm.qnn.pulled_back_expectation(
              target_circuits, target_counts, model_qhbm.operator_shards)
          energy = model_qhbm.ebm.operator_expectation(energy_shards)
        grad_qnn = tape.gradient(energy, model_qhbm.qnn.trainable_variables)
        grad_qnn = [grad_y * g for g in grad_qnn]
      else:
        raise NotImplementedError(
            "Derivative when EBM has no operator is not yet supported.")
      grad_qhbm = grad_ebm + grad_qnn
      if variables is None:
        return grad_qhbm
      return grad_qhbm, [tf.zeros_like(g) for g in grad_qhbm]

    return avg_energy + log_partition_function, grad

  return loss(model_qhbm.trainable_variables)
