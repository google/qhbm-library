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


def qmhl(qhbm_model, target_circuits, target_counts):
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

  @tf.custom_gradient
  def loss(trainable_variables):
    # We use `qhbm_model.ebm.trainable_variables` and
    # `qhbm_model.qnn.trainable_variables` instead
    del trainable_variables

    # log_partition estimate

    if qhbm_model.ebm.is_analytic:
      log_partition_function = qhbm_model.log_partition_function()
    else:
      bitstrings, _ = qhbm_model.ebm.sample(tf.reduce_sum(target_counts))
      energies = qhbm_model.ebm.energy(bitstrings)
      log_partition_function = tf.math.reduce_logsumexp(-1 * energies)

    # pulled back expectation of energy operator
    qnn_bitstrings, qnn_counts = qhbm_model.qnn.pulled_back_sample(
        target_circuits, target_counts)
    qnn_probs = tf.cast(qnn_counts, tf.float32) / tf.cast(
        tf.reduce_sum(qnn_counts), tf.float32)
    energies = qhbm_model.ebm.energy(qnn_bitstrings)
    avg_energy = tf.reduce_sum(qnn_probs * energies)

    def grad(grad_y, variables=None):
      """Gradients are computed using estimators from the QHBM paper."""
      # Thetas derivative.
      ebm_bitstrings, ebm_counts = qhbm_model.ebm.sample(
          tf.reduce_sum(target_counts))
      ebm_probs = tf.cast(ebm_counts, tf.float32) / tf.cast(
          tf.reduce_sum(ebm_counts), tf.float32)
      with tf.GradientTape() as tape:
        tape.watch(qhbm_model.ebm.trainable_variables)
        qnn_energies = qhbm_model.ebm.energy(qnn_bitstrings)
      # jacobian is a list over thetas, with ith entry a tensor of shape
      # [tf.shape(qnn_energies)[0], tf.shape(thetas[i])[0]]
      qnn_energy_jac = tape.jacobian(qnn_energies,
                                     qhbm_model.ebm.trainable_variables)

      with tf.GradientTape() as tape:
        tape.watch(qhbm_model.ebm.trainable_variables)
        ebm_energies = qhbm_model.ebm.energy(ebm_bitstrings)
      ebm_energy_jac = tape.jacobian(ebm_energies,
                                     qhbm_model.ebm.trainable_variables)

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
      if qhbm_model.ebm.has_operator:
        with tf.GradientTape() as tape:
          tape.watch(qhbm_model.qnn.trainable_variables)
          energy_shards = qhbm_model.qnn.pulled_back_expectation(
              target_circuits, target_counts, qhbm_model.operator_shards)
          energy = qhbm_model.ebm.operator_expectation(energy_shards)
        grad_qnn = tape.gradient(energy, qhbm_model.qnn.trainable_variables)
        grad_qnn = [grad_y * g for g in grad_qnn]
      else:
        raise NotImplementedError(
            "Derivative when EBM has no operator is not yet supported.")
      grad_qhbm = grad_ebm + grad_qnn
      if variables is None:
        return grad_qhbm
      return grad_qhbm, [tf.zeros_like(g) for g in grad_qhbm]

    return avg_energy + log_partition_function, grad

  return loss(qhbm_model.trainable_variables)
