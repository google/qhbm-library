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
<<<<<<< HEAD
from qhbmlib import util

def qmhl(qhbm, circuits, counts):
  """Calculate the QMHL loss of the model against the target.
    This loss is differentiable with respect to the trainable variables of the
    model.
  Args:
    model: Parameterized model density operator.
    target_circuits: 1-D tensor of strings which are serialized circuits.
      These circuits represent samples from the data density matrix.
    target_counts: 1-D tensor of integers which are the number of samples to
      draw from the data density matrix: `target_counts[i]` is the number of
        samples to draw from `target_circuits[i]`.
  Returns:
    loss: Quantum cross entropy between the target and model.
  """
  print(f"retracing: qmhl_loss on {qhbm.name}")

  @tf.custom_gradient
  def loss(variables):
    # log_partition estimate
    if qhbm.ebm.is_analytic:
      log_partition = qhbm.log_partition_function()
    else:
      bitstrings, _ = qhbm.ebm.sample(tf.reduce_sum(counts))
      energies = qhbm.ebm.energy(bitstrings)
      log_partition = tf.math.reduce_logsumexp(-1 * energies)

    # pulled back expectation of energy operator
    qnn_bitstrings, qnn_counts = qhbm.qnn.pulled_back_sample(circuits, counts)

    # ragged_samples_pb = qhbm.qnn.pulled_back_sample(target_circuits,
    #                                                  target_counts)
    # all_samples_pb = ragged_samples_pb.values.to_tensor()
    # samples_pb, counts_pb = ebm.unique_bitstrings_with_counts(all_samples_pb)
    qnn_probs = tf.cast(qnn_counts, tf.float32) / tf.cast(
    tf.reduce_sum(qnn_counts), tf.float32)
    energies = qhbm.ebm.energy(qnn_bitstrings)
    avg_energy = tf.reduce_sum(qnn_probs * energies)

    def grad(grad_y, variables=None):
      """Gradients are computed using estimators from the QHBM paper."""

      # Thetas derivative.
      # qnn_bitstrings, qnn_counts = util.unique_bitstrings_with_counts(
      #     all_samples_pb)
      # qnn_probs = tf.cast(qnn_counts, tf.float32) / tf.cast(
      #     tf.reduce_sum(qnn_counts), tf.float32)
      ebm_bitstrings, ebm_counts = qhbm.ebm.sample(
          tf.reduce_sum(counts))
      ebm_probs = tf.cast(ebm_counts, tf.float32) / tf.cast(
          tf.reduce_sum(ebm_counts), tf.float32)

      with tf.GradientTape() as tape:
        qnn_energies = qhbm.ebm.energy(qnn_bitstrings)

      # jacobian is a list over thetas, with ith entry a tensor of shape
      # [tf.shape(qnn_energies)[0], tf.shape(thetas[i])[0]]
      qnn_grads = tape.jacobian(qnn_energies, qhbm.ebm.trainable_variables)

      with tf.GradientTape() as tape:
        ebm_energies = qhbm.ebm.energy(ebm_bitstrings)

      ebm_grads = tape.jacobian(ebm_energies, qhbm.ebm.trainable_variables)

      # contract over bitstring weights
      grad_ebm = [ grad_y * (
          tf.reduce_sum(tf.transpose(qnn_probs * tf.transpose(qnn_grad)), 0) -
          tf.reduce_sum(tf.transpose(ebm_probs * tf.transpose(ebm_grad)), 0))
          for qnn_grad, ebm_grad in zip(qnn_grads, ebm_grads)
      ]

      # Phis derivative.
      if qhbm.ebm.has_operator:
        with tf.GradientTape() as tape:
          energies = qhbm.qnn.pulled_back_expectation(
              circuits, counts, qhbm.operator_shards)
          avg_energy = qhbm.ebm.operator_expectation(
              energies)
        grad_qnn = tape.gradient(avg_energy,
                                  qhbm.qnn.trainable_variables)
        grad_qnn = [grad_y * grad for grad in grad_qnn]
      else:
        raise NotImplementedError(
            "Derivative when EBM has no operator is not yet supported.")
      
      grad_vars = grad_ebm + grad_qnn
      if variables is None:
        return grad_vars
      return grad_vars, grad_vars

    return avg_energy + log_partition, grad

  return loss(qhbm.trainable_variables)
=======

from qhbmlib import ebm
from qhbmlib import qhbm
from qhbmlib import util


def qmhl_loss(model: qhbm.QHBM, target_circuits: tf.Tensor,
              target_counts: tf.Tensor):
  """Calculate the QMHL loss of the model against the target.

    This loss is differentiable with respect to the trainable variables of the model.

    Args:
      model: Parameterized model density operator.
      target_circuits: 1-D tensor of strings which are serialized circuits.
        These circuits represent samples from the data density matrix.
      target_counts: 1-D tensor of integers which are the number of samples to
        draw from the data density matrix: `target_counts[i]` is the number of
          samples to draw from `target_circuits[i]`.

    Returns:
      loss: Quantum cross entropy between the target and model.
    """

  @tf.custom_gradient
  def call(thetas, phis):
    # log_partition estimate
    if model.ebm.is_analytic:
      log_partition = model.log_partition_function()
    else:
      bitstrings, _ = model.ebm.sample(tf.reduce_sum(target_counts))
      energies = model.ebm.energy(bitstrings)
      log_partition = tf.math.reduce_logsumexp(-1 * energies)

    # pulled back expectation of energy operator
    ragged_samples_pb = model.qnn.pulled_back_sample(target_circuits,
                                                     target_counts)
    all_samples_pb = ragged_samples_pb.values.to_tensor()
    samples_pb, counts_pb = util.unique_bitstrings_with_counts(all_samples_pb)
    energies = model.ebm.energy(samples_pb)
    probs_pb = tf.cast(counts_pb, tf.float32) / tf.cast(
        tf.reduce_sum(counts_pb), tf.float32)
    weighted_energies = energies * probs_pb
    avg_energy = tf.reduce_sum(weighted_energies)

    forward_pass_vals = avg_energy + log_partition

    def gradient(grad, variables=None):
      """Gradients are computed using estimators from the QHBM paper."""
      # Thetas derivative.
      qnn_probs = tf.cast(counts_pb, tf.float32) / tf.cast(
          tf.reduce_sum(counts_pb), tf.float32)
      ebm_bitstrings, ebm_counts = model.ebm.sample(
          tf.reduce_sum(target_counts))
      ebm_probs = tf.cast(ebm_counts, tf.float32) / tf.cast(
          tf.reduce_sum(ebm_counts), tf.float32)
      with tf.GradientTape() as tape:
        qnn_energies = model.ebm.energy(samples_pb)
      # jacobian is a list over thetas, with ith entry a tensor of shape
      # [tf.shape(qnn_energies)[0], tf.shape(thetas[i])[0]]
      qnn_jac = tf.ragged.stack(tape.jacobian(qnn_energies, model.thetas))
      # contract over bitstring weights
      qnn_thetas_grad = tf.einsum("ijk,j->ik", qnn_jac, probs_pb)
      with tf.GradientTape() as tape:
        ebm_energies = model.ebm.energy(ebm_bitstrings)
      ebm_jac = tf.ragged.stack(tape.jacobian(ebm_energies, model.thetas))
      ebm_thetas_grad = tf.einsum("ijk,j->ik", ebm_jac, ebm_probs)
      thetas_grad = qnn_thetas_grad - ebm_thetas_grad

      # Phis derivative.
      #      if model.ebm.has_operator:
      model_operators = model.operator_shards
      with tf.GradientTape() as tape:
        pulled_back_energy_shards = model.qnn.pulled_back_expectation(
            target_circuits, target_counts, model_operators)
        pulled_back_energy = model.ebm.operator_expectation(
            pulled_back_energy_shards)
      phis_grad = tape.gradient(pulled_back_energy,
                                model.qnn.trainable_variables)
      # else:
      #   raise NotImplementedError(
      #       "Derivative when EBM has no operator is not yet supported.")
      return grad * thetas_grad, grad * phis_grad

    return forward_pass_vals, gradient

  return call(model.thetas, model.phis)
>>>>>>> main
