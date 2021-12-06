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
from qhbmlib import qhbm


def qmhl(qhbm_model, density_operator, num_samples=1000):
  """Calculate the QMHL loss of the qhbm model against the target.

  This loss is differentiable with respect to the trainable variables of the
    model.

  Args:
    qhbm_model: Parameterized model density operator.
    target_circuits: 1-D tensor of strings which are serialized circuits. These
      circuits represent samples from the data density matrix.
    target_counts: 1-D tensor of integers which are the number of samples to
      draw from the data density matrix: `target_counts[i]` is the number of
        samples to draw from `target_circuits[i]`.

  Returns:
    loss: Quantum cross entropy between the target and model.
  """

  @tf.custom_gradient
  def function(trainable_variables):
    # pulled back expectation of energy operator
    if isinstance(density_operator, tf.string):
      circuits = density_operator
      counts = None
    elif isinstance(density_operator, tuple):
      circuits, counts = density_operator
    elif isinstance(density_operator, QHBM):
      circuits, counts = density_operator.circuits(num_samples)
    else:
      raise TypeError()

    num_samples = tf.reduce_sum(counts)
    if qhbm_model.ebm.has_operator:
      expectation_shards = qhbm_model.qnn.pulled_back_expectation(
          circuits, self.operator_shards, counts=counts)
      expectation = qhbm_model.ebm.operator_expectation(expectation_shards)
    else:
      qnn_bitstrings, qnn_counts = qhbm_model.qnn.pulled_back_sample(
          circuits, counts=counts)
      energies = qhbm_model.ebm.energy(qnn_bitstrings)
      qnn_probs = tf.cast(qnn_counts, tf.float32) / tf.cast(
          tf.reduce_sum(qnn_counts), tf.float32)
      expectation = tf.reduce_sum(qnn_probs * energies)

    # log_partition estimate
    if qhbm_model.ebm.is_analytic:
      log_partition_function = qhbm_model.log_partition_function()
    else:
      bitstrings, _ = qhbm_model.ebm.sample(num_samples)
      energies = qhbm_model.ebm.energy(bitstrings)
      log_partition_function = tf.math.reduce_logsumexp(-energies)

    def gradient(grad_y, variables=None):
      """Gradients are computed using estimators from the QHBM paper."""
      # Thetas derivative.
      if qhbm_model.ebm.has_operator:
        with tf.GradientTape() as tape:
          tape.watch(qhbm_model.ebm.trainable_variables)
          expectation = qhbm_model.ebm.operator_expectation(expectation_shards)
        qnn_energy_grad = tape.gradient(expectation,
                                        qhbm_model.ebm.trainable_variables)
      else:
        with tf.GradientTape() as tape:
          tape.watch(qhbm_model.ebm.trainable_variables)
          energies = qhbm_model.ebm.energy(qnn_bitstrings)
          expectation = tf.reduce_sum(qnn_probs * energies)
        qnn_energy_grad = tape.gradient(expectation,
                                        qhbm_model.ebm.trainable_variables)

      ebm_bitstrings, ebm_counts = qhbm_model.ebm.sample(num_samples)
      ebm_probs = tf.cast(ebm_counts, tf.float32) / tf.cast(
          tf.reduce_sum(ebm_counts), tf.float32)
      with tf.GradientTape() as tape:
        tape.watch(qhbm_model.ebm.trainable_variables)
        energies = qhbm_model.ebm.energy(ebm_bitstrings)
        expectation = tf.reduce_sum(ebm_probs * energies)
      ebm_energy_grad = tape.gradient(expectation,
                                      qhbm_model.ebm.trainable_variables)

      grad_ebm = [
          grad_y * (qnn_grad - ebm_grad)
          for qnn_grad, ebm_grad in zip(qnn_energy_grad, ebm_energy_grad)
      ]

      # Phis derivative.
      if qhbm_model.ebm.has_operator:
        with tf.GradientTape() as tape:
          tape.watch(qhbm_model.qnn.trainable_variables)
          expectation_shards = qhbm_model.qnn.pulled_back_expectation(
              circuits, qhbm_model.operator_shards, counts=counts)
          expectation = qhbm_model.ebm.operator_expectation(expectation_shards)
        grad_qnn = tape.gradient(expectation,
                                 qhbm_model.qnn.trainable_variables)
        grad_qnn = [grad_y * grad for grad in grad_qnn]
      else:
        raise NotImplementedError(
            "Derivative when EBM has no operator is not yet supported.")

      grad_qhbm = grad_ebm + grad_qnn
      if variables:
        return grad_qhbm, [tf.zeros_like(v) for v in variables]
      return grad_qhbm

    return expectation + log_partition_function, gradient

  return function(qhbm_model.trainable_variables)
