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
import tensorflow_quantum as tfq

from qhbmlib import qhbm, ebm


def qmhl_loss(
    model: qhbm.QHBM, target_circuits: tf.Tensor, target_counts: tf.Tensor):
  """Calculate the QMHL loss of the model against the target.

    This loss is differentiable with respect to the trainable variables of the model.

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
  print(f"retracing: qmhl_loss on {model.name}")
  @tf.custom_gradient
  def call(thetas, phis, target_circuits, target_counts):
    # log_partition estimate
    if model.ebm.analytic:
      log_partition = model.log_partition_function()
    else:
      bitstrings, _ = model.ebm.sample(tf.reduce_sum(target_counts))
      energies = model.ebm.energy(bitstrings)
      log_partition = tf.math.reduce_logsumexp(-1 * energies)

    # pulled back expectation of energy operator
    samples, counts = model.qnn.pulled_back_sample(target_circuits, target_counts)
    energies = model.ebm.energy(samples)
    probs = tf.cast(counts, tf.float32) / tf.cast(tf.reduce_sum(counts), tf.float32)
    weighted_energies = energies * probs
    avg_energy = tf.reduce_sum(weighted_energies)
    
    forward_pass_vals = avg_energy + log_partition

    def gradient(grad):
      """Gradients are computed using estimators from the QHBM paper."""
      # Thetas derivative.
      qnn_bitstrings, qnn_counts = model.qnn.pulled_back_sample(
        target_circuits, target_counts)
      qnn_probs = tf.cast(qnn_counts, tf.float32) / tf.cast(tf.reduce_sum(qnn_counts), tf.float32)
      ebm_bitstrings, ebm_counts = model.ebm.sample(tf.reduce_sum(target_counts))
      ebm_probs = tf.cast(ebm_counts, tf.float32) / tf.cast(tf.reduce_sum(ebm_counts), tf.float32)
      with tf.GradientTape() as tape:
        qnn_energies = model.ebm.energy(qnn_bitstrings)
      qnn_thetas_grad_weighted = tape.jacobian(qnn_energies, thetas) * qnn_probs
      qnn_thetas_grad = tf.reduce_sum(qnn_thetas_grad_weights, 0)
      with tf.GradientTape() as tape:
        ebm_energies = model.ebm.energy(ebm_bitstrings)
      ebm_thetas_grad_weighted = tape.jacobian(ebm_energies, thetas) * ebm_probs
      ebm_thetas_grad = tf.reduce_sum(ebm_thetas_grad_weighted, 0)
      thetas_grad = qnn_thetas_grad - ebm_thetas_grad

      # Phis derivative.
      if model.ebm.has_operator:
        with tf.GradientTape() as tape:
          pulled_back_energy = model.qnn.pulled_back_expectation(
            target_circuits, target_counts, model.ebm.operator(model.raw_qubits))
        phis_grad = tape.gradient(pulled_back_energy, phis)
      else:
        raise NotImplementedError("Derivative when EBM has no operator is not yet supported.")
  
    return forward_pass_vals, gradient
  return call(model.thetas, model.phis, target_circuits, target_counts)
