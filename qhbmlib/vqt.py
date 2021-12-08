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
from qhbmlib import qhbm


def vqt(qhbm_model, hamiltonian, beta=1.0, num_samples=1000):
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
  def function(trainable_variables):
    bitstrings, counts = qhbm_model.ebm.sample(num_samples)
    probs = tf.cast(counts, tf.float32) / tf.cast(num_samples, tf.float32)

    if isinstance(hamiltonian, qhbm.QHBM):
      circuits = qhbm_model.qnn.circuits(bitstrings, resolve=False)
      if hamiltonian.ebm.has_operator:
        expectation_shards = hamiltonian.qnn.pulled_back_expectation(
            circuits,
            counts,
            hamiltonian.operator_shards,
            symbol_names=qhbm_model.qnn.symbols,
            symbol_values=qhbm_model.qnn.values)
        expectation = hamiltonian.ebm.operator_expectation(expectation_shards)
      else:
        qnn_bitstrings, qnn_counts = hamiltonian.qnn.pulled_back_sample(
            circuits,
            counts,
            symbol_names=qhbm_model.qnn.symbols,
            symbol_values=qhbm_model.qnn.values)
        energies = hamiltonian.ebm.energy(qnn_bitstrings)
        qnn_probs = tf.cast(qnn_counts, tf.float32) / tf.cast(
            tf.reduce_sum(qnn_counts), tf.float32)
        expectation = tf.reduce_sum(qnn_probs * energies)
    else:
      expectation = tf.squeeze(
          qhbm_model.qnn.expectation(bitstrings, counts, hamiltonian), -1)

    if qhbm_model.is_analytic:
      entropy = qhbm_model.entropy()
    else:
      entropy = -tf.reduce_sum(probs * tf.math.log(probs))

    def gradient(grad_y, variables=None):
      with tf.GradientTape() as tape:
        tape.watch(qhbm_model.qnn.trainable_variables)
        if isinstance(hamiltonian, qhbm.QHBM):
          if hamiltonian.ebm.has_operator:
            expectation_shards = hamiltonian.qnn.pulled_back_expectation(
                circuits,
                counts,
                hamiltonian.operator_shards,
                symbol_names=qhbm_model.qnn.symbols,
                symbol_values=qhbm_model.qnn.values,
                reduce=False)
            beta_expectations = beta * hamiltonian.ebm.operator_expectation(
                expectation_shards)
          else:
            raise NotImplementedError()
        else:
          beta_expectations = beta * tf.squeeze(
              qhbm_model.qnn.expectation(
                  bitstrings, counts, hamiltonian, reduce=False), -1)
        beta_expectation = tf.reduce_sum(probs * beta_expectations)
      grad_qnn = tape.gradient(beta_expectation,
                               qhbm_model.qnn.trainable_variables)
      grad_qnn = [grad_y * grad for grad in grad_qnn]

      with tf.GradientTape() as tape:
        tape.watch(qhbm_model.ebm.trainable_variables)
        energies = qhbm_model.ebm.energy(bitstrings)
      energy_jac = tape.jacobian(energies, qhbm_model.ebm.trainable_variables)
      probs_diffs = probs * (beta_expectations - energies)
      grad_ebm = [
          grad_y *
          (tf.reduce_sum(probs_diffs) *
           tf.reduce_sum(tf.transpose(probs * tf.transpose(jac)), 0) -
           tf.reduce_sum(tf.transpose(probs_diffs * tf.transpose(jac)), 0))
          for jac in energy_jac
      ]

      grad_qhbm = grad_ebm + grad_qnn
      if variables:
        return grad_qhbm, [tf.zeros_like(var) for var in variables]
      return grad_qhbm

    return beta * expectation - entropy, gradient

  return function(qhbm_model.trainable_variables)
