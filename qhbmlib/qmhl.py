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

# from qhbmlib import util
import tensorflow as tf

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

from qhbmlib import ebm
from qhbmlib import qhbm
from qhbmlib import util


def qmhl_loss(qhbm, circuits, counts):
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
      ebm_bitstrings, ebm_counts = qhbm.ebm.sample(tf.reduce_sum(counts))
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
      grad_ebm = [
          grad_y *
          (tf.reduce_sum(tf.transpose(qnn_probs * tf.transpose(qnn_grad)), 0) -
           tf.reduce_sum(tf.transpose(ebm_probs * tf.transpose(ebm_grad)), 0))
          for qnn_grad, ebm_grad in zip(qnn_grads, ebm_grads)
      ]

      # Phis derivative.
      if qhbm.ebm.has_operator:
        with tf.GradientTape() as tape:
          energies = qhbm.qnn.pulled_back_expectation(circuits, counts,
                                                      qhbm.operator_shards)
          avg_energy = qhbm.ebm.operator_expectation(energies)
        grad_qnn = tape.gradient(avg_energy, qhbm.qnn.trainable_variables)
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


def qmhl(qhbm, circuits, counts):
  """Calculates the QMHL loss of the model against the target.

  This loss is differentiable with respect to the trainable variables of the
  model.

  Args:
    model: Parameterized model density operator.
    target_circuits: 1-D tensor of strings which are serialized circuits. These
      circuits represent samples from the data density matrix.
    target_counts: 1-D tensor of integers which are the number of samples to
      draw from the data density matrix: `target_counts[i]` is the number of
        samples to draw from `target_circuits[i]`.

  Returns:
    loss: Quantum cross entropy between the target and model.
  """
  # print(f"retracing: qmhl_loss on {qhbm.name}")

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
    qnn_probs = tf.cast(qnn_counts, tf.float32) / tf.cast(
        tf.reduce_sum(qnn_counts), tf.float32)
    energies = qhbm.ebm.energy(qnn_bitstrings)
    avg_energy = tf.reduce_sum(qnn_probs * energies)

    def grad(grad_y, variables=None):
      """Gradients are computed using estimators from the QHBM paper."""

      # Thetas derivative.
      ebm_bitstrings, ebm_counts = qhbm.ebm.sample(tf.reduce_sum(counts))
      ebm_probs = tf.cast(ebm_counts, tf.float32) / tf.cast(
          tf.reduce_sum(ebm_counts), tf.float32)

      with tf.GradientTape() as tape:
        qnn_energies = qhbm.ebm.energy(qnn_bitstrings)

      # jacobian is a list over thetas, with ith entry a tensor of shape
      qnn_grads = tape.jacobian(qnn_energies, qhbm.ebm.trainable_variables)

      with tf.GradientTape() as tape:
        ebm_energies = qhbm.ebm.energy(ebm_bitstrings)

      ebm_grads = tape.jacobian(ebm_energies, qhbm.ebm.trainable_variables)

      # contract over bitstring weights
      grad_ebm = [
          grad_y *
          (tf.reduce_sum(tf.transpose(qnn_probs * tf.transpose(qnn_grad)), 0) -
           tf.reduce_sum(tf.transpose(ebm_probs * tf.transpose(ebm_grad)), 0))
          for qnn_grad, ebm_grad in zip(qnn_grads, ebm_grads)
      ]

      # Phis derivative.
      if qhbm.ebm.has_operator:
        with tf.GradientTape() as tape:
          energies = qhbm.qnn.pulled_back_expectation(circuits, counts,
                                                      qhbm.operator_shards)
          avg_energy = qhbm.ebm.operator_expectation(energies)
        grad_qnn = tape.gradient(avg_energy, qhbm.qnn.trainable_variables)
        grad_qnn = [grad_y * grad for grad in grad_qnn]
      else:
        raise NotImplementedError(
            "Derivative when EBM has no operator is not yet supported.")

      grad_vars = grad_ebm + grad_qnn

      # print("variables: ", variables)
      # if variables is None:
      #   return grad_vars, None, None
      return grad_vars

    return avg_energy + log_partition, grad

  return loss(qhbm.trainable_variables)


def qmhl_hacked(qhbm, circuits, counts):
  """Calculates the QMHL loss of the model against the target.

  This loss is differentiable with respect to the trainable variables of the
  model.

  Args:
    model: Parameterized model density operator.
    target_circuits: 1-D tensor of strings which are serialized circuits. These
      circuits represent samples from the data density matrix.
    target_counts: 1-D tensor of integers which are the number of samples to
      draw from the data density matrix: `target_counts[i]` is the number of
        samples to draw from `target_circuits[i]`.

  Returns:
    loss: Quantum cross entropy between the target and model.
  """
  # print(f"retracing: qmhl_loss on {qhbm.name}")

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
    qnn_probs = tf.cast(qnn_counts, tf.float32) / tf.cast(
        tf.reduce_sum(qnn_counts), tf.float32)
    energies = qhbm.ebm.energy(qnn_bitstrings)
    avg_energy = tf.reduce_sum(qnn_probs * energies)

    def grad(grad_y, variables=None):
      """Gradients are computed using estimators from the QHBM paper."""

      # Thetas derivative.
      ebm_bitstrings, ebm_counts = qhbm.ebm.sample(tf.reduce_sum(counts))
      ebm_probs = tf.cast(ebm_counts, tf.float32) / tf.cast(
          tf.reduce_sum(ebm_counts), tf.float32)

      with tf.GradientTape() as tape:
        qnn_energies = qhbm.ebm.energy(qnn_bitstrings)

      # jacobian is a list over thetas, with ith entry a tensor of shape
      qnn_grads = tape.jacobian(qnn_energies, qhbm.ebm.trainable_variables)

      with tf.GradientTape() as tape:
        ebm_energies = qhbm.ebm.energy(ebm_bitstrings)

      ebm_grads = tape.jacobian(ebm_energies, qhbm.ebm.trainable_variables)

      # contract over bitstring weights
      grad_ebm = [
          grad_y *
          (tf.reduce_sum(tf.transpose(qnn_probs * tf.transpose(qnn_grad)), 0) -
           tf.reduce_sum(tf.transpose(ebm_probs * tf.transpose(ebm_grad)), 0))
          for qnn_grad, ebm_grad in zip(qnn_grads, ebm_grads)
      ]

      # Phis derivative.
      if qhbm.ebm.has_operator:
        with tf.GradientTape() as tape:
          energies = qhbm.qnn.pulled_back_expectation(circuits, counts,
                                                      qhbm.operator_shards)
          avg_energy = qhbm.ebm.operator_expectation(energies)
        grad_qnn = tape.gradient(avg_energy, qhbm.qnn.trainable_variables)
        grad_qnn = [grad_y * grad for grad in grad_qnn]
      else:
        raise NotImplementedError(
            "Derivative when EBM has no operator is not yet supported.")

      grad_vars = grad_ebm + grad_qnn

      # print("variables: ", variables)
      if variables is None:
        return grad_vars, None, None
      return grad_vars, grad_vars

    return avg_energy + log_partition, grad

  return loss(qhbm.trainable_variables)


# def exact_qmhl_loss(qhbm_model, target_circuits: tf.Tensor,
#                     target_counts: tf.Tensor):
#   """Calculate the QMHL loss of the model against the target.

#   Args:
#     qhbm_model: Parameterized model density operator.
#     target_circuits: 1-D tensor of strings which are serialized circuits.  These
#       circuits represent samples from the data density matrix.
#     target_counts: 1-D tensor of integers which are the number of samples to
#       draw from the data density matrix: `target_counts[i]` is the number of
#         samples to draw from `target_circuits[i]`.

#   Returns:
#     loss: Quantum cross entropy between the target and model.
#   """
#   print("retracing: qmhl_loss")
#   expected = qhbm_model.pulled_back_energy_expectation(target_circuits,
#                                                        target_counts)
#   log_partition = qhbm_model.log_partition_function()
#   return expected + log_partition

# def exact_qmhl_loss_thetas_grad(qhbm_model,
#                                 num_model_samples: tf.Tensor,
#                                 target_circuits: tf.Tensor,
#                                 target_counts: tf.Tensor):
#   """Calculate thetas gradient of the QMHL loss of the model against the target.

#   Args:
#     qhbm_model: `QHBM` which is the parameterized model density operator.
#     num_model_samples: Scalar integer tensor which is the number of bitstrings
#       sampled from the classical distribution of `qhbm_model` to average over
#       when estimating the model density operator.
#     target_circuits: 1-D tensor of strings which are serialized circuits.  These
#       circuits represent samples from the data density matrix.
#     target_counts: 1-D tensor of integers which are the number of samples to
#       draw from the data density matrix: `target_counts[i]` is the number of
#         samples to draw from `target_circuits[i]`.

#   Returns:
#     Stochastic estimate of the gradient of the QMHL loss with respect to the
#       classical model parameters.
#   """
#   print("retracing: exact_qmhl_loss_thetas_grad")

#   # Get energy gradients for the classical bitstrings
#   unique_samples_c, counts_c = qhbm_model.sample_bitstrings(num_model_samples)
#   expanded_counts_c = tf.cast(
#       tf.tile(tf.expand_dims(counts_c, 1),
#               [1, tf.shape(qhbm_model.thetas)[0]]), tf.dtypes.float32)
#   _, e_grad_list_c = tf.map_fn(
#       qhbm_model.energy_and_energy_grad,
#       unique_samples_c,
#       fn_output_signature=(tf.float32, tf.float32))

#   # Get energy gradients for the pulled-back data bitstrings
#   ragged_samples_pb = qhbm_model.sample_pulled_back_bitstrings(
#       target_circuits, target_counts)
#   # safe when all circuits have the same number of qubits
#   all_samples_pb = ragged_samples_pb.values.to_tensor()
#   unique_samples_pb, _, counts_pb = util.unique_with_counts(all_samples_pb)
#   expanded_counts_pb = tf.cast(
#       tf.tile(
#           tf.expand_dims(counts_pb, 1), [1, tf.shape(qhbm_model.thetas)[0]]),
#       tf.dtypes.float32)
#   _, e_grad_list_pb = tf.map_fn(
#       qhbm_model.energy_and_energy_grad,
#       unique_samples_pb,
#       fn_output_signature=(tf.float32, tf.float32))

#   # Build theta gradients after reweighting
#   e_grad_c_avg = tf.divide(
#       tf.reduce_sum(expanded_counts_c * e_grad_list_c, 0),
#       tf.cast(tf.reduce_sum(counts_c), tf.float32))
#   e_grad_pb_avg = tf.divide(
#       tf.reduce_sum(expanded_counts_pb * e_grad_list_pb, 0),
#       tf.cast(tf.reduce_sum(counts_pb), tf.float32))
#   return tf.math.subtract(e_grad_pb_avg, e_grad_c_avg)

# def exact_qmhl_loss_phis_grad(qhbm, op_tensor: tf.Tensor,
#                               target_circuits: tf.Tensor,
#                               target_count: tf.Tensor):
#   """Calculate phis gradient of the QMHL loss of the model against the target.

#   Args:
#     qhbm: Parameterized model density operator.
#     op_tensor: Result of calling `tfq.convert_to_tensor` on a list of Cirq
#       PauliSums of the form `[[op0, op1, ..., opN-1]]`.  Each op should be
#       diagonal in the computational basis.  The assumption made is that
#       `qhbm.thetas[i]` is the weight of operator `op_tensor[0][i]` in the latent
#       modular Hamiltonian.
#     target_circuits: 1-D tensor of strings which are serialized circuits.  These
#       circuits represent samples from the data density matrix.
#     target_counts: 1-D tensor of integers which are the number of samples to
#       draw from the data density matrix: `target_counts[i]` is the number of
#         samples to draw from `target_circuits[i]`.

#   Returns:
#     Stochastic estimate of the gradient of the QMHL loss with respect to the
#       unitary model parameters.
#   """
#   print("retracing: exact_qmhl_loss_phis_grad")
#   num_circuits = tf.shape(target_circuits)[0]
#   circuits_pb = tfq.append_circuit(target_circuits,
#                                    tf.tile(qhbm.u_dagger, [num_circuits]))
#   new_dup_phis = tf.identity(qhbm.phis)
#   tiled_thetas = tf.tile(tf.expand_dims(qhbm.thetas, 0), [num_circuits, 1])
#   with tf.GradientTape() as tape:
#     tape.watch(new_dup_phis)
#     sub_energy_list = tfq.layers.Expectation()(
#         circuits_pb,
#         symbol_names=qhbm.phis_symbols,
#         symbol_values=tf.tile(
#             tf.expand_dims(new_dup_phis, 0), [num_circuits, 1]),
#         operators=tf.tile(op_tensor, [num_circuits, 1]))
#     # Weight each operator by the corresponding classical model parameter.
#     scaled_sub_energy_list = tiled_thetas * sub_energy_list
#     # Get the total latent modular Hamiltonian energy for each input circuit.
#     pre_e_avg = tf.reduce_sum(scaled_sub_energy_list, 1)
#     e_avg = tf.divide(
#         tf.reduce_sum(tf.cast(target_count, tf.float32) * pre_e_avg),
#         tf.reduce_sum(tf.cast(target_count, tf.float32)))
#   return tape.gradient(e_avg, new_dup_phis)
