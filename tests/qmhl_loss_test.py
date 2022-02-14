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
"""Tests for the QMHL loss and gradients."""

import cirq
import tensorflow as tf

from qhbmlib import qmhl_loss
from qhbmlib import quantum_data
from tests import test_util


class QMHLTest(tf.test.TestCase):
  """Tests for the QMHL loss and gradients."""

  def setUp(self):
    """Initializes test objects."""
    super().setUp()
    self.num_qubits_list = [1, 2, 3, 4]
    self.tf_random_seed = 7
    self.tfp_seed = tf.constant([3, 4], tf.int32)
    self.close_rtol = 1e-2
    self.not_zero_atol = 1e-1

  def test_self_qmhl(self):
    """Confirms known value of the QMHL loss of a model against itself."""
    num_layers = 5
    for num_qubits in self.num_qubits_list:
      qubits = cirq.GridQubit.rect(1, num_qubits)
      data_h, data_infer = test_util.get_random_hamiltonian_and_inference(qubits, num_layers,
                                         f"data_objects_{num_qubits}")
      model_h, model_infer = test_util.get_random_hamiltonian_and_inference(qubits, num_layers,
                                         f"hamiltonian_objects_{num_qubits}")
      # Set data equal to the model
      num_expectation_samples = int(1e6)
      data_h.set_weights(model_h.get_weights())
      data = quantum_data.QHBMData(data_infer, data_h, num_expectation_samples)

      # Trained loss is the entropy.
      expected_loss = model_infer.e_inference.entropy()
      # Since this is the optimum, derivatives should all be zero.
      expected_loss_derivative = [
        tf.zeros_like(v) for v in model_h.trainable_variables]

      with tf.GradientTape() as tape:
        actual_loss = qmhl_loss.qmhl(data, model_infer, model_h)
      actual_loss_derivative = tape.gradient(actual_loss, model_h.trainable_variables)

      self.assertAllClose(actual_loss, expected_loss)
      self.assertAllClose(actual_loss_derivative, expected_loss_derivative)

  # def test_loss_value_x_rot(self):
  #   """Confirms correct values for a single qubit X rotation QHBM.

  #   We use a data state which is a Y rotation of an initially diagonal density
  #   operator.  The QHBM is a Bernoulli latent state with X rotation QNN.

  #   See the colab notebook at the following link for derivations:
  #   https://colab.research.google.com/drive/14987JCMju_8AVvvVoojwe6hA7Nlw-Dhe?usp=sharing

  #   Since each qubit is independent, the loss is the sum over the individual
  #   qubit losses, and the gradients are the the per-qubit gradients.
  #   """
  #   seed = None
  #   ebm_const = 2.0
  #   q_const = 2 * math.pi
  #   num_data_samples = 1e6
  #   for qmhl_func in [
  #       qmhl.qmhl,
  #       tf.function(qmhl.qmhl, experimental_compile=False),
  #       #        tf.function(qmhl.qmhl, experimental_compile=True)
  #   ]:
  #     for num_qubits in [1, 2, 3, 4, 5]:
  #       # EBM
  #       ebm_init = tf.keras.initializers.RandomUniform(
  #           minval=-ebm_const, maxval=ebm_const, seed=seed)
  #       test_ebm = ebm.Bernoulli(num_qubits, ebm_init, True)

  #       # QNN
  #       qubits = cirq.GridQubit.rect(1, num_qubits)
  #       r_symbols = [sympy.Symbol(f"phi_{n}") for n in range(num_qubits)]
  #       r_circuit = cirq.Circuit(
  #           cirq.rx(r_s)(q) for r_s, q in zip(r_symbols, qubits))
  #       qnn_init = tf.keras.initializers.RandomUniform(
  #           minval=-q_const, maxval=q_const, seed=seed)
  #       test_qnn = qnn.QNN(r_circuit, initializer=qnn_init, is_analytic=True)

  #       # Confirm qhbm_model QHBM
  #       test_qhbm = qhbm.QHBM(test_ebm, test_qnn)
  #       test_thetas = test_qhbm.ebm.trainable_variables[0]
  #       test_phis = test_qhbm.qnn.trainable_variables[0]
  #       actual_log_partition = test_qhbm.log_partition_function()
  #       expected_log_partition = tf.reduce_sum(
  #           tf.math.log(2 * tf.math.cosh(test_thetas)))
  #       self.assertAllClose(
  #           actual_log_partition, expected_log_partition, atol=ATOL, rtol=RTOL)
  #       # Confirm qhbm_model modular Hamiltonian for 1 qubit case
  #       if num_qubits == 1:
  #         actual_dm = test_qhbm.density_matrix()
  #         actual_log_dm = tf.linalg.logm(actual_dm)
  #         actual_ktp = -actual_log_dm - tf.eye(
  #             2, dtype=tf.complex64) * tf.cast(actual_log_partition,
  #                                              tf.complex64)
  #         a = complex((test_thetas[0] * tf.math.cos(test_phis[0])).numpy(), 0)
  #         b = 1j * (test_thetas[0] * tf.math.sin(test_phis[0])).numpy()
  #         c = -1j * (test_thetas[0] * tf.math.sin(test_phis[0])).numpy()
  #         d = complex(-(test_thetas[0] * tf.math.cos(test_phis[0])).numpy(), 0)
  #         expected_ktp = tf.constant([[a, b], [c, d]], dtype=tf.complex64)
  #         self.assertAllClose(actual_ktp, expected_ktp, atol=ATOL, rtol=RTOL)

  #       # Build target data
  #       alphas = tf.random.uniform([num_qubits],
  #                                  minval=-q_const,
  #                                  maxval=q_const)
  #       y_rot = cirq.Circuit(
  #           cirq.ry(r.numpy())(q) for r, q in zip(alphas, qubits))
  #       data_probs = tf.random.uniform([num_qubits])
  #       raw_samples = tfp.distributions.Bernoulli(
  #           probs=1 - data_probs, dtype=tf.int8).sample(num_data_samples)
  #       unique_bitstrings, target_counts = utils.unique_bitstrings_with_counts(
  #           raw_samples)
  #       # Flip bits according to the distribution
  #       target_states_list = []
  #       for b in unique_bitstrings:
  #         c = y_rot.copy()
  #         for i, b_i in enumerate(b.numpy()):
  #           if b_i:
  #             c = cirq.X(qubits[i]) + c
  #         target_states_list.append(c)
  #       target_states = tfq.convert_to_tensor(target_states_list)

  #       with tf.GradientTape() as tape:
  #         actual_loss = qmhl_func(test_qhbm, target_states, target_counts)
  #       # TODO(zaqqwerty): add way to use a log QHBM as observable on states
  #       expected_expectation = tf.reduce_sum(
  #           test_thetas * (2 * data_probs - 1) * tf.math.cos(alphas) *
  #           tf.math.cos(test_phis))
  #       expected_loss = expected_expectation + expected_log_partition
  #       self.assertAllClose(actual_loss, expected_loss, atol=ATOL, rtol=RTOL)

  #       actual_thetas_grads, actual_phis_grads = tape.gradient(
  #           actual_loss, (test_thetas, test_phis))
  #       expected_thetas_grads = (2 * data_probs - 1) * tf.math.cos(
  #           alphas) * tf.math.cos(test_phis) + tf.math.tanh(test_thetas)
  #       expected_phis_grads = -test_thetas * (
  #           2 * data_probs - 1) * tf.math.cos(alphas) * tf.math.sin(test_phis)
  #       self.assertAllClose(
  #           actual_thetas_grads, expected_thetas_grads, atol=ATOL, rtol=RTOL)
  #       self.assertAllClose(
  #           actual_phis_grads, expected_phis_grads, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
  print("Running qmhl_loss_test.py ...")
  tf.test.main()
