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
import sympy
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_quantum as tfq

from qhbmlib import ebm
from qhbmlib import qhbm
from qhbmlib import qmhl
from qhbmlib import qnn
from qhbmlib import util
from tests import test_util

RTOL = 3e-2


class QMHLTest(tf.test.TestCase):
  """Tests for the QMHL loss and gradients."""

  def test_zero_grad(self):
    """Confirm correct gradients and loss at the optimal settings."""
    for num_qubits in [1, 2, 3, 4, 5]:
      qubits = cirq.GridQubit.rect(1, num_qubits)
      target = test_util.get_random_qhbm(qubits, 1,
                                         "QMHLLossTest{}".format(num_qubits))
      model = target.copy()

      # Get the QMHL loss gradients
      model_samples = tf.constant(1e6)
      target_samples = tf.constant(1e6)
      target_circuits, target_counts = target.circuits(target_samples)
      with tf.GradientTape() as tape:
        loss = qmhl.qmhl_loss(model, target_circuits, target_counts)
      thetas_grads, phis_grads = tape.gradient(loss, (model.thetas, model.phis))
      self.assertAllClose(loss, target.ebm.entropy(), atol=5e-3)
      self.assertAllClose(
          thetas_grads, tf.zeros(tf.shape(thetas_grads)), atol=5e-3)
      self.assertAllClose(phis_grads, tf.zeros(tf.shape(phis_grads)), atol=5e-3)

  def test_loss_value_x_rot(self):
    """Confirms correct values for a single qubit X rotation QHBM.

    We use a data state which is a Y rotation of an initially diagonal density
    operator.  The QHBM is a Bernoulli latent state with X rotation QNN.

    See the colab notebook at the following link for derivations:
    https://colab.research.google.com/drive/14987JCMju_8AVvvVoojwe6hA7Nlw-Dhe?usp=sharing

    Since each qubit is independent, the loss is the sum over the individual
    qubit losses, and the gradients are the the per-qubit gradients.
    """
    seed = 107
    for num_qubits in [1]:
      # EBM
      ebm_const = 2.0
      ebm_init = tf.keras.initializers.RandomUniform(minval=-ebm_const, maxval=ebm_const, seed=seed)
      test_ebm = ebm.Bernoulli(num_qubits, ebm_init, True)

      # QNN
      qubits = cirq.GridQubit.rect(1, num_qubits)
      q_const = 3 * 3.14159 / 2
      r_symbols = [sympy.Symbol(f"phi_{n}") for n in range(num_qubits)]
      r_circuit = cirq.Circuit(
          cirq.rx(r_s)(q) for r_s, q in zip(r_symbols, qubits))
      qnn_init = tf.keras.initializers.RandomUniform(minval=q_const, maxval=q_const, seed=seed)
      test_qnn = qnn.QNN(r_circuit, qnn_init, is_analytic=True)
      
      # Confirm model QHBM
      test_qhbm = qhbm.QHBM(test_ebm, test_qnn)
      test_thetas = test_qhbm.thetas[0]
      test_phis = test_qhbm.phis[0]
      print(f"test_thetas: {test_thetas}")
      print(f"test_phis: {test_phis}")
      actual_log_partition = test_qhbm.log_partition_function()
      expected_log_partition = tf.reduce_sum(
          tf.math.log(2 * tf.math.cosh(test_thetas)))
      self.assertAllClose(
          actual_log_partition, expected_log_partition, rtol=RTOL)
      # Confirm model modular Hamiltonian for 1 qubit case
      if num_qubits == 2:
        actual_dm = test_qhbm.density_matrix()
        actual_log_dm = tf.linalg.logm(actual_dm)
        actual_ktp = -actual_log_dm - tf.eye(2, dtype=tf.complex64) * tf.cast(actual_log_partition, tf.complex64)
        a = complex((test_thetas[0] * tf.math.cos(test_phis[0])).numpy(), 0)
        b = 1j * (test_thetas[0] * tf.math.sin(test_phis[0])).numpy()
        c = -1j * (test_thetas[0] * tf.math.sin(test_phis[0])).numpy()
        d = complex(-(test_thetas[0] * tf.math.cos(test_phis[0])).numpy(), 0)
        expected_ktp = tf.constant([[a, b], [c, d]], dtype=tf.complex64)
        self.assertAllClose(actual_ktp, expected_ktp, rtol=RTOL)
      
      # Build target data
#      alphas = tf.random.uniform([num_qubits], minval=-q_const, maxval=q_const)
      alphas = tf.constant([0.0])
      y_rot = cirq.Circuit(cirq.ry(r.numpy())(q) for r, q in zip(alphas, qubits))
      data_probs = tf.random.uniform([num_qubits])
      print(f"data_probs: {data_probs}")
      print(f"alphas: {alphas}")
      num_data_samples = 1e6
      raw_samples = tfp.distributions.Bernoulli(probs=1-data_probs, dtype=tf.int8).sample(num_data_samples)
      unique_bitstrings, target_counts = util.unique_bitstrings_with_counts(raw_samples)

      # TODO(zaqqwerty): add way to use a log QHBM as observable on states
      expected_expectation = tf.reduce_sum(test_thetas * (2 * data_probs - 1) *
                                           tf.math.cos(alphas) *
                                           tf.math.cos(test_phis))

      # Flip bits according to the distribution
      target_states_list = []
      for b in unique_bitstrings:
        c = y_rot.copy()
        for i, b_i in enumerate(b.numpy()):
          if b_i:
            c = cirq.X(qubits[i]) + c
        target_states_list.append(c)
        print(c)
      print(unique_bitstrings)
      print(target_counts)
      target_states = tfq.convert_to_tensor(target_states_list)
      
      with tf.GradientTape() as tape:
        actual_loss = qmhl.qmhl_loss(test_qhbm, target_states, target_counts)
      expected_loss = expected_expectation + expected_log_partition
      print(expected_log_partition)
      self.assertAllClose(actual_loss, expected_loss, rtol=RTOL)

      actual_thetas_grads, actual_phis_grads = tape.gradient(
          actual_loss, (test_thetas, test_phis))
      expected_thetas_grads = (2 * data_probs - 1) * tf.math.cos(
          alphas) * tf.math.cos(test_phis) + tf.math.tanh(test_thetas)
      expected_phis_grads = -test_thetas * (
          2 * data_probs - 1) * tf.math.cos(alphas) * tf.math.sin(test_phis)
      print(expected_thetas_grads)
      print(actual_thetas_grads)
      print(expected_phis_grads)
      print(actual_phis_grads)
      self.assertAllClose(actual_thetas_grads, expected_thetas_grads, rtol=RTOL)
      self.assertAllClose(actual_phis_grads, expected_phis_grads, rtol=RTOL)
      assert(False)


if __name__ == "__main__":
  print("Running qmhl_test.py ...")
  tf.test.main()
