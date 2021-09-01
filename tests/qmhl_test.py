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

from qhbmlib import ebm
from qhbmlib import qhbm
from qhbmlib import qmhl
from qhbmlib import qnn
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

    We use a data state which is the thermal state of a Pauli Y at inverse
    temperature beta.  The QHBM is a Bernoulli latent state with X rotation QNN.

    See the colab notebook at the following link for derivations:
    https://colab.research.google.com/drive/14987JCMju_8AVvvVoojwe6hA7Nlw-Dhe?usp=sharing

    Since each qubit is independent, the loss is the sum over the individual
    qubit losses, and the gradients are the the per-qubit gradients.
    """
    seed = None
    for num_qubits in [1, 2, 3, 4, 5]:
      # EBM
      ebm_init = tf.keras.initializers.RandomUniform(
          minval=-2.0, maxval=2.0, seed=seed)
      test_ebm = ebm.Bernoulli(num_qubits, ebm_init, True)

      # QNN
      qubits = cirq.GridQubit.rect(1, num_qubits)
      q_const_limit = 6.2
      r_symbols = [sympy.Symbol(f"phi_{n}") for n in range(num_qubits)]
      r_circuit = cirq.Circuit(
          cirq.rx(r_s)(q) for r_s, q in zip(r_symbols, qubits))
      qnn_init = tf.keras.initializers.RandomUniform(
          minval=-q_const_limit, maxval=q_const_limit, seed=seed)
      test_qnn = qnn.QNN(r_circuit, qnn_init, is_analytic=True)

      # Build target data
      alphas = tf.random.uniform([num_qubits], minval=-q_const, maxval=q_const)
      y_rot = cirq.Circuit(cirq.ry(r)(q) for r, q in zip(alphas, qubits))
      data_probs = tf.random.uniform([num_qubits])
      count_scale = 1e6
      target_states_list = []
      target_counts_list = []
      # Enumerate all possible excitations.
      for m in range(2 ** num_qubits):
        c = y_rot.copy()
        p = 1
        for n, q in enumerate(qubits):
          if m % 2:  # state |1> on qubit n
            c += cirq.X(q)
            m //= 2
            p *= (1 - data_probs[n])
          p *= data_probs[n]
        target_states_list.append(c)
        target_counts_list.append(round(p * count_scale))
      target_states = tfq.convert_to_tensor(target_states_list)
      target_counts = tf.convert_to_tensor*(target_counts_list)
      
      # Compute losses
      test_qhbm = qhbm.QHBM(test_ebm, test_qnn)
      test_thetas = test_qhbm.thetas[0]
      test_phis = test_qhbm.phis[0]
      actual_expectation = test_qhbm.pulled_back_expectationtest_qhbm.expectation(test_h, test_num_samples)[0]
      expected_expectation = tf.reduce_sum(
          tf.math.tanh(test_thetas) * tf.math.sin(test_phis))
      self.assertAllClose(actual_expectation, expected_expectation, rtol=RTOL)

      actual_entropy = test_qhbm.entropy()
      expected_entropy = tf.reduce_sum(
          -test_thetas * tf.math.tanh(test_thetas) +
          tf.math.log(2 * tf.math.cosh(test_thetas)))
      self.assertAllClose(actual_entropy, expected_entropy, rtol=RTOL)

      with tf.GradientTape() as tape:
        actual_loss = vqt.vqt(test_qhbm, test_num_samples, test_h, test_beta)
      expected_loss = test_beta * expected_expectation - expected_entropy
      self.assertAllClose(actual_loss, expected_loss, rtol=RTOL)

      actual_thetas_grads, actual_phis_grads = tape.gradient(
          actual_loss, (test_thetas, test_phis))
      expected_thetas_grads = (1 - tf.math.tanh(test_thetas)**2) * (
          test_beta * tf.math.sin(test_phis) + test_thetas)
      expected_phis_grads = test_beta * tf.math.tanh(test_thetas) * tf.math.cos(
          test_phis)
      self.assertAllClose(actual_thetas_grads, expected_thetas_grads, rtol=RTOL)
      self.assertAllClose(actual_phis_grads, expected_phis_grads, rtol=RTOL)


if __name__ == "__main__":
  print("Running qmhl_test.py ...")
  tf.test.main()
