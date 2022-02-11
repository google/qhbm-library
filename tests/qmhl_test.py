# pylint: skip-file
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

import math

import cirq
import sympy
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_quantum as tfq

from qhbmlib import ebm
from qhbmlib import qhbm
from qhbmlib import qmhl
from qhbmlib import qnn
from qhbmlib import utils
from tests import test_util

ATOL = 1e-3
RTOL = 3e-2


class QMHLTest(tf.test.TestCase):
  """Tests for the QMHL loss and gradients."""

  def test_zero_grad(self):
    """Confirm correct gradients and loss at the optimal settings."""
    for num_qubits in [1, 2, 3, 4, 5]:
      qubits = cirq.GridQubit.rect(1, num_qubits)
      target = test_util.get_random_qhbm(qubits, 1,
                                         "QMHLLossTest{}".format(num_qubits))
      qhbm_model = target.copy()

      # Get the QMHL loss gradients
      qhbm_model_samples = tf.constant(1e6)
      target_samples = tf.constant(1e6)
      target_circuits, target_counts = target.circuits(target_samples)
      with tf.GradientTape() as tape:
        loss = qmhl.qmhl(qhbm_model, target_circuits, target_counts)
      thetas_grads, phis_grads = tape.gradient(
          loss, (qhbm_model.ebm.trainable_variables,
                 qhbm_model.qnn.trainable_variables))
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
    seed = None
    ebm_const = 2.0
    q_const = 2 * math.pi
    num_data_samples = 1e6
    for qmhl_func in [
        qmhl.qmhl,
        tf.function(qmhl.qmhl, experimental_compile=False),
        #        tf.function(qmhl.qmhl, experimental_compile=True)
    ]:
      for num_qubits in [1, 2, 3, 4, 5]:
        # EBM
        ebm_init = tf.keras.initializers.RandomUniform(
            minval=-ebm_const, maxval=ebm_const, seed=seed)
        test_ebm = ebm.Bernoulli(num_qubits, ebm_init, True)

        # QNN
        qubits = cirq.GridQubit.rect(1, num_qubits)
        r_symbols = [sympy.Symbol(f"phi_{n}") for n in range(num_qubits)]
        r_circuit = cirq.Circuit(
            cirq.rx(r_s)(q) for r_s, q in zip(r_symbols, qubits))
        qnn_init = tf.keras.initializers.RandomUniform(
            minval=-q_const, maxval=q_const, seed=seed)
        test_qnn = qnn.QNN(r_circuit, initializer=qnn_init, is_analytic=True)

        # Confirm qhbm_model QHBM
        test_qhbm = qhbm.QHBM(test_ebm, test_qnn)
        test_thetas = test_qhbm.ebm.trainable_variables[0]
        test_phis = test_qhbm.qnn.trainable_variables[0]
        actual_log_partition = test_qhbm.log_partition_function()
        expected_log_partition = tf.reduce_sum(
            tf.math.log(2 * tf.math.cosh(test_thetas)))
        self.assertAllClose(
            actual_log_partition, expected_log_partition, atol=ATOL, rtol=RTOL)
        # Confirm qhbm_model modular Hamiltonian for 1 qubit case
        if num_qubits == 1:
          actual_dm = test_qhbm.density_matrix()
          actual_log_dm = tf.linalg.logm(actual_dm)
          actual_ktp = -actual_log_dm - tf.eye(
              2, dtype=tf.complex64) * tf.cast(actual_log_partition,
                                               tf.complex64)
          a = complex((test_thetas[0] * tf.math.cos(test_phis[0])).numpy(), 0)
          b = 1j * (test_thetas[0] * tf.math.sin(test_phis[0])).numpy()
          c = -1j * (test_thetas[0] * tf.math.sin(test_phis[0])).numpy()
          d = complex(-(test_thetas[0] * tf.math.cos(test_phis[0])).numpy(), 0)
          expected_ktp = tf.constant([[a, b], [c, d]], dtype=tf.complex64)
          self.assertAllClose(actual_ktp, expected_ktp, atol=ATOL, rtol=RTOL)

        # Build target data
        alphas = tf.random.uniform([num_qubits],
                                   minval=-q_const,
                                   maxval=q_const)
        y_rot = cirq.Circuit(
            cirq.ry(r.numpy())(q) for r, q in zip(alphas, qubits))
        data_probs = tf.random.uniform([num_qubits])
        raw_samples = tfp.distributions.Bernoulli(
            probs=1 - data_probs, dtype=tf.int8).sample(num_data_samples)
        unique_bitstrings, _, target_counts = utils.unique_bitstrings_with_counts(
            raw_samples)
        # Flip bits according to the distribution
        target_states_list = []
        for b in unique_bitstrings:
          c = y_rot.copy()
          for i, b_i in enumerate(b.numpy()):
            if b_i:
              c = cirq.X(qubits[i]) + c
          target_states_list.append(c)
        target_states = tfq.convert_to_tensor(target_states_list)

        with tf.GradientTape() as tape:
          actual_loss = qmhl_func(test_qhbm, target_states, target_counts)
        # TODO(zaqqwerty): add way to use a log QHBM as observable on states
        expected_expectation = tf.reduce_sum(
            test_thetas * (2 * data_probs - 1) * tf.math.cos(alphas) *
            tf.math.cos(test_phis))
        expected_loss = expected_expectation + expected_log_partition
        self.assertAllClose(actual_loss, expected_loss, atol=ATOL, rtol=RTOL)

        actual_thetas_grads, actual_phis_grads = tape.gradient(
            actual_loss, (test_thetas, test_phis))
        expected_thetas_grads = (2 * data_probs - 1) * tf.math.cos(
            alphas) * tf.math.cos(test_phis) + tf.math.tanh(test_thetas)
        expected_phis_grads = -test_thetas * (
            2 * data_probs - 1) * tf.math.cos(alphas) * tf.math.sin(test_phis)
        self.assertAllClose(
            actual_thetas_grads, expected_thetas_grads, atol=ATOL, rtol=RTOL)
        self.assertAllClose(
            actual_phis_grads, expected_phis_grads, atol=ATOL, rtol=RTOL)

  def test_hypernetwork(self):
    for num_qubits in [1, 2, 3, 4, 5]:
      qubits = cirq.GridQubit.rect(1, num_qubits)
      target = test_util.get_random_qhbm(
          qubits, 1, "QMHLHypernetworkTest{}".format(num_qubits))
      qhbm_model = target.copy()

      trainable_variables_shapes = [
          tf.shape(var) for var in qhbm_model.trainable_variables
      ]
      trainable_variables_sizes = [
          tf.size(var) for var in qhbm_model.trainable_variables
      ]
      trainable_variables_size = tf.reduce_sum(
          tf.stack(trainable_variables_sizes))

      input_size = 15
      hypernetwork = tf.keras.Sequential([
          tf.keras.layers.Dense(15, 'relu', input_shape=(input_size,)),
          tf.keras.layers.Dense(10, 'tanh'),
          tf.keras.layers.Dense(5, 'sigmoid'),
          tf.keras.layers.Dense(trainable_variables_size)
      ])
      input = tf.random.uniform([1, input_size])

      # Get the QMHL loss gradients
      qhbm_model_samples = tf.constant(1e6)
      target_samples = tf.constant(1e6)
      target_circuits, target_counts = target.circuits(target_samples)
      with tf.GradientTape() as tape:
        output = tf.squeeze(hypernetwork(input))
        index = 0
        output_trainable_variables = []
        for size, shape in zip(trainable_variables_sizes,
                               trainable_variables_shapes):
          output_trainable_variables.append(
              tf.reshape(output[index:index + size], shape))
          index += size
        qhbm_model.trainable_variables = output_trainable_variables
        loss = qmhl.qmhl(qhbm_model, target_circuits, target_counts)

      grads = tape.gradient(loss, [
          hypernetwork.trainable_variables, output,
          qhbm_model.trainable_variables
      ])
      hyper_grads = grads[0]
      output_grad = grads[1]
      qhbm_grads = grads[2]

      qhbm_grad_flat = []
      for grad in qhbm_grads:
        qhbm_grad_flat.append(tf.reshape(grad, [-1]))
      qhbm_grad_flat = tf.concat(qhbm_grad_flat, 0)
      self.assertAllEqual(qhbm_grad_flat, output_grad)

      for grad in hyper_grads:
        self.assertIsNotNone(grad)

      c = tf.Variable(tf.random.uniform([trainable_variables_size]))
      input = tf.random.uniform([trainable_variables_size])
      with tf.GradientTape() as tape:
        output = c * input
        index = 0
        output_trainable_variables = []
        for size, shape in zip(trainable_variables_sizes,
                               trainable_variables_shapes):
          output_trainable_variables.append(
              tf.reshape(output[index:index + size], shape))
          index += size
        qhbm_model.trainable_variables = output_trainable_variables
        loss = qmhl.qmhl(qhbm_model, target_circuits, target_counts)
      grads = tape.gradient(loss, [c, qhbm_model.trainable_variables])
      c_grad = grads[0]
      qhbm_grads = grads[1]
      qhbm_grad_flat = []
      for grad in qhbm_grads:
        qhbm_grad_flat.append(tf.reshape(grad, [-1]))
      qhbm_grad_flat = tf.concat(qhbm_grad_flat, 0)
      self.assertAllEqual(input * qhbm_grad_flat, c_grad)


if __name__ == "__main__":
  print("Running qmhl_test.py ...")
  tf.test.main()
