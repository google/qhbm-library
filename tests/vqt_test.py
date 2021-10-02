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
"""Tests for the VQT loss and gradients."""

import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib import ebm
from qhbmlib import qhbm
from qhbmlib import qnn
from qhbmlib import vqt
from tests import test_util

RTOL = 3e-2


class VQTTest(tf.test.TestCase):
  """Tests for the sample-based VQT."""

  num_bits = 5
  raw_phis_symbols = [sympy.Symbol("s0"), sympy.Symbol("s1")]
  phis_symbols = tf.constant([str(s) for s in raw_phis_symbols])
  raw_qubits = cirq.GridQubit.rect(1, num_bits)
  u = cirq.Circuit()
  for s in raw_phis_symbols:
    for q in raw_qubits:
      u += cirq.X(q)**s
  name = "TestQHBM"

  def test_loss_consistency(self):
    """Confirms that the sample-based and exact losses are close."""
    test_ebm = ebm.Bernoulli(self.num_bits, is_analytic=True)
    test_qnn = qnn.QNN(self.u, is_analytic=True)
    test_qhbm = qhbm.QHBM(test_ebm, test_qnn, self.name)
    test_ebm_copy = ebm.Bernoulli(self.num_bits, is_analytic=False)
    test_qnn_copy = qnn.QNN(self.u, is_analytic=False)
    test_qhbm_copy = qhbm.QHBM(test_ebm_copy, test_qnn_copy, self.name)
    num_samples = tf.constant(int(5e6))
    num_random_hamiltonians = 2
    for beta in tf.constant([0.1, 0.4, 1.6, 6.4]):
      for _ in range(num_random_hamiltonians):
        cirq_ham = test_util.get_random_pauli_sum(self.raw_qubits)
        tf_ham = tfq.convert_to_tensor([cirq_ham])
        loss = vqt.vqt(test_qhbm, num_samples, tf_ham, beta)
        loss_copy = vqt.vqt(test_qhbm_copy, num_samples, tf_ham, beta)
        self.assertAllClose(loss_copy, loss, rtol=RTOL)

  def test_zero_grad(self):
    """Confirm correct gradients and loss at the optimal settings."""
    qubit = cirq.GridQubit(0, 0)
    cirq_ham = cirq.X(qubit)
    tf_ham = tfq.convert_to_tensor([cirq_ham])
    test_ebm = ebm.Bernoulli(1, is_analytic=True)
    test_ebm.kernel.assign(tf.constant([1.0]))
    symbol = sympy.Symbol("p")
    pqc = cirq.Circuit(cirq.H(qubit)**symbol)
    test_qnn = qnn.QNN(pqc, is_analytic=True)
    test_qnn.values.assign(tf.constant([1.0]))
    test_qhbm = qhbm.QHBM(test_ebm, test_qnn)
    with tf.GradientTape() as tape:
      loss = vqt.vqt(test_qhbm, tf.constant(int(5e6)), tf_ham, tf.constant(1.0))
    gradient = tape.gradient(loss, test_qhbm.trainable_variables)
    for grad in gradient:
      self.assertAllClose(grad, tf.zeros_like(grad), rtol=RTOL)
    self.assertAllClose(loss, -test_qhbm.log_partition_function(), rtol=RTOL)

  def test_loss_value_x_rot(self):
    """Confirms correct values for a single qubit X rotation with H=Y.

    See the colab notebook at the following link in for derivations:
    https://colab.research.google.com/drive/14987JCMju_8AVvvVoojwe6hA7Nlw-Dhe?usp=sharing

    Since each qubit is independent, the loss is the sum over the individual
    qubit losses, and the gradients are the the per-qubit gradients.
    """
    for vqt_func in [
        vqt.vqt,
        tf.function(vqt.vqt, experimental_compile=False),
        #        tf.function(vqt.vqt, experimental_compile=True)
    ]:
      seed = None
      for num_qubits in [1, 2, 3, 4, 5]:
        # EBM
        ebm_init = tf.keras.initializers.RandomUniform(
            minval=-2.0, maxval=2.0, seed=seed)
        test_ebm = ebm.Bernoulli(num_qubits, ebm_init, True)

        # QNN
        qubits = cirq.GridQubit.rect(1, num_qubits)
        r_symbols = [sympy.Symbol(f"phi_{n}") for n in range(num_qubits)]
        r_circuit = cirq.Circuit(
            cirq.rx(r_s)(q) for r_s, q in zip(r_symbols, qubits))
        qnn_init = tf.keras.initializers.RandomUniform(
            minval=-6.2, maxval=6.2, seed=seed)
        test_qnn = qnn.QNN(r_circuit, qnn_init, is_analytic=True)

        # VQT arguments
        test_qhbm = qhbm.QHBM(test_ebm, test_qnn)
        test_num_samples = tf.constant(1e7)
        test_h = tfq.convert_to_tensor(
            [cirq.PauliSum.from_pauli_strings(cirq.Y(q) for q in qubits)])
        test_beta = tf.random.uniform([], minval=0.01, maxval=100.0, seed=seed)

        # Compute losses
        # Bernoulli has only one tf.Variable
        test_thetas = test_qhbm.ebm.trainable_variables[0]
        # QNN has only one tf.Variable
        test_phis = test_qhbm.qnn.trainable_variables[0]
        actual_expectation = test_qhbm.expectation(test_h, test_num_samples)[0]
        expected_expectation = tf.reduce_sum(
            tf.math.tanh(test_thetas) * tf.math.sin(test_phis))
        self.assertAllClose(actual_expectation, expected_expectation, rtol=RTOL)

        actual_entropy = test_qhbm.entropy()
        expected_entropy = tf.reduce_sum(
            -test_thetas * tf.math.tanh(test_thetas) +
            tf.math.log(2 * tf.math.cosh(test_thetas)))
        self.assertAllClose(actual_entropy, expected_entropy, rtol=RTOL)

        with tf.GradientTape() as tape:
          actual_loss = vqt_func(test_qhbm, test_num_samples, test_h, test_beta)
        expected_loss = test_beta * expected_expectation - expected_entropy
        self.assertAllClose(actual_loss, expected_loss, rtol=RTOL)

        actual_thetas_grads, actual_phis_grads = tape.gradient(
            actual_loss, (test_thetas, test_phis))
        expected_thetas_grads = (1 - tf.math.tanh(test_thetas)**2) * (
            test_beta * tf.math.sin(test_phis) + test_thetas)
        expected_phis_grads = test_beta * tf.math.tanh(
            test_thetas) * tf.math.cos(test_phis)
        self.assertAllClose(
            actual_thetas_grads, expected_thetas_grads, rtol=RTOL)
        self.assertAllClose(actual_phis_grads, expected_phis_grads, rtol=RTOL)

  def test_hypernetwork(self):
    for num_qubits in [1, 2, 3, 4, 5]:
      qubits = cirq.GridQubit.rect(1, num_qubits)
      test_qhbm = test_util.get_random_qhbm(qubits, 1,
                                            "VQTHyperTest{}".format(num_qubits))
      ham = test_util.get_random_pauli_sum(qubits)
      tf_ham = tfq.convert_to_tensor([ham])
      trainable_variables_shapes = [
          tf.shape(var) for var in test_qhbm.trainable_variables
      ]
      trainable_variables_sizes = [
          tf.size(var) for var in test_qhbm.trainable_variables
      ]
      trainable_variables_size = tf.reduce_sum(
          tf.stack(trainable_variables_sizes))

      input_size = 15
      hypernetwork = tf.keras.Sequential([
          tf.keras.layers.Dense(15, 'relu', input_shape=(input_size,)),
          tf.keras.layers.Dense(10, 'tanh', input_shape=(input_size,)),
          tf.keras.layers.Dense(5, 'sigmoid', input_shape=(input_size,)),
          tf.keras.layers.Dense(trainable_variables_size)
      ])
      input = tf.random.uniform([1, input_size])

      with tf.GradientTape() as tape:
        output = tf.squeeze(hypernetwork(input))
        index = 0
        output_trainable_variables = []
        for size, shape in zip(trainable_variables_sizes,
                               trainable_variables_shapes):
          output_trainable_variables.append(
              tf.reshape(output[index:index + size], shape))
          index += size
        test_qhbm.trainable_variables = output_trainable_variables
        loss = vqt.vqt(test_qhbm, tf.constant(int(5e6)), tf_ham,
                       tf.constant(1.0))
      grads = tape.gradient(loss, test_qhbm.trainable_variables)
      for grad in grads:
        self.assertIsNotNone(grad)


if __name__ == "__main__":
  print("Running vqt_test.py ...")
  tf.test.main()
