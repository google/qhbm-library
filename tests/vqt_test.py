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
    test_ebm._variables.assign(tf.constant([1.0]))
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

    See the notebook linked in the PR for derivations.
    Since each qubit is independent, the loss is the sum over the individual
    qubit losses, and the gradients are the the per-qubit gradients.
    """
    seed = None
    for num_qubits in [1, 2, 3, 4, 5]:
      # EBM
      ebm_init = tf.keras.initializers.RandomUniform(
          minval=-2.0, maxval=2.0, seed=seed)
      test_ebm = ebm.Bernoulli(num_qubits, ebm_init, True)
      test_thetas = test_ebm.trainable_variables[0]

      # QNN
      qubits = cirq.GridQubit.rect(1, num_qubits)
      r_symbols = [sympy.Symbol(f"phi_{n}") for n in range(num_qubits)]
      r_circuit = cirq.Circuit(
          cirq.rx(r_s)(q) for r_s, q in zip(r_symbols, qubits))
      qnn_init = tf.keras.initializers.RandomUniform(
          minval=-6.2, maxval=6.2, seed=seed)
      test_qnn = qnn.QNN(r_circuit, qnn_init, is_analytic=True)
      test_phis = test_qnn.values

      # VQT arguments
      test_qhbm = qhbm.QHBM(test_ebm, test_qnn)
      test_num_samples = tf.constant(1000000)
      test_h = tfq.convert_to_tensor(
          [cirq.PauliSum.from_pauli_strings(cirq.Y(q) for q in qubits)])
      test_beta = tf.random.uniform([], minval=0.01, maxval=100.0, seed=seed)

      # Compute losses
      test_expectation = test_qhbm.expectation(test_h, test_num_samples)[0]
      ref_expectation = tf.reduce_sum(
          tf.math.tanh(test_thetas) * tf.math.sin(test_phis))

      self.assertAllClose(test_expectation, ref_expectation, rtol=RTOL)
      test_entropy = test_qhbm.entropy()
      ref_entropy = tf.reduce_sum(-test_thetas * tf.math.tanh(test_thetas) +
                                  tf.math.log(2 * tf.math.cosh(test_thetas)))
      self.assertAllClose(test_entropy, ref_entropy, rtol=RTOL)

      with tf.GradientTape() as tape:
        test_loss = vqt.vqt(test_qhbm, test_num_samples, test_h, test_beta)
      ref_loss = test_beta * ref_expectation - ref_entropy
      self.assertAllClose(test_loss, ref_loss, rtol=RTOL)

      test_thetas_grads, test_phis_grads = tape.gradient(
          test_loss, (test_ebm.trainable_variables[0], test_qnn.values))
      ref_thetas_grads = (1 - tf.math.tanh(test_thetas)**2) * (
          test_beta * tf.math.sin(test_phis) + test_thetas)
      ref_phis_grads = test_beta * tf.math.tanh(test_thetas) * tf.math.cos(
          test_phis)

      self.assertAllClose(test_thetas_grads, ref_thetas_grads, rtol=RTOL)
      self.assertAllClose(test_phis_grads, ref_phis_grads, rtol=RTOL)


if __name__ == "__main__":
  print("Running vqt_test.py ...")
  tf.test.main()
