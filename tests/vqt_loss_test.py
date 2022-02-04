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

from qhbmlib import circuit_infer
from qhbmlib import circuit_model
from qhbmlib import energy_infer
from qhbmlib import energy_model
from qhbmlib import hamiltonian_infer
from qhbmlib import hamiltonian_model
from qhbmlib import vqt_loss
from tests import test_util

RTOL = 3e-2


class VQTTest(tf.test.TestCase):
  """Tests for VQT."""

  @test_util.eager_mode_toggle
  def test_loss_value_x_rot(self):
    """Confirms correct values for a single qubit X rotation with H=Y.

    # TODO(#159): remove colab link
    See the colab notebook at the following link in for derivations:
    https://colab.research.google.com/drive/14987JCMju_8AVvvVoojwe6hA7Nlw-Dhe?usp=sharing

    Since each qubit is independent, the loss is the sum over the individual
    qubit losses, and the gradients are the the per-qubit gradients.
    """

    @tf.function
    def vqt_wrapper(qhbm_infer, model, num_samples, hamiltonian, beta):
      return vqt_loss.vqt(qhbm_infer, model, num_samples, hamiltonian, beta)

    for num_qubits in [1, 2, 3, 4, 5]:
      # model definition
      tf_random_seed = None
      ebm_init = tf.keras.initializers.RandomUniform(
          minval=-2.0, maxval=2.0, seed=tf_random_seed)
      energy = energy_model.BernoulliEnergy(list(range(num_qubits)), ebm_init)
      energy.build([None, num_qubits])

      qubits = cirq.GridQubit.rect(1, num_qubits)
      r_symbols = [sympy.Symbol(f"phi_{n}") for n in range(num_qubits)]
      r_circuit = cirq.Circuit(
          cirq.rx(r_s)(q) for r_s, q in zip(r_symbols, qubits))
      qnn_init = tf.keras.initializers.RandomUniform(
          minval=-1, maxval=1, seed=tf_random_seed)
      circuit = circuit_model.DirectQuantumCircuit(r_circuit, qnn_init)
      circuit.build([None, num_qubits])

      model = hamiltonian_model.Hamiltonian(energy, circuit)

      # Inference definition
      tfp_seed = tf.constant([3, 4], tf.int32)
      e_infer = energy_infer.BernoulliEnergyInference(seed=tfp_seed)
      q_infer = circuit_infer.QuantumInference()
      qhbm_infer = hamiltonian_infer.QHBM(e_infer, q_infer)

      # Generate remaining VQT arguments
      test_num_samples = tf.constant(1e7)
      test_h = tfq.convert_to_tensor(
          [cirq.PauliSum.from_pauli_strings(cirq.Y(q) for q in qubits)])
      test_beta = tf.random.uniform([], 0.01, 100.0, seed=tf_random_seed)

      # Compute losses
      # Bernoulli has only one tf.Variable
      test_thetas = model.energy.trainable_variables[0]
      # QNN has only one tf.Variable
      test_phis = model.circuit.trainable_variables[0]
      actual_expectation = qhbm_infer.expectation(model, test_h,
                                                  test_num_samples)[0]
      expected_expectation = tf.reduce_sum(
          tf.math.tanh(test_thetas) * tf.math.sin(test_phis))
      self.assertAllClose(actual_expectation, expected_expectation, rtol=RTOL)

      actual_entropy = qhbm_infer.e_inference.entropy()
      expected_entropy = tf.reduce_sum(
          -test_thetas * tf.math.tanh(test_thetas) +
          tf.math.log(2 * tf.math.cosh(test_thetas)))
      self.assertAllClose(actual_entropy, expected_entropy, rtol=RTOL)

      with tf.GradientTape() as tape:
        actual_loss = vqt_wrapper(qhbm_infer, model, test_num_samples, test_h,
                                  test_beta)
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
  print("Running vqt_test.py ...")
  tf.test.main()
