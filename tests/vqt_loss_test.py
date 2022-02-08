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


class VQTTest(tf.test.TestCase):
  """Tests for VQT."""

  def setUp(self):
    """Initializes test objects."""
    super().setUp()
    self.num_qubits_list = [1, 2, 3, 4]
    self.tf_random_seed = 7
    self.tfp_seed = tf.constant([3, 4], tf.int32)
    self.close_rtol = 1e-2
    self.zero_atol = 1e-4
    self.not_zero_atol = 1e-1

  @test_util.eager_mode_toggle
  def test_self_vqt(self):
    """Confirms known value of the VQT loss of a model against itself."""
    for num_qubits in self.num_qubits_list:
      qubits = cirq.GridQubit.rect(1, num_qubits)
      num_layers = 5
      data_h, data_infer = test_util.get_random_hamiltonian_and_inference(qubits, num_layers,
                                                                          f"data_objects_{num_qubits}", ebm_seed=self.tfp_seed)
      model_h, model_infer = test_util.get_random_hamiltonian_and_inference(qubits, num_layers,
                                                                            f"hamiltonian_objects_{num_qubits}", ebm_seed=self.tfp_seed)

      # Set data equal to the model
      data_h.set_weights(model_h.get_weights())
      data_infer.e_inference.infer(data_h.energy)

      num_samples = tf.constant(int(5e6))
      beta = 1.0
      vqt = tf.function(vqt_loss.vqt)

      # Trained loss is minus log partition of the data.
      expected_loss = -1.0 * data_infer.e_inference.log_partition()
      # Since this is the optimum, derivatives should all be zero.
      expected_loss_derivatives = [
        tf.zeros_like(v) for v in model_h.trainable_variables]

      with tf.GradientTape() as tape:
        actual_loss = vqt(model_infer, model_h, num_samples, data_h, beta)
      actual_loss_derivative = tape.gradient(actual_loss, model_h.trainable_variables)
      self.assertAllClose(actual_loss, expected_loss, self.close_rtol)
      self.assertAllClose(actual_loss_derivative, expected_loss_derivative, atol=self.zero_atol)

  @test_util.eager_mode_toggle
  def test_loss_value_x_rot(self):
    """Confirms correct values for a single qubit X rotation with H=Y.

    # TODO(#159): remove colab link
    See the colab notebook at the following link in for derivations:
    https://colab.research.google.com/drive/14987JCMju_8AVvvVoojwe6hA7Nlw-Dhe?usp=sharing

    Since each qubit is independent, the loss is the sum over the individual
    qubit losses, and the gradients are the the per-qubit gradients.
    """

    vqt = tf.function(vqt_loss.vqt)

    for num_qubits in self.num_qubits_list:
      # model definition
      ebm_init = tf.keras.initializers.RandomUniform(
          minval=-2.0, maxval=2.0, seed=self.tf_random_seed)
      energy = energy_model.BernoulliEnergy(list(range(num_qubits)), ebm_init)
      energy.build([None, num_qubits])

      qubits = cirq.GridQubit.rect(1, num_qubits)
      r_symbols = [sympy.Symbol(f"phi_{n}") for n in range(num_qubits)]
      r_circuit = cirq.Circuit(
          cirq.rx(r_s)(q) for r_s, q in zip(r_symbols, qubits))
      qnn_init = tf.keras.initializers.RandomUniform(
          minval=-1, maxval=1, seed=self.tf_random_seed)
      circuit = circuit_model.DirectQuantumCircuit(r_circuit, qnn_init)
      circuit.build([None, num_qubits])

      model = hamiltonian_model.Hamiltonian(energy, circuit)

      # Inference definition
      e_infer = energy_infer.BernoulliEnergyInference(seed=self.tfp_seed)
      q_infer = circuit_infer.QuantumInference()
      qhbm_infer = hamiltonian_infer.QHBM(e_infer, q_infer)

      # Generate remaining VQT arguments
      test_num_samples = tf.constant(5e6)
      test_h = tfq.convert_to_tensor(
          [cirq.PauliSum.from_pauli_strings(cirq.Y(q) for q in qubits)])
      test_beta = tf.random.uniform([], 0.01, 100.0, tf.float32,
                                    self.tf_random_seed)

      # Compute losses
      # Bernoulli has only one tf.Variable
      test_thetas = model.energy.trainable_variables[0]
      # QNN has only one tf.Variable
      test_phis = model.circuit.trainable_variables[0]
      actual_expectation = qhbm_infer.expectation(model, test_h,
                                                  test_num_samples)[0]
      expected_expectation = tf.reduce_sum(
          tf.math.tanh(test_thetas) * tf.math.sin(test_phis))
      self.assertAllClose(
          actual_expectation, expected_expectation, rtol=self.close_rtol)

      actual_entropy = qhbm_infer.e_inference.entropy()
      expected_entropy = tf.reduce_sum(
          -test_thetas * tf.math.tanh(test_thetas) +
          tf.math.log(2 * tf.math.cosh(test_thetas)))
      self.assertAllClose(
          actual_entropy, expected_entropy, rtol=self.close_rtol)

      with tf.GradientTape() as tape:
        actual_loss = vqt(qhbm_infer, model, test_num_samples, test_h,
                          test_beta)
      expected_loss = test_beta * expected_expectation - expected_entropy
      self.assertAllClose(actual_loss, expected_loss, rtol=self.close_rtol)

      actual_thetas_grads, actual_phis_grads = tape.gradient(
          actual_loss, (test_thetas, test_phis))
      expected_thetas_grads = (1 - tf.math.tanh(test_thetas)**2) * (
          test_beta * tf.math.sin(test_phis) + test_thetas)
      expected_phis_grads = test_beta * tf.math.tanh(test_thetas) * tf.math.cos(
          test_phis)
      self.assertAllClose(
          actual_thetas_grads, expected_thetas_grads, rtol=self.close_rtol)
      self.assertAllClose(
          actual_phis_grads, expected_phis_grads, rtol=self.close_rtol)


if __name__ == "__main__":
  print("Running vqt_loss_test.py ...")
  tf.test.main()
