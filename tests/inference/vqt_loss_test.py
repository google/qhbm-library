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
"""Tests for qhbmlib.inference.vqt_loss"""

import functools

import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib import inference
from qhbmlib import models
from tests import test_util


class VQTTest(tf.test.TestCase):
  """Tests for VQT."""

  def setUp(self):
    """Initializes test objects."""
    super().setUp()
    self.num_qubits_list = [1, 2]
    self.tf_random_seed = 5
    self.tf_random_seed_alt = 6
    self.tfp_seed = tf.constant([7, 8], tf.int32)
    self.tfp_seed_alt = tf.constant([9, 10], tf.int32)
    self.num_samples = int(1e7)
    self.close_rtol = 3e-2  # tolerance depends on samples
    self.zero_atol = 1e-3
    self.not_zero_atol = 2e-3

  @test_util.eager_mode_toggle
  def test_self_vqt(self):
    """Confirms known value of the VQT loss of a model against itself."""
    for num_qubits in self.num_qubits_list:
      qubits = cirq.GridQubit.rect(1, num_qubits)
      num_layers = 5
      data_h, data_infer = test_util.get_random_hamiltonian_and_inference(
          qubits,
          num_layers,
          f"data_objects_{num_qubits}",
          self.num_samples,
          ebm_seed=self.tfp_seed)
      model_h, model_infer = test_util.get_random_hamiltonian_and_inference(
          qubits,
          num_layers,
          f"hamiltonian_objects_{num_qubits}",
          self.num_samples,
          ebm_seed=self.tfp_seed)

      # Set data equal to the model
      data_h.set_weights(model_h.get_weights())

      beta = 1.0  # Data and model are only the same if beta == 1
      vqt = tf.function(inference.vqt)

      # Trained loss is minus log partition of the data.
      expected_loss = -1.0 * data_infer.e_inference.log_partition()
      # Since this is the optimum, derivatives should all be zero.
      expected_loss_derivative = [
          tf.zeros_like(v) for v in model_h.trainable_variables
      ]

      with tf.GradientTape() as tape:
        actual_loss = vqt(model_infer, data_h, beta)
      actual_loss_derivative = tape.gradient(actual_loss,
                                             model_h.trainable_variables)
      self.assertAllClose(actual_loss, expected_loss, self.close_rtol)
      self.assertAllClose(
          actual_loss_derivative, expected_loss_derivative, atol=self.zero_atol)

  @test_util.eager_mode_toggle
  def test_hamiltonian_vqt(self):
    """Tests derivatives of VQT with respect to both model and data."""

    vqt_wrapper = tf.function(inference.vqt)
    for num_qubits in self.num_qubits_list:
      qubits = cirq.GridQubit.rect(1, num_qubits)
      num_layers = 1
      data_h, _ = test_util.get_random_hamiltonian_and_inference(
          qubits,
          num_layers,
          f"data_objects_{num_qubits}",
          self.num_samples,
          initializer_seed=self.tf_random_seed,
          ebm_seed=self.tfp_seed)
      model_h, model_infer = test_util.get_random_hamiltonian_and_inference(
          qubits,
          num_layers,
          f"hamiltonian_objects_{num_qubits}",
          self.num_samples,
          initializer_seed=self.tf_random_seed_alt,
          ebm_seed=self.tfp_seed_alt)

      beta = tf.random.uniform([], 0.1, 10, tf.float32, self.tf_random_seed)

      with tf.GradientTape() as tape:
        actual_loss = vqt_wrapper(model_infer, data_h, beta)
      actual_gradient_model, actual_gradient_data = tape.gradient(
          actual_loss,
          (model_h.trainable_variables, data_h.trainable_variables))

      expected_gradient_model, expected_gradient_data = test_util.approximate_gradient(
          functools.partial(vqt_wrapper, model_infer, data_h, beta),
          (model_h.trainable_variables, data_h.trainable_variables))
      # Changing model parameters is working if finite difference derivatives
      # are non-zero.  Also confirms that model_h and data_h are different.
      tf.nest.map_structure(
          lambda x: self.assertAllGreater(tf.abs(x), self.not_zero_atol),
          expected_gradient_model)
      tf.nest.map_structure(
          lambda x: self.assertAllGreater(tf.abs(x), self.not_zero_atol),
          expected_gradient_data)
      self.assertAllClose(
          actual_gradient_model, expected_gradient_model, rtol=self.close_rtol)
      self.assertAllClose(
          actual_gradient_data, expected_gradient_data, rtol=self.close_rtol)

  @test_util.eager_mode_toggle
  def test_loss_value_x_rot(self):
    """Confirms correct values for a single qubit X rotation with H=Y.

    # TODO(#159): remove colab link
    See the colab notebook at the following link in for derivations:
    https://colab.research.google.com/drive/14987JCMju_8AVvvVoojwe6hA7Nlw-Dhe?usp=sharing

    Since each qubit is independent, the loss is the sum over the individual
    qubit losses, and the gradients are the the per-qubit gradients.
    """

    vqt = tf.function(inference.vqt)

    for num_qubits in self.num_qubits_list:
      # model definition
      ebm_init = tf.keras.initializers.RandomUniform(
          minval=-2.0, maxval=2.0, seed=self.tf_random_seed)
      actual_energy = models.BernoulliEnergy(list(range(num_qubits)), ebm_init)
      e_infer = inference.BernoulliEnergyInference(
          actual_energy, self.num_samples, initial_seed=self.tfp_seed)

      qubits = cirq.GridQubit.rect(1, num_qubits)
      r_symbols = [sympy.Symbol(f"phi_{n}") for n in range(num_qubits)]
      r_circuit = cirq.Circuit(
          cirq.rx(r_s)(q) for r_s, q in zip(r_symbols, qubits))
      qnn_init = tf.keras.initializers.RandomUniform(
          minval=-1, maxval=1, seed=self.tf_random_seed)
      actual_circuit = models.DirectQuantumCircuit(r_circuit, qnn_init)
      q_infer = inference.AnalyticQuantumInference(actual_circuit)
      qhbm_infer = inference.QHBM(e_infer, q_infer)

      # TODO(#171): code around here seems like boilerplate.
      model_h = qhbm_infer.modular_hamiltonian

      # Generate remaining VQT arguments
      test_h = tfq.convert_to_tensor(
          [cirq.PauliSum.from_pauli_strings(cirq.Y(q) for q in qubits)])
      test_beta = tf.random.uniform([], 0.01, 100.0, tf.float32,
                                    self.tf_random_seed)

      # Compute losses
      # Bernoulli has only one tf.Variable
      test_thetas = model_h.energy.trainable_variables[0]
      # QNN has only one tf.Variable
      test_phis = model_h.circuit.trainable_variables[0]
      actual_expectation = qhbm_infer.expectation(test_h)[0]
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
        actual_loss = vqt(qhbm_infer, test_h, test_beta)
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
