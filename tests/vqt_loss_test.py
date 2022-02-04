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
      test_beta = tf.random.uniform([],
                                    minval=0.01,
                                    maxval=100.0,
                                    seed=tf_random_seed)

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

  def test_hypernetwork(self):
    for num_qubits in [1, 2, 3, 4, 5]:
      qubits = cirq.GridQubit.rect(1, num_qubits)
      model, qhbm_infer = test_util.get_random_hamiltonian_and_inference(
          qubits, 1, f"VQTHyperTest{num_qubits}")
      ham = test_util.get_random_pauli_sum(qubits)
      tf_ham = tfq.convert_to_tensor([ham])

      # There is only one variable in energy and one in circuit.
      energy_trainable_variables_size = tf.size(
          model.energy.post_process[0].kernel)
      circuit_trainable_variables_size = tf.size(
          model.circuit.value_layers_inputs[0])
      trainable_variables_size = energy_trainable_variables_size + circuit_trainable_variables_size

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
        # The model variables are 1D.
        model_parameters = [
            output[:energy_trainable_variables_size],
            output[energy_trainable_variables_size:]
        ]
        model.energy.post_process[0].kernel = model_parameters[0]
        model.circuit.value_layers_inputs[0] = model_parameters[1]
        num_samples = tf.constant(int(5e6))
        beta = tf.constant(1.0)
        loss = vqt_loss.vqt(qhbm_infer, model, num_samples, tf_ham, beta)
      grads = tape.gradient(
          loss, [hypernetwork.trainable_variables, output, model_parameters])
      print(f"grads: {grads}")
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
        model_parameters = [
            output[:energy_trainable_variables_size],
            output[energy_trainable_variables_size:]
        ]
        model.energy.post_process[0].kernel = model_parameters[0]
        model.circuit.value_layers_inputs[0] = model_parameters[1]
        loss = vqt_loss.vqt(qhbm_infer, model, num_samples, tf_ham, beta)
      grads = tape.gradient(loss, [c, model_parameters])
      c_grad = grads[0]
      qhbm_grads = grads[1]
      qhbm_grad_flat = []
      for grad in qhbm_grads:
        qhbm_grad_flat.append(tf.reshape(grad, [-1]))
      qhbm_grad_flat = tf.concat(qhbm_grad_flat, 0)
      self.assertAllEqual(input * qhbm_grad_flat, c_grad)


if __name__ == "__main__":
  print("Running vqt_test.py ...")
  tf.test.main()
