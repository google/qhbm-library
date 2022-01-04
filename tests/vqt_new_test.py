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
"""Tests for the VQT loss and gradients."""

import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib import architectures
from qhbmlib import circuit_infer
from qhbmlib import circuit_model
from qhbmlib import energy_infer
from qhbmlib import energy_model
from qhbmlib import hamiltonian_infer
from qhbmlib import hamiltonian_model
from qhbmlib import vqt_new
from tests import test_util

RTOL = 3e-2


class VQTTest(tf.test.TestCase):
  """Tests for the sample-based VQT."""

  def setUp(self):
    """Initializes test objects."""

    num_bits = 5
    raw_phis_symbols = [sympy.Symbol("s0"), sympy.Symbol("s1")]
    phis_symbols = tf.constant([str(s) for s in raw_phis_symbols])
    raw_qubits = cirq.GridQubit.rect(1, num_bits)
    u = cirq.Circuit()
    for s in raw_phis_symbols:
      for q in raw_qubits:
        u += cirq.X(q)**s

  def test_zero_grad(self):
    """Confirm correct gradients and loss at the optimal settings."""

    # energy
    energy = energy_model.BernoulliEnergy([0])
    energy.build([None, 1])
    energy.set_weights([tf.constant([1.0])])

    # circuit
    qubit = cirq.GridQubit(0, 0)
    symbol = sympy.Symbol("p")
    pqc = cirq.Circuit(cirq.H(qubit)**symbol)
    test_qnn = circuit_model.DirectQuantumCircuit(pqc)
    test_qnn.build([])
    test_qnn.set_weights([tf.constant([1.0])])
    test_model = hamiltonian_model.Hamiltonian(energy, test_qnn)

    # other stuff
    cirq_ham = cirq.X(qubit)
    tf_ham = tfq.convert_to_tensor([cirq_ham])
    e_inf = energy_infer.BernoulliEnergyInference()
    q_inf = circuit_infer.QuantumInference()
    test_qhbm = hamiltonian_infer.QHBM(e_inf, q_inf)
    
    with tf.GradientTape() as tape:
      loss = vqt_new.vqt(test_qhbm, test_model, tf.constant(int(5e6)), tf_ham, tf.constant(1.0))
    gradient = tape.gradient(loss, test_qhbm.trainable_variables)
    for grad in gradient:
      self.assertAllClose(grad, tf.zeros_like(grad), rtol=RTOL)
    self.assertAllClose(loss, -test_qhbm.e_inference.log_partition(), rtol=RTOL)

  def test_loss_value_x_rot(self):
    """Confirms correct values for a single qubit X rotation with H=Y.

    See the colab notebook at the following link in for derivations:
    https://colab.research.google.com/drive/14987JCMju_8AVvvVoojwe6hA7Nlw-Dhe?usp=sharing

    Since each qubit is independent, the loss is the sum over the individual
    qubit losses, and the gradients are the the per-qubit gradients.
    """
    for vqt_func in [
        vqt_new.vqt,
        tf.function(vqt_new.vqt, experimental_compile=False),
        #        tf.function(vqt.vqt, experimental_compile=True)
    ]:
      seed = 79
      for num_qubits in [1, 2, 3, 4, 5]:
        # set up model
        ebm_init = tf.keras.initializers.RandomUniform(
            minval=-2.0, maxval=2.0, seed=seed)
        e_model = energy_model.BernoulliEnergy(list(range(num_qubits)), ebm_init)
        e_model.build([None, num_qubits])

        qubits = cirq.GridQubit.rect(1, num_qubits)
        r_symbols = [sympy.Symbol(f"phi_{n}") for n in range(num_qubits)]
        r_circuit = cirq.Circuit(
            cirq.rx(r_s)(q) for r_s, q in zip(r_symbols, qubits))
        qnn_init = tf.keras.initializers.RandomUniform(
            minval=-6.2, maxval=6.2, seed=seed)
        q_model = circuit_model.DirectQuantumCircuit(r_circuit, qnn_init)
        q_model.build([])

        h_model = hamiltonian_model.Hamiltonian(e_model, q_model)
        
        # VQT arguments
        e_inf = energy_infer.BernoulliEnergyInference()
        e_inf.infer(e_model)
        q_inf = circuit_infer.QuantumInference()
        h_inf = hamiltonian_infer.QHBM(e_inf, q_inf)
        test_num_samples = tf.constant(1e7)
        test_h = tfq.convert_to_tensor(
            [cirq.PauliSum.from_pauli_strings(cirq.Y(q) for q in qubits)])
        test_beta = tf.random.uniform([], minval=0.01, maxval=100.0, seed=seed)

        # Compute losses
        # Bernoulli has only one tf.Variable
        test_thetas = e_model.trainable_variables[0]
        # QNN has only one tf.Variable
        test_phis = q_model.trainable_variables[0]
        actual_expectation = h_inf.expectation(h_model, test_h, test_num_samples)[0]
        expected_expectation = tf.reduce_sum(
            tf.math.tanh(test_thetas) * tf.math.sin(test_phis))
        self.assertAllClose(actual_expectation, expected_expectation, rtol=RTOL)

        actual_entropy = e_inf.entropy()
        expected_entropy = tf.reduce_sum(
            -test_thetas * tf.math.tanh(test_thetas) +
            tf.math.log(2 * tf.math.cosh(test_thetas)))
        self.assertAllClose(actual_entropy, expected_entropy, rtol=RTOL)

        with tf.GradientTape() as tape:
          actual_loss = vqt_func(h_inf, h_model, test_num_samples, test_h, test_beta)
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

  # def test_hypernetwork(self):
  #   for num_qubits in [1, 2, 3, 4, 5]:
  #     qubits = cirq.GridQubit.rect(1, num_qubits)
  #     e_model = energy_model.BernoulliEnergy(list(range(num_qubits)))
  #     e_model.build([None, num_qubits])
  #     unitary, phis_symbols = architectures.get_hardware_efficient_model_unitary(
  #         qubits, 3, "TestName")
  #     q_model = circuit_model.DirectQuantumCircuit(unitary)
  #     q_model.build([])
  #     h_model = hamiltonian_model.Hamiltonian(e_model, q_model)

  #     ham = test_util.get_random_pauli_sum(qubits)
  #     tf_ham = tfq.convert_to_tensor([ham])
  #     trainable_variables_shapes = [
  #         tf.shape(var) for var in h_model.trainable_variables
  #     ]
  #     trainable_variables_sizes = [
  #         tf.size(var) for var in h_model.trainable_variables
  #     ]
  #     trainable_variables_size = tf.reduce_sum(
  #         tf.stack(trainable_variables_sizes))

  #     input_size = 15
  #     hypernetwork = tf.keras.Sequential([
  #         tf.keras.layers.Dense(15, 'relu', input_shape=(input_size,)),
  #         tf.keras.layers.Dense(10, 'tanh', input_shape=(input_size,)),
  #         tf.keras.layers.Dense(5, 'sigmoid', input_shape=(input_size,)),
  #         tf.keras.layers.Dense(trainable_variables_size)
  #     ])
  #     input = tf.random.uniform([1, input_size])

  #     e_inf = energy_infer.BernoulliEnergyInference()
  #     q_inf = circuit_infer.QuantumInference()
  #     test_qhbm = hamiltonian_infer.QHBM(e_inf, q_inf)
  #     with tf.GradientTape() as tape:
  #       output = tf.squeeze(hypernetwork(input))
  #       index = 0
  #       output_trainable_variables = []
  #       for size, shape in zip(trainable_variables_sizes,
  #                              trainable_variables_shapes):
  #         output_trainable_variables.append(
  #             tf.reshape(output[index:index + size], shape))
  #         index += size
  #       test_qhbm.trainable_variables = output_trainable_variables
  #       loss = vqt.vqt(test_qhbm, h_model, tf.constant(int(5e6)), tf_ham,
  #                      tf.constant(1.0))
  #     grads = tape.gradient(loss, [
  #         hypernetwork.trainable_variables, output,
  #         model.trainable_variables
  #     ])
  #     hyper_grads = grads[0]
  #     output_grad = grads[1]
  #     qhbm_grads = grads[2]

  #     qhbm_grad_flat = []
  #     for grad in qhbm_grads:
  #       qhbm_grad_flat.append(tf.reshape(grad, [-1]))
  #     qhbm_grad_flat = tf.concat(qhbm_grad_flat, 0)
  #     self.assertAllEqual(qhbm_grad_flat, output_grad)

  #     for grad in hyper_grads:
  #       self.assertIsNotNone(grad)

  #     c = tf.Variable(tf.random.uniform([trainable_variables_size]))
  #     input = tf.random.uniform([trainable_variables_size])
  #     with tf.GradientTape() as tape:
  #       output = c * input
  #       index = 0
  #       output_trainable_variables = []
  #       for size, shape in zip(trainable_variables_sizes,
  #                              trainable_variables_shapes):
  #         output_trainable_variables.append(
  #             tf.reshape(output[index:index + size], shape))
  #         index += size
  #       test_qhbm.trainable_variables = output_trainable_variables
  #       loss = vqt.vqt(test_qhbm, tf.constant(int(5e6)), tf_ham,
  #                      tf.constant(1.0))
  #     grads = tape.gradient(loss, [c, test_qhbm.trainable_variables])
  #     c_grad = grads[0]
  #     qhbm_grads = grads[1]
  #     qhbm_grad_flat = []
  #     for grad in qhbm_grads:
  #       qhbm_grad_flat.append(tf.reshape(grad, [-1]))
  #     qhbm_grad_flat = tf.concat(qhbm_grad_flat, 0)
  #     self.assertAllEqual(input * qhbm_grad_flat, c_grad)


if __name__ == "__main__":
  print("Running vqt_new_test.py ...")
  tf.test.main()
