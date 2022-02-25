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
"""Tests for qhbmlib.models.energy"""

import itertools
import random

import cirq
import tensorflow as tf

from qhbmlib import models
from qhbmlib import utils
from tests import test_util


class BitstringEnergyTest(tf.test.TestCase):
  """Tests the BitstringEnergy class."""

  def test_init(self):
    """Tests that components are initialized correctly."""
    expected_num_bits = 5
    expected_bits = [5, 17, 22, 30, 42]
    expected_name = "init_test"
    expected_energy_layers = [tf.keras.layers.Dense(1), utils.Squeeze(-1)]
    actual_b = models.BitstringEnergy(
        expected_bits, expected_energy_layers, name=expected_name)
    self.assertEqual(actual_b.num_bits, expected_num_bits)
    self.assertAllEqual(actual_b.bits, expected_bits)
    self.assertAllEqual(actual_b.energy_layers, expected_energy_layers)
    self.assertEqual(actual_b.name, expected_name)

  @test_util.eager_mode_toggle
  def test_call(self):
    """Checks that building and calling works for a simple energy."""
    num_bits = 5
    energy_layers = [
        tf.keras.layers.Dense(
            1, kernel_initializer=tf.keras.initializers.Constant(1)),
        utils.Squeeze(-1)
    ]
    test_b = models.BitstringEnergy(list(range(num_bits)), energy_layers)

    @tf.function
    def test_b_wrapper(bitstrings):
      return test_b(bitstrings)

    test_bitstrings = tf.constant(
        list(itertools.product([0, 1], repeat=num_bits)))
    actual_energy = test_b_wrapper(test_bitstrings)
    expected_energy = tf.reduce_sum(test_bitstrings, -1)
    self.assertAllEqual(actual_energy, expected_energy)

  @test_util.eager_mode_toggle
  def test_energy_mlp(self):
    """Tests energy and derivatives for an MLP."""
    num_bits = 7
    num_layers = 4
    test_bitstrings = tf.constant(
        list(itertools.product([0, 1], repeat=num_bits)))
    num_tests = 3

    @tf.function
    def layer_wrapper(layer, bitstrings):
      return layer(bitstrings)

    for _ in range(num_tests):
      bits = random.sample(range(1000), num_bits)
      units = random.sample(range(1, 100), num_layers)
      activations = random.sample([
          "elu", "exponential", "gelu", "hard_sigmoid", "linear", "relu",
          "selu", "sigmoid", "softmax", "softplus", "softsign", "swish", "tanh"
      ], num_layers)
      expected_layer_list = []
      for i in range(num_layers):
        expected_layer_list.append(
            tf.keras.layers.Dense(units[i], activation=activations[i]))
      expected_layer_list.append(tf.keras.layers.Dense(1))
      expected_layer_list.append(utils.Squeeze(-1))
      actual_mlp = models.BitstringEnergy(bits, expected_layer_list)

      expected_mlp = tf.keras.Sequential(expected_layer_list)
      expected_mlp.build([None, num_bits])
      expected_mlp.set_weights(actual_mlp.get_weights())

      with tf.GradientTape(persistent=True) as tape:
        actual_energy = layer_wrapper(actual_mlp, test_bitstrings)
        expected_energy = expected_mlp(test_bitstrings)
      self.assertAllClose(actual_energy, expected_energy)
      actual_energy_grad = tape.jacobian(actual_energy,
                                         actual_mlp.trainable_variables)
      expected_energy_grad = tape.jacobian(expected_energy,
                                           expected_mlp.trainable_variables)
      del tape
      self.assertAllClose(actual_energy_grad, expected_energy_grad)


class BernoulliEnergyTest(tf.test.TestCase):
  """Tests the BernoulliEnergy class."""

  @test_util.eager_mode_toggle
  def test_energy_simple(self):
    r"""Tests the energy and its derivative in a simple case.

    For a given bitstring b, the energy is
      $$E_\theta(b) = \sum_i (1-2b_i)\theta_i$$
    Then the derivative of the energy with respect to the thetas vector is
      $$\partial / \partial \theta E_\theta(b) = [(1-2b_i) for b_i in b]$$
    """
    test_b = models.BernoulliEnergy([1, 2, 3])
    test_vars = tf.constant([1.0, 1.7, -2.8], dtype=tf.float32)
    test_b.build([None, 3])
    test_b.set_weights([test_vars])
    test_bitstrings = tf.constant([[0, 0, 0], [1, 0, 0], [0, 1, 1]])
    test_spins = 1 - 2 * test_bitstrings
    expected_logits = 2 * test_vars
    actual_logits = test_b.logits
    self.assertAllClose(actual_logits, expected_logits)

    @tf.function
    def test_b_wrapper(bitstrings):
      return test_b(bitstrings)

    with tf.GradientTape() as tape:
      actual_energy = test_b_wrapper(test_bitstrings)
    actual_energy_grad = tape.jacobian(actual_energy,
                                       test_b.trainable_variables)
    expected_energy = [
        test_vars[0] + test_vars[1] + test_vars[2],
        -test_vars[0] + test_vars[1] + test_vars[2],
        test_vars[0] - test_vars[1] - test_vars[2]
    ]
    self.assertAllClose(actual_energy, expected_energy)
    self.assertAllClose(actual_energy_grad, [test_spins])

  @test_util.eager_mode_toggle
  def test_energy_bernoulli(self):
    r"""Tests the energy and its derivative in a more complicated case.

    For a given bitstring b, the energy is
      $$E_\theta(b) = \sum_i (1-2b_i)\theta_i$$
    Then the derivative of the energy with respect to the thetas vector is
      $$\partial / \partial \theta E_\theta(b) = [(1-2b_i) for b_i in b]$$
    """
    num_bits = 7
    test_bitstrings = tf.constant(
        list(itertools.product([0, 1], repeat=num_bits)))
    test_spins = 1 - 2 * test_bitstrings
    num_tests = 5

    @tf.function
    def layer_wrapper(layer, bitstrings):
      return layer(bitstrings)

    for _ in range(num_tests):
      bits = random.sample(range(1000), num_bits)
      thetas = tf.random.uniform([num_bits], -100, 100, tf.float32)
      test_b = models.BernoulliEnergy(bits)
      test_b.build([None, num_bits])
      test_b.set_weights([thetas])

      with tf.GradientTape() as tape:
        actual_energy = layer_wrapper(test_b, test_bitstrings)

      expected_energy = tf.reduce_sum(
          tf.cast(test_spins, tf.float32) * thetas, -1)
      self.assertAllClose(actual_energy, expected_energy)

      actual_energy_grad = tape.jacobian(actual_energy,
                                         test_b.trainable_variables)
      self.assertAllClose(actual_energy_grad, [test_spins])

  def test_operator_shards(self):
    """Confirms operators are single qubit Z only."""
    num_bits = 10
    test_b = models.BernoulliEnergy(list(range(num_bits)))
    qubits = cirq.GridQubit.rect(1, num_bits)
    actual_ops = test_b.operator_shards(qubits)
    expected_ops = [cirq.PauliSum.from_pauli_strings(cirq.Z(q)) for q in qubits]
    self.assertAllEqual(actual_ops, expected_ops)

  @test_util.eager_mode_toggle
  def test_operator_expectation(self):
    """Tests combining expectations of operators in energy."""
    # Build Bernoulli
    num_bits = 3
    test_b = models.BernoulliEnergy(list(range(num_bits)))

    @tf.function
    def operator_expectation_wrapper(sub_expectations):
      return test_b.operator_expectation(sub_expectations)

    qubits = cirq.GridQubit.rect(1, num_bits)
    # Pin at bitstring [1, 0, 1]
    test_b.build([None, num_bits])
    test_b.set_weights([tf.constant([1000.0, -1000.0, 1000.0])])
    operators = test_b.operator_shards(qubits)

    # True energy
    bitstring = tf.constant([[0, 0, 1]])  # not the pinned bitstring
    expected_energy = test_b(bitstring)[0]

    # Test energy
    circuit = cirq.Circuit(
        [cirq.I(qubits[0]),
         cirq.I(qubits[1]),
         cirq.X(qubits[2])])
    output_state_vector = cirq.Simulator().simulate(circuit).final_state_vector
    op_expectations = []
    qubit_map = {q: i for i, q in enumerate(qubits)}
    for op in operators:
      op_expectations.append(
          op.expectation_from_state_vector(output_state_vector, qubit_map).real)
    actual_energy = operator_expectation_wrapper(op_expectations)
    self.assertAllClose(actual_energy, expected_energy, atol=1e-4)


class KOBETest(tf.test.TestCase):
  """Tests the KOBE class."""

  @test_util.eager_mode_toggle
  def test_energy(self):
    """Tests every energy on two bits."""
    bits = [0, 1]
    order = 2
    test_thetas = tf.constant([1.5, 2.7, -4.0])
    expected_energies = tf.constant([0.2, 2.8, 5.2, -8.2])
    test_k = models.KOBE(bits, order)

    @tf.function
    def test_k_wrapper(bitstrings):
      return test_k(bitstrings)

    all_strings = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]])
    test_k.build([None, 2])
    test_k.set_weights([test_thetas])
    actual_energies = test_k_wrapper(all_strings)
    self.assertAllClose(actual_energies, expected_energies)

  def test_operator_shards(self):
    """Confirms correct operators for a simple Boltzmann."""
    num_bits = 3
    test_k = models.KOBE(list(range(num_bits)), 2)
    qubits = cirq.GridQubit.rect(1, num_bits)
    test_ops = test_k.operator_shards(qubits)
    ref_ops = [
        cirq.Z(qubits[0]),
        cirq.Z(qubits[1]),
        cirq.Z(qubits[2]),
        cirq.Z(qubits[0]) * cirq.Z(qubits[1]),
        cirq.Z(qubits[0]) * cirq.Z(qubits[2]),
        cirq.Z(qubits[1]) * cirq.Z(qubits[2])
    ]
    for actual_op, expected_op in zip(test_ops, ref_ops):
      self.assertEqual(actual_op, cirq.PauliSum.from_pauli_strings(expected_op))

  @test_util.eager_mode_toggle
  def test_operator_expectation(self):
    """Confirms the expectations combine to the correct total energy."""
    # Build simple Boltzmann
    num_bits = 3
    test_b = models.KOBE(list(range(num_bits)), 2)

    @tf.function
    def operator_expectation_wrapper(sub_expectations):
      return test_b.operator_expectation(sub_expectations)

    qubits = cirq.GridQubit.rect(1, num_bits)

    # Pin at bitstring [1, 0, 1]
    test_b.build([None, num_bits])
    test_b.set_weights([tf.constant([100.0, -200.0, 300.0, 10, -20, 30])])
    operators = test_b.operator_shards(qubits)

    # True energy
    bitstring = tf.constant([[0, 0, 1]])  # not the pinned bitstring
    expected_energy = test_b(bitstring)[0]

    # Test energy
    circuit = cirq.Circuit(
        [cirq.I(qubits[0]),
         cirq.I(qubits[1]),
         cirq.X(qubits[2])])
    output_state_vector = cirq.Simulator().simulate(circuit).final_state_vector
    op_expectations = []
    qubit_map = {q: i for i, q in enumerate(qubits)}
    for op in operators:
      op_expectations.append(
          op.expectation_from_state_vector(output_state_vector, qubit_map).real)
    actual_energy = operator_expectation_wrapper(op_expectations)
    self.assertAllClose(actual_energy, expected_energy, atol=1e-4)


if __name__ == "__main__":
  print("Running energy_test.py ...")
  tf.test.main()
