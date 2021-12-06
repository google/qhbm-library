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
"""Tests for the energy_model module."""

import itertools
import random

import cirq
import tensorflow as tf

from qhbmlib import energy_model


class BitstringDistributionTest(tf.test.TestCase):
  """Tests basic implementation of abstract class."""

  class Basic(energy_model.BitstringDistribution):
    """Test class."""

    @property
    def trainable_variables(self):
      return None

    @trainable_variables.setter
    def trainable_variables(self, value):
      pass

    def energy(self, bitstrings):
      """Simple uniform distribution."""
      return tf.tile(tf.constant([1.0]), [tf.shape(bitstrings)[0]])

  def test_init(self):
    """Test that components are initialized correctly."""
    expected_num_bits = 5
    expected_bits = [5, 17, 22, 30, 42]
    expected_name = "init_test"
    test_b = self.Basic(expected_bits, name=expected_name)
    self.assertEqual(test_b.num_bits, expected_num_bits)
    self.assertAllEqual(test_b.bits, expected_bits)
    self.assertEqual(test_b.name, expected_name)

  def test_init_error(self):
    """Confirms bad inputs are caught."""
    with self.assertRaisesRegex(TypeError, expected_regex="a list of integers"):
      _ = self.Basic(90)
    with self.assertRaisesRegex(TypeError, expected_regex="a list of integers"):
      _ = self.Basic(["junk"])
    with self.assertRaisesRegex(ValueError, expected_regex="must be unique"):
      _ = self.Basic([1, 1])

  def test_energy(self):
    """Confirms all bitstrings are accepted."""
    num_bits = 5
    test_b = self.Basic(list(range(num_bits)))
    test_bitstrings = tf.constant(
        list(itertools.product([0, 1], repeat=num_bits)))
    actual_energy = test_b(test_bitstrings)
    expected_energy = tf.constant([1.0] * (2**num_bits))
    self.assertAllEqual(actual_energy, expected_energy)
    actual_energy = test_b(test_bitstrings)


class MLPTest(tf.test.TestCase):
  """Test the MLP class."""

  def test_init_and_copy(self):
    """Test that components are initialized correctly."""
    expected_num_bits = 5
    expected_bits = [6, 17, 22, 30, 42]
    units = [2, 9]
    expected_units = [5, 2, 9, 1]
    activations = ["elu", "gelu"]
    expected_activations = [
        tf.keras.activations.elu, tf.keras.activations.gelu,
        tf.keras.activations.linear
    ]
    kernel_init_const = 1.5
    bias_init_const = 2.7
    expected_name = "init_test"
    test_mlp = energy_model.MLP(
        expected_bits,
        units,
        activations,
        kernel_initializer=tf.keras.initializers.Constant(kernel_init_const),
        bias_initializer=tf.keras.initializers.Constant(bias_init_const),
        name=expected_name)
    self.assertEqual(test_mlp.num_bits, expected_num_bits)
    self.assertAllEqual(test_mlp.bits, expected_bits)
    self.assertEqual(test_mlp.name, expected_name)

    # Note that for dense layers,
    # output = activation(dot(input, kernel) + bias),
    # which yields opposite shape from that expected for matmul implementation.
    for i in range(3):
      self.assertAllEqual(
          test_mlp.layers[i].kernel,
          tf.constant(kernel_init_const, shape=expected_units[i:i + 2]))
      self.assertAllEqual(
          test_mlp.layers[i].bias,
          tf.constant(bias_init_const, shape=expected_units[i + 1]))
      self.assertEqual(test_mlp.layers[i].activation, expected_activations[i])

    test_mlp_copy = test_mlp.copy()
    self.assertEqual(test_mlp_copy.num_bits, test_mlp.num_bits)
    self.assertAllEqual(test_mlp_copy.bits, test_mlp.bits)
    self.assertEqual(test_mlp_copy.name, test_mlp.name)
    for i in range(3):
      self.assertAllEqual(test_mlp_copy.layers[i].kernel,
                          test_mlp.layers[i].kernel)
      self.assertNotEqual(
          id(test_mlp_copy.layers[i].kernel), id(test_mlp.layers[i].kernel))
      self.assertAllEqual(test_mlp_copy.layers[i].bias, test_mlp.layers[i].bias)
      self.assertNotEqual(
          id(test_mlp_copy.layers[i].bias), id(test_mlp.layers[i].bias))
      self.assertEqual(test_mlp_copy.layers[i].activation,
                       test_mlp.layers[i].activation)

  def test_trainable_variables_mlp(self):
    test_mlp = energy_model.MLP(
        list(range(5)), [4, 3, 2], activations=['relu', 'tanh', 'sigmoid'])

    i = 0
    for layer in test_mlp.layers:
      self.assertAllEqual(layer.kernel, test_mlp.trainable_variables[i])
      self.assertAllEqual(layer.bias, test_mlp.trainable_variables[i + 1])
      i += 2

    variables = [
        tf.random.uniform(tf.shape(v)) for v in test_mlp.trainable_variables
    ]
    test_mlp.trainable_variables = variables
    for i in range(len(test_mlp.trainable_variables)):
      self.assertAllEqual(variables[i], test_mlp.trainable_variables[i])

    variables = [tf.Variable(v) for v in variables]
    test_mlp.trainable_variables = variables
    for i in range(len(test_mlp.trainable_variables)):
      self.assertAllEqual(variables[i], test_mlp.trainable_variables[i])

  def test_energy_mlp(self):
    """Tests MLP.energy."""
    num_bits = 9
    num_layers = 4
    test_bitstrings = tf.constant(
        list(itertools.product([0, 1], repeat=num_bits)))
    num_tests = 5
    for _ in range(num_tests):
      bits = random.sample(range(1000), num_bits)
      units = random.sample(range(1, 100), num_layers)
      activations = random.sample([
          "elu", "exponential", "gelu", "hard_sigmoid", "linear", "relu",
          "selu", "sigmoid", "softmax", "softplus", "sofsign", "swish", "tanh"
      ], num_layers)
      kernel_init = tf.keras.initializers.RandomUniform()
      bias_init = tf.keras.initializers.RandomUniform()
      actual_mlp = energy_model.MLP(bits, units, activations, kernel_init,
                                    bias_init)

      expected_layer_list = []
      for i in range(num_layers):
        expected_layer_list.append(
            tf.keras.layers.Dense(units[i], activation=activations[i]))
      expected_layer_list.append(tf.keras.layers.Dense(1))
      expected_mlp = tf.keras.Sequential(expected_layer_list)
      _ = expected_mlp(tf.constant([[0] * num_bits], tf.int8))
      for i, layer in enumerate(expected_mlp.layers):
        layer.kernel.assign(actual_mlp.layers[i].kernel)
        layer.bias.assign(actual_mlp.layers[i].bias)

      @tf.function
      def special_energy(bitstrings):
        return actual_mlp(bitstrings)

      for e_func in [actual_mlp, special_energy]:
        with tf.GradientTape(persistent=True) as tape:
          actual_energy = e_func(test_bitstrings)
          expected_energy = tf.squeeze(expected_mlp(test_bitstrings), -1)
        self.assertAllClose(actual_energy, expected_energy)
        actual_energy_grad = tape.jacobian(actual_energy,
                                           actual_mlp.trainable_variables)
        expected_energy_grad = tape.jacobian(expected_energy,
                                             expected_mlp.trainable_variables)
        del tape
        self.assertAllClose(actual_energy_grad, expected_energy_grad)


class BernoulliTest(tf.test.TestCase):
  """Test the Bernoulli class."""

  def test_init(self):
    """Test that components are initialized correctly."""
    expected_num_bits = 5
    expected_bits = [5, 17, 22, 30, 42]
    init_const = 1.5
    expected_name = "init_test"
    test_b = energy_model.Bernoulli(
        expected_bits,
        tf.keras.initializers.Constant(init_const),
        name=expected_name)
    self.assertEqual(test_b.num_bits, expected_num_bits)
    self.assertAllEqual(test_b.bits, expected_bits)
    self.assertAllEqual(test_b.kernel, [init_const] * expected_num_bits)
    self.assertEqual(test_b.name, expected_name)

  def test_copy(self):
    """Test that the copy has the same values, but new variables."""
    expected_num_bits = 8
    expected_bits = list(range(expected_num_bits))
    test_b = energy_model.Bernoulli(expected_bits, name="test_copy")
    test_b_copy = test_b.copy()
    self.assertEqual(test_b_copy.num_bits, test_b.num_bits)
    self.assertAllEqual(test_b_copy.bits, test_b.bits)
    self.assertAllEqual(test_b_copy.kernel, test_b.kernel)
    self.assertNotEqual(id(test_b_copy.kernel), id(test_b.kernel))
    self.assertEqual(test_b_copy.name, test_b.name)

  def test_trainable_variables_bernoulli(self):
    bits = [1, 3, 4, 8, 9]
    test_b = energy_model.Bernoulli(bits, name="test")
    self.assertAllEqual(test_b.kernel, test_b.trainable_variables[0])

    expected_kernel = tf.random.uniform([len(bits)])
    test_b.trainable_variables = [expected_kernel]
    self.assertAllEqual(test_b.trainable_variables[0], expected_kernel)

    expected_kernel = tf.Variable(tf.random.uniform([len(bits)]))
    test_b.trainable_variables = [expected_kernel]
    self.assertAllEqual(test_b.trainable_variables[0], expected_kernel)

  def test_energy_bernoulli_simple(self):
    """Test Bernoulli.energy and its derivative in a simple case.

    For a given bitstring b, the energy is
      $$E_\theta(b) = \sum_i (1-2b_i)\theta_i$$
    Then the derivative of the energy with respect to the thetas vector is
      $$\partial / \partial \theta E_\theta(b) = [(1-2b_i) for b_i in b]$$
    """
    test_b = energy_model.Bernoulli([1, 2, 3])
    test_vars = tf.constant([1.0, 1.7, -2.8], dtype=tf.float32)
    test_b.kernel.assign(test_vars)
    test_bitstrings = tf.constant([[0, 0, 0], [1, 0, 0], [0, 0, 1]])
    test_spins = 1 - 2 * test_bitstrings

    @tf.function
    def special_energy(bitstrings):
      return test_b(bitstrings)

    for e_func in [test_b, special_energy]:
      with tf.GradientTape() as tape:
        actual_energy = e_func(test_bitstrings)
      actual_energy_grad = tape.jacobian(actual_energy,
                                         test_b.trainable_variables)
      expected_energy = [
          test_vars[0] + test_vars[1] + test_vars[2],
          -test_vars[0] + test_vars[1] + test_vars[2],
          test_vars[0] + test_vars[1] - test_vars[2]
      ]
      self.assertAllClose(actual_energy, expected_energy)
      self.assertAllClose(actual_energy_grad, [test_spins])

  def test_energy_bernoulli(self):
    """Test Bernoulli.energy and its derivative in all cases.

    For a given bitstring b, the energy is
      $$E_\theta(b) = \sum_i (1-2b_i)\theta_i$$
    Then the derivative of the energy with respect to the thetas vector is
      $$\partial / \partial \theta E_\theta(b) = [(1-2b_i) for b_i in b]$$
    """
    num_bits = 9
    test_bitstrings = tf.constant(
        list(itertools.product([0, 1], repeat=num_bits)))
    test_spins = 1 - 2 * test_bitstrings
    num_tests = 5
    for _ in range(num_tests):
      bits = random.sample(range(1000), num_bits)
      thetas = tf.random.uniform([num_bits], minval=-100, maxval=100)
      test_b = energy_model.Bernoulli(bits)
      test_b.kernel.assign(thetas)

      @tf.function
      def special_energy(bitstrings):
        return test_b(bitstrings)

      for e_func in [test_b, special_energy]:
        with tf.GradientTape() as tape:
          actual_energy = e_func(test_bitstrings)

        expected_energy = tf.reduce_sum(
            tf.cast(test_spins, tf.float32) * thetas, -1)
        self.assertAllClose(actual_energy, expected_energy)

        actual_energy_grad = tape.jacobian(actual_energy,
                                           test_b.trainable_variables)
        self.assertAllClose(actual_energy_grad, [test_spins])

  def test_operator_shards(self):
    """Confirm operators are single qubit Z only."""
    num_bits = 10
    test_b = energy_model.Bernoulli(list(range(num_bits)))
    qubits = cirq.GridQubit.rect(1, num_bits)
    actual_ops = test_b.operator_shards(qubits)
    expected_ops = [cirq.PauliSum.from_pauli_strings(cirq.Z(q)) for q in qubits]
    self.assertAllEqual(actual_ops, expected_ops)

  def test_operator_expectation(self):
    """Test combining expectations of operators in energy."""
    # Build Bernoulli
    num_bits = 3
    test_b = energy_model.Bernoulli(list(range(num_bits)))
    qubits = cirq.GridQubit.rect(1, num_bits)
    # Pin at bitstring [1, 0, 1]
    test_b.kernel.assign(tf.constant([1000.0, -1000.0, 1000.0]))
    operators = test_b.operator_shards(qubits)

    # True energy
    bitstring = tf.constant([[0, 0, 1]])  # not the pinned bitstring
    ref_energy = test_b(bitstring)[0]

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
    test_energy = test_b.operator_expectation(op_expectations)
    self.assertAllClose(test_energy, ref_energy, atol=1e-4)


if __name__ == "__main__":
  print("Running energy_model_test.py ...")
  tf.test.main()
