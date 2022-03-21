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
"""Tests for tests.test_util"""

from absl.testing import parameterized

import cirq
import numpy as np
import sympy
import tensorflow as tf

from tests import test_util


class RPQCTest(tf.test.TestCase, parameterized.TestCase):
  """Test RPQC functions in the test_util module."""

  def test_get_xz_rotation(self):
    """Confirm an XZ rotation is returned."""
    q = cirq.GridQubit(7, 9)
    a, b = sympy.symbols("a b")
    expected_circuit = cirq.Circuit(cirq.X(q)**a, cirq.Z(q)**b)
    actual_circuit = test_util.get_xz_rotation(q, a, b)
    self.assertEqual(actual_circuit, expected_circuit)

  def test_get_cz_exp(self):
    """Confirm an exponentiated CNOT is returned."""
    q0 = cirq.GridQubit(4, 1)
    q1 = cirq.GridQubit(2, 5)
    a = sympy.Symbol("a")
    expected_circuit = cirq.Circuit(cirq.CZ(q0, q1)**a)
    actual_circuit = test_util.get_cz_exp(q0, q1, a)
    self.assertEqual(actual_circuit, expected_circuit)

  def test_get_xz_rotation_layer(self):
    """Confirm an XZ rotation on every qubit is returned."""
    qubits = cirq.GridQubit.rect(1, 2)
    layer_num = 3
    name = "test_rot"
    expected_circuit = cirq.Circuit()
    for n, q in enumerate(qubits):
      s = sympy.Symbol("sx_{0}_{1}_{2}".format(name, layer_num, n))
      expected_circuit += cirq.Circuit(cirq.X(q)**s)
      s = sympy.Symbol("sz_{0}_{1}_{2}".format(name, layer_num, n))
      expected_circuit += cirq.Circuit(cirq.Z(q)**s)
    actual_circuit = test_util.get_xz_rotation_layer(qubits, layer_num, name)
    self.assertEqual(actual_circuit, expected_circuit)

  @parameterized.parameters([{"n_qubits": 11}, {"n_qubits": 12}])
  def test_get_cz_exp_layer(self, n_qubits):
    """Confirm an exponentiated CZ on every qubit is returned."""
    qubits = cirq.GridQubit.rect(1, n_qubits)
    layer_num = 0
    name = "test_cz"
    expected_circuit = cirq.Circuit()
    for n, (q0, q1) in enumerate(zip(qubits, qubits[1:])):
      if n % 2 == 0:
        s = sympy.Symbol("sc_{0}_{1}_{2}".format(name, layer_num, n))
        expected_circuit += cirq.Circuit(cirq.CZ(q0, q1)**s)
    for n, (q0, q1) in enumerate(zip(qubits, qubits[1:])):
      if n % 2 == 1:
        s = sympy.Symbol("sc_{0}_{1}_{2}".format(name, layer_num, n))
        expected_circuit += cirq.Circuit(cirq.CZ(q0, q1)**s)
    actual_circuit = test_util.get_cz_exp_layer(qubits, layer_num, name)
    self.assertEqual(actual_circuit, expected_circuit)

  @parameterized.parameters([{"n_qubits": 11}, {"n_qubits": 12}])
  def test_get_hardware_efficient_model_unitary(self, n_qubits):
    """Confirm a multi-layered circuit is returned."""
    qubits = cirq.GridQubit.rect(1, n_qubits)
    name = "test_hardware_efficient_model"
    expected_circuit = cirq.Circuit()
    this_circuit = test_util.get_xz_rotation_layer(qubits, 0, name)
    expected_circuit += this_circuit
    this_circuit = test_util.get_cz_exp_layer(qubits, 0, name)
    expected_circuit += this_circuit
    this_circuit = test_util.get_xz_rotation_layer(qubits, 1, name)
    expected_circuit += this_circuit
    this_circuit = test_util.get_cz_exp_layer(qubits, 1, name)
    expected_circuit += this_circuit
    actual_circuit = test_util.get_hardware_efficient_model_unitary(
        qubits, 2, name)
    self.assertEqual(actual_circuit, expected_circuit)


class EagerModeToggleTest(tf.test.TestCase):
  """Tests eager_mode_toggle."""

  def test_eager_mode_toggle(self):
    """Ensure eager mode really gets toggled."""

    def fail_in_eager():
      """Raises AssertionError if run in eager."""
      if tf.config.functions_run_eagerly():
        raise AssertionError()

    def fail_out_of_eager():
      """Raises AssertionError if run outside of eager."""
      if not tf.config.functions_run_eagerly():
        raise AssertionError()

    with self.assertRaises(AssertionError):
      test_util.eager_mode_toggle(fail_in_eager)()

    # Ensure eager mode still turned off even though exception was raised.
    self.assertFalse(tf.config.functions_run_eagerly())

    with self.assertRaises(AssertionError):
      test_util.eager_mode_toggle(fail_out_of_eager)()


class PerturbFunctionTest(tf.test.TestCase, parameterized.TestCase):
  """Tests perturb_function."""

  @test_util.eager_mode_toggle
  def test_side_effects(self):
    """Checks that variable is perturbed and then returned to initial value."""
    initial_value = tf.constant([4.5, -1.3])
    basic_variable = tf.Variable(initial_value)

    def f():
      """Basic test function."""
      return basic_variable.read_value()

    test_delta = 0.5
    wrapped_perturb_function = tf.function(test_util.perturb_function)
    actual_return = wrapped_perturb_function(f, basic_variable, 1, test_delta)
    expected_return = initial_value + [0, test_delta]
    self.assertIsInstance(actual_return, tf.Tensor)
    self.assertAllClose(actual_return, expected_return)
    self.assertAllClose(basic_variable, initial_value)

  @parameterized.parameters([{
      "this_type": t
  } for t in [tf.float16, tf.float32, tf.float64, tf.complex64, tf.complex128]])
  def test_multi_variable(self, this_type):
    """Tests perturbation when there are multiple differently shaped vars."""
    dimension = 7
    minval = -5
    maxval = 5
    scalar_initial_value = tf.cast(
        tf.random.uniform([], minval, maxval), this_type)
    scalar_var = tf.Variable(scalar_initial_value)
    vector_initial_value = tf.cast(
        tf.random.uniform([dimension], minval, maxval), this_type)
    vector_var = tf.Variable(vector_initial_value)
    matrix_initial_value = tf.cast(
        tf.random.uniform([dimension, dimension], minval, maxval), this_type)
    matrix_var = tf.Variable(matrix_initial_value)

    def f():
      """Vector result of combining the variables."""
      val = tf.linalg.matvec(matrix_var, vector_var) * scalar_var
      return [val, [val, val]]

    test_delta_raw = tf.random.uniform([], dtype=tf.float32)
    test_delta_python = test_delta_raw.numpy().tolist()
    test_delta = tf.cast(test_delta_raw, this_type)
    wrapped_perturb_function = tf.function(test_util.perturb_function)

    # check scalar perturbation
    perturbed_scalar = scalar_var + test_delta
    expected_val = tf.linalg.matvec(matrix_var, vector_var) * perturbed_scalar
    expected_return = [expected_val, [expected_val, expected_val]]
    actual_return = wrapped_perturb_function(f, scalar_var, 0, test_delta)
    tf.nest.map_structure(lambda x: self.assertIsInstance(x, tf.Tensor),
                          actual_return)
    tf.nest.map_structure(self.assertAllClose, actual_return, expected_return)
    self.assertAllClose(scalar_var, scalar_initial_value)

    # check vector perturbations
    for i in range(dimension):
      vector_list = vector_initial_value.numpy().tolist()
      perturbation_vector = [
          test_delta_python if j == i else 0 for j in range(dimension)
      ]
      perturbed_vector_list = [
          v + v_p for v, v_p in zip(vector_list, perturbation_vector)
      ]
      perturbed_vector = tf.constant(perturbed_vector_list, this_type)
      expected_val = tf.linalg.matvec(matrix_var, perturbed_vector) * scalar_var
      expected_return = [expected_val, [expected_val, expected_val]]
      actual_return = wrapped_perturb_function(f, vector_var, i, test_delta)
      tf.nest.map_structure(lambda x: self.assertIsInstance(x, tf.Tensor),
                            actual_return)
      tf.nest.map_structure(self.assertAllClose, actual_return, expected_return)
      self.assertAllClose(vector_var, vector_initial_value)

    # check matrix perturbations
    for i in range(dimension * dimension):
      matrix_list = tf.reshape(matrix_initial_value,
                               [dimension * dimension]).numpy().tolist()
      perturbation_matrix = [
          test_delta_python if j == i else 0
          for j in range(dimension * dimension)
      ]
      perturbed_matrix_list = [
          m + m_p for m, m_p in zip(matrix_list, perturbation_matrix)
      ]
      perturbed_matrix = tf.reshape(
          tf.constant(perturbed_matrix_list, this_type), [dimension, dimension])
      expected_val = tf.linalg.matvec(perturbed_matrix, vector_var) * scalar_var
      expected_return = [expected_val, [expected_val, expected_val]]
      actual_return = wrapped_perturb_function(f, matrix_var, i, test_delta)
      tf.nest.map_structure(lambda x: self.assertIsInstance(x, tf.Tensor),
                            actual_return)
      tf.nest.map_structure(self.assertAllClose, actual_return, expected_return)
      self.assertAllClose(matrix_var, matrix_initial_value)


class ApproximateDerivativesTest(tf.test.TestCase):
  """Tests approximate_gradient and approximate_jacobian functions."""

  def setUp(self):
    super().setUp()
    self.close_atol = 1e-4
    self.not_zero_atol = 2e-4
    dimension_0 = 7
    dimension_1 = 4
    minval = -5
    maxval = 5
    self.scalar_shape = []
    self.scalar_initial_value = tf.random.uniform(self.scalar_shape, minval, maxval)
    self.scalar_var = tf.Variable(self.scalar_initial_value)
    self.vector_shape = [dimension_0]
    self.vector_initial_value = tf.random.uniform(self.vector_shape, minval, maxval)
    self.vector_var = tf.Variable(self.vector_initial_value)
    self.matrix_shape = [dimension_1, dimension_0]
    self.matrix_initial_value = tf.random.uniform(
        self.matrix_shape, minval, maxval)
    self.matrix_var = tf.Variable(self.matrix_initial_value)
    self.variables_list = [self.scalar_var, self.vector_var, self.matrix_var]

    def linear_scalar():
      """Returns a scalar."""
      return self.scalar_var.read_value()
    self.linear_scalar = linear_scalar

    def linear_vector():
      """Returns a vector."""
      return self.vector_var.read_value()
    self.linear_vector = linear_vector

    def linear_matrix():
      """Returns a matrix."""
      return self.matrix_var.read_value()
    self.linear_matrix = linear_matrix

  def test_linear_gradient(self):
    """Confirms correct Gradient values for linear functions."""

    # scalar
    expected_gradient = [tf.ones_like(self.scalar_var), tf.zeros_like(self.vector_var), tf.zeros_like(self.matrix_var)]
    actual_gradient = test_util.approximate_gradient(self.linear_scalar, self.variables_list)
    for a, e in zip(actual_gradient, expected_gradient):
      self.assertAllClose(a, e, atol=self.close_atol)

    # vector
    expected_gradient = [tf.zeros_like(self.scalar_var), tf.ones_like(self.vector_var), tf.zeros_like(self.matrix_var)]
    actual_gradient = test_util.approximate_gradient(self.linear_vector, self.variables_list)
    for a, e in zip(actual_gradient, expected_gradient):
      self.assertAllClose(a, e, atol=self.close_atol)

    # matrix
    expected_gradient = [tf.zeros_like(self.scalar_var), tf.zeros_like(self.vector_var), tf.ones_like(self.matrix_var)]
    actual_gradient = test_util.approximate_gradient(self.linear_matrix, self.variables_list)
    for a, e in zip(actual_gradient, expected_gradient):
      self.assertAllClose(a, e, atol=self.close_atol)

  def test_linear_jacobian(self):
    """Confirms correct Jacobian values for linear functions."""

    # scalar
    scalar_jacobian = tf.constant(1.0)
    expected_jacobian = [scalar_jacobian, tf.zeros(self.vector_shape), tf.zeros(self.matrix_shape)]
    actual_jacobian = test_util.approximate_jacobian(self.linear_scalar, self.variables_list)
    for a, e in zip(actual_jacobian, expected_jacobian):
      self.assertAllClose(a, e, atol=self.close_atol)

    # vector
    vector_jacobian = np.zeros(self.vector_shape + self.vector_shape)
    for i in range(self.vector_shape[0]):
      for j in range(self.vector_shape[0]):
        if i == j:
          vector_jacobian[i, j] = 1.0
    vector_jacobian = tf.constant(vector_jacobian)
    expected_jacobian = [tf.zeros(self.vector_shape), vector_jacobian, tf.zeros(self.vector_shape + self.matrix_shape)]
    actual_jacobian = test_util.approximate_jacobian(self.linear_vector, self.variables_list)
    for a, e in zip(actual_jacobian, expected_jacobian):
      self.assertAllClose(a, e, atol=self.close_atol)

    # matrix
    matrix_jacobian = np.zeros(self.matrix_shape + self.matrix_shape)
    for i in range(self.matrix_shape[0]):
      for j in range(self.matrix_shape[1]):
        for k in range(self.matrix_shape[0]):
          for l in range(self.matrix_shape[1]):
            if i == k and j == l:
              matrix_jacobian[i, j, k, l] = 1.0
    matrix_jacobian = tf.constant(matrix_jacobian)
    expected_jacobian = [tf.zeros(self.matrix_shape), tf.zeros(self.matrix_shape + self.vector_shape), matrix_jacobian]
    actual_jacobian = test_util.approximate_jacobian(self.linear_matrix, self.variables_list)
    for a, e in zip(actual_jacobian, expected_jacobian):
      self.assertAllClose(a, e, atol=self.close_atol)

  # def test_gradient(self):
  #   """Checks gradient of nested return function."""

  # def test_jacobian(self):
  #   """Compares approximation against exact jacobian."""

  #   # First, test with a linear function of the matrix variable.
  #   def f_linear():
  #     """Just the variable itself."""
  #     return matrix_var.read_value()
  #   actual_derivative = test_util.approximate_jacobian(f_linear, [matrix_var, vector_var])
  #   expected_derivative_shapes = [[dimension_1, dimension_0, dimension_1, dimension_0], [dimension_1, dimension_0, dimension_0]]
  #   self.assertEqual(len(actual_derivative), len(expected_derivative_shapes))
  #   for a, e_shape in zip(actual_derivative, expected_derivative_shapes):
  #     self.assertAllEqual(tf.shape(a), e_shape)
  #   # The derivative of `f_linear()` should be 1 at [i, j, i, j].
  #   expected_matrix_derivative = tf.one_(matrix_var)
  #   expected_vector_derivative = tf.zeros_like(vector_var)
  #   self.assertAllClose(actual_derivative[0], expected_matrix_derivative, rtol=self.close_rtol)
  #   self.assertAllClose(actual_derivative[1], expected_vector_derivative, atol=self.zero_atol)
    
  #   def f_scalar():
  #     """Scalar result of combining the variables."""
  #     return tf.reduce_sum(tf.linalg.matvec(matrix_var, vector_var) * scalar_var)

  #   def f_vector():
  #     """Vector result of combining the variables."""
  #     return tf.linalg.matvec(matrix_var, vector_var) * scalar_var

  #   def f_matrix():
  #     """Matrix result of combining the variables."""
  #     new_vec = vector_var * scalar_var
  #     tiled_vec = tf.tile(tf.expand_dims(new_vec, 1), [1, dimension_1])
  #     return tf.linalg.matmul(matrix_var, tiled_vec)

  #   for f in [f_scalar, f_vector, f_matrix]:
  #     with tf.GradientTape() as tape:
  #       value = f()
  #     expected_derivative = tape.jacobian(value, variable_list)
  #     actual_derivative = test_util.approximate_jacobian(f, variable_list)
  #     self.assertEqual(len(actual_derivative), len(variable_list))
  #     for a, e in zip(actual_derivative, expected_derivative):
  #       print(f"tf.shape(a): {tf.shape(a)}")
  #       print(f"tf.shape(e): {tf.shape(e)}")
  #       self.assertAllClose(a, e, rtol=self.close_rtol)
