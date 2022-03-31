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
import tensorflow_probability as tfp

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
      s = sympy.Symbol(f"sx_{name}_{layer_num}_{n}")
      expected_circuit += cirq.Circuit(cirq.X(q)**s)
      s = sympy.Symbol(f"sz_{name}_{layer_num}_{n}")
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
        s = sympy.Symbol(f"sc_{name}_{layer_num}_{n}")
        expected_circuit += cirq.Circuit(cirq.CZ(q0, q1)**s)
    for n, (q0, q1) in enumerate(zip(qubits, qubits[1:])):
      if n % 2 == 1:
        s = sympy.Symbol(f"sc_{name}_{layer_num}_{n}")
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


class RandomMatrixTest(tf.test.TestCase):
  """Tests the random matrix functions for expected properties."""

  def setUp(self):
    """Initialize test objects."""
    super().setUp()
    self.num_qubits_list = [1, 2, 3, 4, 5]

  def test_random_hermitian_matrix(self):
    """Checks that returned matrix is Hermitian.

    Hermitian matrices are equal to their complex conjugates.  See Wikipedia,
    https://en.wikipedia.org/wiki/Hermitian_matrix
    """
    for n in self.num_qubits_list:
      actual_matrix = test_util.random_hermitian_matrix(n)
      expected_dim = 2**n
      self.assertAllEqual(tf.shape(actual_matrix), [expected_dim, expected_dim])
      self.assertAllEqual(actual_matrix, tf.linalg.adjoint(actual_matrix))

  def test_random_unitary_matrix(self):
    """Checks that returned matrix is unitary.

    Unitary matrices are their own inverse.  See Wikipedia,
    https://en.wikipedia.org/wiki/Unitary_matrix
    """
    for n in self.num_qubits_list:
      actual_matrix = test_util.random_unitary_matrix(n)
      expected_dim = 2**n
      self.assertAllEqual(tf.shape(actual_matrix), [expected_dim, expected_dim])
      udu = tf.linalg.matmul(tf.linalg.adjoint(actual_matrix), actual_matrix)
      uud = tf.linalg.matmul(actual_matrix, tf.linalg.adjoint(actual_matrix))
      self.assertAllClose(udu, tf.eye(expected_dim))
      self.assertAllClose(uud, tf.eye(expected_dim))

  def test_random_mixed_density_operator(self):
    """Checks return is a density matrix, and its eigenvalues match probs return

    Density matrices are positive-semidefinite operators with trace one.
    See Wikipedia, https://en.wikipedia.org/wiki/Density_matrix
    """
    for n in self.num_qubits_list:
      expected_dim = 2**n
      mixture_list = list(range(1, expected_dim + 1))
      for m in mixture_list:
        actual_matrix, actual_probs = test_util.random_mixed_density_matrix(
            n, m)
        self.assertAllEqual(
            tf.shape(actual_matrix), [expected_dim, expected_dim])
        self.assertAllEqual(tf.shape(actual_probs), [m])
        self.assertAllEqual(actual_matrix, tf.linalg.adjoint(actual_matrix))
        # Eigenvalues are real since we confirmed self-adjointness
        eigvals = tf.cast(tf.linalg.eigvalsh(actual_matrix), tf.float32)
        eigval_tol = 1e-7
        self.assertAllGreaterEqual(eigvals, -eigval_tol)
        self.assertAllClose(tf.linalg.trace(actual_matrix), 1.0)
        # Check the non-zero eigenvalues
        actual_sorted_probs = tf.sort(actual_probs, 0, "DESCENDING")
        expected_sorted_probs = tf.sort(eigvals, 0, "DESCENDING")[:m]
        self.assertAllClose(tf.reduce_sum(expected_sorted_probs), 1.0)
        self.assertAllClose(actual_sorted_probs, expected_sorted_probs)


class EntropyTest(tf.test.TestCase):
  """Tests the entropy function."""

  @test_util.eager_mode_toggle
  def test_exact_entropy(self):
    """Compares function against analytic entropy."""
    num_probs = 10
    num_zeros = 3
    pre_probs = tf.concat([
        tf.random.uniform([num_probs], dtype=tf.float32),
        tf.zeros([num_zeros])
    ], 0)
    probs = pre_probs / tf.math.reduce_sum(pre_probs)
    expected_entropy = tfp.distributions.Categorical(probs=probs).entropy()

    entropy_wrapper = tf.function(test_util.entropy)
    actual_entropy = entropy_wrapper(probs)
    self.assertAllClose(actual_entropy, expected_entropy)


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
    self.scalar_initial_value = tf.random.uniform(self.scalar_shape, minval,
                                                  maxval)
    self.scalar_var = tf.Variable(self.scalar_initial_value)
    self.vector_shape = [dimension_0]
    self.vector_initial_value = tf.random.uniform(self.vector_shape, minval,
                                                  maxval)
    self.vector_var = tf.Variable(self.vector_initial_value)
    self.matrix_shape = [dimension_1, dimension_0]
    self.matrix_initial_value = tf.random.uniform(self.matrix_shape, minval,
                                                  maxval)
    self.matrix_var = tf.Variable(self.matrix_initial_value)
    self.variables_structure = [[self.scalar_var, self.vector_var],
                                [self.matrix_var]]

    def linear_scalar():
      """Returns a scalar."""
      return tf.identity(self.scalar_var)

    self.linear_scalar = linear_scalar

    def linear_vector():
      """Returns a vector."""
      return tf.identity(self.vector_var)

    self.linear_vector = linear_vector

    def linear_matrix():
      """Returns a matrix."""
      return tf.identity(self.matrix_var)

    self.linear_matrix = linear_matrix

    def f_tensor():
      """Returns a combination of all three variables."""
      return tf.linalg.matvec(self.matrix_var,
                              self.vector_var) * self.scalar_var

    self.f_tensor = f_tensor

  @test_util.eager_mode_toggle
  def test_linear_gradient(self):
    """Confirms correct Gradient values for linear functions."""

    approximate_gradient_wrapper = tf.function(test_util.approximate_gradient)

    # scalar
    expected_gradient = [[
        tf.ones_like(self.scalar_var),
        tf.zeros_like(self.vector_var)
    ], [tf.zeros_like(self.matrix_var)]]
    actual_gradient = approximate_gradient_wrapper(self.linear_scalar,
                                                   self.variables_structure)
    tf.nest.map_structure(
        lambda a, e: self.assertAllClose(a, e, atol=self.close_atol),
        actual_gradient, expected_gradient)

    # vector
    expected_gradient = [[
        tf.zeros_like(self.scalar_var),
        tf.ones_like(self.vector_var)
    ], [tf.zeros_like(self.matrix_var)]]
    actual_gradient = approximate_gradient_wrapper(self.linear_vector,
                                                   self.variables_structure)
    tf.nest.map_structure(
        lambda a, e: self.assertAllClose(a, e, atol=self.close_atol),
        actual_gradient, expected_gradient)

    # matrix
    expected_gradient = [[
        tf.zeros_like(self.scalar_var),
        tf.zeros_like(self.vector_var)
    ], [tf.ones_like(self.matrix_var)]]
    actual_gradient = approximate_gradient_wrapper(self.linear_matrix,
                                                   self.variables_structure)
    tf.nest.map_structure(
        lambda a, e: self.assertAllClose(a, e, atol=self.close_atol),
        actual_gradient, expected_gradient)

  @test_util.eager_mode_toggle
  def test_linear_jacobian(self):
    """Confirms correct Jacobian values for linear functions."""

    # scalar
    scalar_jacobian = tf.constant(1.0)
    expected_jacobian = [[scalar_jacobian,
                          tf.zeros(self.vector_shape)],
                         [tf.zeros(self.matrix_shape)]]
    actual_jacobian = test_util.approximate_jacobian(self.linear_scalar,
                                                     self.variables_structure)
    tf.nest.map_structure(
        lambda a, e: self.assertAllClose(a, e, atol=self.close_atol),
        actual_jacobian, expected_jacobian)

    # vector
    vector_jacobian = np.zeros(self.vector_shape + self.vector_shape)
    for i in range(self.vector_shape[0]):
      for j in range(self.vector_shape[0]):
        if i == j:
          vector_jacobian[i, j] = 1.0
    vector_jacobian = tf.constant(vector_jacobian)
    expected_jacobian = [[tf.zeros(self.vector_shape), vector_jacobian],
                         [tf.zeros(self.vector_shape + self.matrix_shape)]]
    actual_jacobian = test_util.approximate_jacobian(self.linear_vector,
                                                     self.variables_structure)
    tf.nest.map_structure(
        lambda a, e: self.assertAllClose(a, e, atol=self.close_atol),
        actual_jacobian, expected_jacobian)

    # matrix
    matrix_jacobian = np.zeros(self.matrix_shape + self.matrix_shape)
    for i in range(self.matrix_shape[0]):
      for j in range(self.matrix_shape[1]):
        for k in range(self.matrix_shape[0]):
          for l in range(self.matrix_shape[1]):
            if i == k and j == l:
              matrix_jacobian[i, j, k, l] = 1.0
    matrix_jacobian = tf.constant(matrix_jacobian)
    expected_jacobian = [[
        tf.zeros(self.matrix_shape),
        tf.zeros(self.matrix_shape + self.vector_shape)
    ], [matrix_jacobian]]
    actual_jacobian = test_util.approximate_jacobian(self.linear_matrix,
                                                     self.variables_structure)
    tf.nest.map_structure(
        lambda a, e: self.assertAllClose(a, e, atol=self.close_atol),
        actual_jacobian, expected_jacobian)

  @test_util.eager_mode_toggle
  def test_gradient(self):
    """Compares approximation against exact gradient."""

    def f_nested():
      """Returns a nested structure."""
      return [self.f_tensor(), [[self.f_tensor()], self.f_tensor()]]

    for f in [self.f_tensor, f_nested]:
      # test with respect to single variable
      with tf.GradientTape() as tape:
        value = f()
      expected_gradient = tape.gradient(value, self.vector_var)
      actual_gradient = test_util.approximate_gradient(f, self.vector_var)
      self.assertNotAllClose(
          expected_gradient,
          tf.zeros_like(expected_gradient),
          atol=self.not_zero_atol)
      self.assertAllClose(
          actual_gradient, expected_gradient, atol=self.close_atol)

      # test with respect to nested variable structure
      with tf.GradientTape() as tape:
        value = f()
      expected_gradient = tape.gradient(value, self.variables_structure)
      actual_gradient = test_util.approximate_gradient(f,
                                                       self.variables_structure)
      tf.nest.map_structure(
          lambda a: self.assertNotAllClose(
              a, tf.zeros_like(a), atol=self.not_zero_atol), actual_gradient)
      tf.nest.map_structure(
          lambda a, e: self.assertAllClose(a, e, atol=self.close_atol),
          actual_gradient, expected_gradient)

  @test_util.eager_mode_toggle
  def test_jacobian(self):
    """Compares approximation against exact jacobian."""

    # test with respect to single variable
    with tf.GradientTape() as tape:
      value = self.f_tensor()
    expected_jacobian = tape.jacobian(value, self.vector_var)
    actual_jacobian = test_util.approximate_jacobian(self.f_tensor,
                                                     self.vector_var)
    self.assertNotAllClose(
        expected_jacobian,
        tf.zeros_like(expected_jacobian),
        atol=self.not_zero_atol)
    self.assertAllClose(
        actual_jacobian, expected_jacobian, atol=self.close_atol)

    # test with respect to nested variable structure
    with tf.GradientTape() as tape:
      value = self.f_tensor()
    expected_jacobian = tape.jacobian(value, self.variables_structure)
    actual_jacobian = test_util.approximate_jacobian(self.f_tensor,
                                                     self.variables_structure)
    tf.nest.map_structure(
        lambda a: self.assertNotAllClose(
            a, tf.zeros_like(a), atol=self.not_zero_atol), expected_jacobian)
    tf.nest.map_structure(
        lambda a, e: self.assertAllClose(a, e, atol=self.close_atol),
        actual_jacobian, expected_jacobian)
