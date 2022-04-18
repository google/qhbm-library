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
"""Utility functions for running tests."""

import cirq
import sympy
import tensorflow as tf

from qhbmlib import inference
from qhbmlib import models


def get_xz_rotation(q, a, b):
  """Two-axis single qubit rotation."""
  return cirq.Circuit(cirq.X(q)**a, cirq.Z(q)**b)


def get_cz_exp(q0, q1, a):
  """Exponent of entangling CZ gate."""
  return cirq.Circuit(cirq.CZPowGate(exponent=a)(q0, q1))


def get_xz_rotation_layer(qubits, layer_num, name):
  """Apply two-axis single qubit rotations to all the given qubits."""
  circuit = cirq.Circuit()
  for n, q in enumerate(qubits):
    sx, sz = sympy.symbols(
        f"sx_{name}_{layer_num}_{n} sz_{name}_{layer_num}_{n}")
    circuit += get_xz_rotation(q, sx, sz)
  return circuit


def get_cz_exp_layer(qubits, layer_num, name):
  """Apply parameterized CZ gates to all pairs of nearest-neighbor qubits."""
  circuit = cirq.Circuit()
  for n, (q0, q1) in enumerate(zip(qubits[::2], qubits[1::2])):
    a = sympy.symbols(f"sc_{name}_{layer_num}_{2 * n}")
    circuit += get_cz_exp(q0, q1, a)
  shifted_qubits = qubits[1::]
  for n, (q0, q1) in enumerate(zip(shifted_qubits[::2], shifted_qubits[1::2])):
    a = sympy.symbols(f"sc_{name}_{layer_num}_{2 * n + 1}")
    circuit += get_cz_exp(q0, q1, a)
  return circuit


def get_hardware_efficient_model_unitary(qubits, num_layers, name):
  """Build our full parameterized model unitary."""
  circuit = cirq.Circuit()
  for layer_num in range(num_layers):
    new_circ = get_xz_rotation_layer(qubits, layer_num, name)
    circuit += new_circ
    if len(qubits) > 1:
      new_circ = get_cz_exp_layer(qubits, layer_num, name)
      circuit += new_circ
  return circuit


def get_random_hamiltonian_and_inference(qubits,
                                         num_layers,
                                         identifier,
                                         num_samples,
                                         minval_thetas=-1.0,
                                         maxval_thetas=1.0,
                                         minval_phis=-1.0,
                                         maxval_phis=1.0,
                                         initializer_seed=None,
                                         ebm_seed=None):
  """Create a random QHBM for use in testing."""
  num_qubits = len(qubits)
  ebm_init = tf.keras.initializers.RandomUniform(minval_thetas, maxval_thetas,
                                                 initializer_seed)
  actual_energy = models.KOBE(list(range(num_qubits)), num_qubits, ebm_init)
  e_infer = inference.AnalyticEnergyInference(
      actual_energy, num_samples, name=identifier, initial_seed=ebm_seed)

  qnn_init = tf.keras.initializers.RandomUniform(minval_phis, maxval_phis,
                                                 initializer_seed)
  unitary = get_hardware_efficient_model_unitary(qubits, num_layers, identifier)
  actual_circuit = models.DirectQuantumCircuit(unitary, qnn_init)
  q_infer = inference.AnalyticQuantumInference(actual_circuit, name=identifier)
  random_qhbm = inference.QHBM(e_infer, q_infer)

  return random_qhbm.modular_hamiltonian, random_qhbm


def random_hermitian_matrix(num_qubits):
  """Returns a random Hermitian matrix.

  Uses the property that A + A* is Hermitian for any matrix A.

  Args:
    num_qubits: Number of qubits on which the matrix acts.
  """
  dim = 2**num_qubits
  val_range = 2
  random_real = tf.cast(
      tf.random.uniform([dim, dim], -val_range, val_range), tf.complex128)
  random_imag = 1j * tf.cast(
      tf.random.uniform([dim, dim], -val_range, val_range), tf.complex128)
  random_matrix = random_real + random_imag
  return random_matrix + tf.linalg.adjoint(random_matrix)


def random_unitary_matrix(num_qubits):
  """Returns a random unitary matrix.

  Uses the property that e^{-iH} is unitary for any Hermitian matrix H.

  Args:
    num_qubits: Number of qubits on which the matrix acts.
  """
  hermitian_matrix = random_hermitian_matrix(num_qubits)
  return tf.linalg.expm(-1j * hermitian_matrix)


def random_mixed_density_matrix(num_qubits, num_mixtures=5):
  """Returns a random pure density matrix.

  Applies a common random unitary to `num_mixtures` orthogonal states, then
  mixes them with random weights.

  Args:
    num_qubits: Number of qubits on which the matrix acts.
    num_mixtures: The number of orthogonal pure states to mix.

  Returns:
    final_state: The mixed density matrix.
    mixture_probabilities: The probability of each state in the mixture.
  """
  pre_probs = tf.random.uniform([num_mixtures], 1e-9)
  mixture_probabilities = pre_probs / tf.reduce_sum(pre_probs)
  random_unitary = random_unitary_matrix(num_qubits)
  dim = 2**num_qubits
  final_state = tf.zeros([dim, dim], tf.complex128)
  for i in range(num_mixtures):
    pure_state = tf.one_hot(i, dim, 1.0, 0.0, 0, tf.complex128)
    evolved_pure_state = tf.linalg.matvec(random_unitary, pure_state)
    adjoint_evolved_pure_state = tf.squeeze(
        tf.linalg.adjoint(tf.expand_dims(evolved_pure_state, 0)))
    final_state = final_state + tf.cast(
        mixture_probabilities[i], tf.complex128) * tf.einsum(
            "i,j->ij", evolved_pure_state, adjoint_evolved_pure_state)
  return final_state, mixture_probabilities


def entropy(probs):
  """Entropy function for a list of probabilities, allowing zeros."""
  return -tf.reduce_sum(tf.math.multiply_no_nan(tf.math.log(probs), probs))


def check_bitstring_exists(bitstring, bitstring_list):
  """True if `bitstring` is an entry of `bitstring_list`."""
  return tf.math.reduce_any(
      tf.reduce_all(tf.math.equal(bitstring, bitstring_list), 1))


def eager_mode_toggle(func):
  """Parameterizes the given test function to toggle `tf.function` tracing."""

  def toggled_function(*args, **kwargs):
    tf.config.run_functions_eagerly(True)
    # Ensures eager is turned back off even if first call raises.
    try:
      func(*args, **kwargs)
    except Exception as e:
      raise e
    finally:
      tf.config.run_functions_eagerly(False)
    func(*args, **kwargs)

  return toggled_function


def perturb_function(f, var, k, delta):
  """Evaluates the function with a specified variable perturbed.

  Args:
    f: Callable taking no arguments and returning a possibly nested structure
      whose atomic elements are `tf.Tensor`.
    var: `tf.Variable` to perturb.
    k: Entry of `var` to perturb.
    delta: Amount to perturb entry `k` of `var`.

  Return:
    f_value: Return of `f()` evaluated while `var` is perturbed.
  """
  num_elts = tf.size(var)
  old_value = var.read_value()
  perturbation_direction = tf.one_hot(k, num_elts, 1.0, 0.0, None, var.dtype)
  perturbation = tf.reshape(
      tf.cast(delta, var.dtype) * perturbation_direction, tf.shape(var))
  var.assign(old_value + perturbation)
  f_value = f()
  var.assign(old_value)
  return f_value


def _five_point_stencil(f, var, i, delta):
  """Computes the five point stencil.

  See wikipedia page on "five point stencil",
  https://en.wikipedia.org/wiki/Five-point_stencil
  Note: the error of this method scales with delta ** 4.

  Args:
    f: Callable taking no arguments and returning a possibly nested structure
      whose atomic elements are `tf.Tensor`.
    var: `tf.Variable` in which to differentiate `f`.
    i: Entry of `var` to perturb.
    delta: Size of the fundamental perturbation in the stencil.

  Returns:
    stencil: The five point stencil approximation of the jacobian of `f()` with
      respect to entry `i` of `var`.
  """
  forward_twice = perturb_function(f, var, i, 2.0 * delta)
  forward_once = perturb_function(f, var, i, delta)
  backward_once = perturb_function(f, var, i, -1.0 * delta)
  backward_twice = perturb_function(f, var, i, -2.0 * delta)
  numerator = tf.nest.map_structure(
      lambda a, b, c, d: -1.0 * a + 8.0 * b - 8.0 * c + d, forward_twice,
      forward_once, backward_once, backward_twice)
  stencil = tf.nest.map_structure(lambda x: x / (12.0 * delta), numerator)
  return stencil


def approximate_gradient(f, variables, delta=1e-1):
  """Approximates the gradient of f using five point stencil.

  Suppose the input function returns a possibly nested structure `r` under
  gradient tape `t`.  Then this function returns an approximation to
  `t.gradient(r, variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)`

  Args:
    f: Callable taking no arguments and returning a possibly nested structure
      whose atomic elements are `tf.Tensor`.
    variables: Possibly nested structure of `tf.Variable` in which to
      differentiate `f`.
    delta: Size of the fundamental perturbation in the stencil.

  Returns:
    The approximate gradient.  Has the same structure as the return from a
      corresponding call to `tf.GradientTape().gradient`.
  """

  def var_gradient(var):
    """Returns gradient of `f()` with respect to `var`."""

    def mapper_func(i):
      """Function to map across indices of flat `var`."""
      stencil = _five_point_stencil(f, var, i, delta)
      inner_sum = tf.nest.map_structure(tf.math.reduce_sum,
                                        tf.nest.flatten(stencil))
      outer_sum = tf.math.reduce_sum(tf.stack(inner_sum))
      entry_derivative = tf.reduce_sum(outer_sum)
      return entry_derivative

    derivatives = tf.map_fn(
        mapper_func, tf.range(tf.size(var)), fn_output_signature=tf.float32)
    return tf.reshape(derivatives, tf.shape(var))

  return tf.nest.map_structure(var_gradient, variables)


def approximate_jacobian(f, variables, delta=1e-1):
  """Approximates the jacobian of f using five point stencil.

  Suppose the input function returns a tensor `r` under gradient tape `t`.  Then
  this function returns an approximation to
  `t.jacobian(r, variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)`

  Args:
    f: Callable taking no arguments and returning a `tf.Tensor`.
    variables: Possibly nested structure of `tf.Variable` in which to
      differentiate `f`.
    delta: Size of the fundamental perturbation in the stencil.

  Returns:
    The approximate jacobian.  Has the same structure as the return from a
      corresponding call to `tf.GradientTape().jacobian`.
  """

  def var_jacobian(var):
    """Returns jacobian of `f()` with respect to `var`."""
    derivatives = tf.map_fn(
        lambda x: _five_point_stencil(f, var, x, delta),
        tf.range(tf.size(var)),
        fn_output_signature=tf.float32)
    f_shape = tf.shape(derivatives)[1:]  # shape of f()
    # swap function and variable dims
    transpose_perm = list(range(1, len(f_shape) + 1)) + [0]
    transpose_derivatives = tf.transpose(derivatives, transpose_perm)
    # reshape to correct Jacobian shape.
    reshape_shape = tf.concat([f_shape, tf.shape(var)], 0)
    return tf.reshape(transpose_derivatives, reshape_shape)

  return tf.nest.map_structure(var_jacobian, variables)
