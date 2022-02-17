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
"""Tests for the circuit_infer module."""

from absl import logging
from absl.testing import parameterized
import functools
import itertools
import random
import string

import cirq
import math
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
from tensorflow_quantum.python import util as tfq_util

from qhbmlib import circuit_infer
from qhbmlib import circuit_model
from qhbmlib import circuit_model_utils
from qhbmlib import energy_model
from qhbmlib import hamiltonian_model
from tests import test_util


class QuantumInferenceTest(parameterized.TestCase, tf.test.TestCase):
  """Tests the QuantumInference class."""

  def setUp(self):
    """Initializes test objects."""
    super().setUp()
    self.tf_random_seed = 10
    self.tfp_seed = tf.constant([5, 6], dtype=tf.int32)

    self.close_atol = 1e-3
    self.close_rtol = 1e-3
    self.not_zero_atol = 2e-3

    # Build QNN representing X^p|s>
    self.num_bits = 3
    self.raw_qubits = cirq.GridQubit.rect(1, self.num_bits)
    p_param = sympy.Symbol("p")
    p_circuit = cirq.Circuit(cirq.X(q)**p_param for q in self.raw_qubits)
    self.p_qnn = circuit_model.DirectQuantumCircuit(
        p_circuit,
        initializer=tf.keras.initializers.RandomUniform(
            minval=-1.0, maxval=1.0, seed=self.tf_random_seed),
        name="p_qnn")

  def test_init(self):
    """Confirms QuantumInference is initialized correctly."""
    expected_backend = "noiseless"
    expected_differentiator = None
    expected_name = "TestOE"
    actual_exp = circuit_infer.QuantumInference(
        self.p_qnn,
        backend=expected_backend,
        differentiator=expected_differentiator,
        name=expected_name)
    self.assertEqual(actual_exp.name, expected_name)
    self.assertEqual(actual_exp.backend, expected_backend)
    self.assertEqual(actual_exp.circuit, self.p_qnn)
    self.assertEqual(actual_exp.differentiator, expected_differentiator)

  @test_util.eager_mode_toggle
  def test_expectation(self):
    r"""Confirms basic correct expectation values and derivatives.
    Consider a circuit where each qubit has a gate X^p. Diagonalization of X is
        |0 1|   |-1 1||-1 0||-1/2 1/2|
    X = |1 0| = | 1 1|| 0 1|| 1/2 1/2|
    Therefore we have
          |-1 1||(-1)^p 0||-1/2 1/2|       |(-1)^p  -(-1)^p|       |1 1|
    X^p = | 1 1||   0   1|| 1/2 1/2| = 1/2 |-(-1)^p  (-1)^p| + 1/2 |1 1|
    where (-1)^p = cos(pi * p) + isin(pi * p).
    Consider the action on the two basis states:
                 | (-1)^p|       |1|
    X^p|0> = 1/2 |-(-1)^p| + 1/2 |1|
                 |-(-1)^p|       |1|
    X^p|1> = 1/2 | (-1)^p| + 1/2 |1|
    so, for s in {0, 1},
                 | ((-1)^s)(-1)^p|       |1|
    X^p|s> = 1/2 |-((-1)^s)(-1)^p| + 1/2 |1|
    similarly,
    <0|(X^p)^dagger = 1/2 |((-1)^p)^dagger -((-1)^p)^dagger| + 1/2 |1 1|
    <1|(X^p)^dagger = 1/2 |-((-1)^p)^dagger ((-1)^p)^dagger| + 1/2 |1 1|
    so
    <s|(X^p)^dagger = 1/2 |((-1)^s)((-1)^p)^dagger -((-1)^s)((-1)^p)^dagger|
                                                                     + 1/2 |1 1|
    where ((-1)^p)^dagger = cos(pi * p) - isin(pi * p).
    We want to see what the expectation values <s|(X^p)^dagger W X^p|s> are,
    for W in {X, Y, Z}.  Applying the above results, we have
    <s|(X^p)^dagger X X^p|s> = 0
    <s|(X^p)^dagger Y X^p|s> = -(-1)^s sin(pi * p)
    <s|(X^p)^dagger Z X^p|s> = (-1)^s cos(pi * p)
    Since these expectation values are in terms of p, we can calculate their
    derivatives with respect to p:
    d/dp <s|(X^p)^dagger X X^p|s> = 0
    d/dp <s|(X^p)^dagger Y X^p|s> = -(-1)^s pi cos(pi * p)
    d/dp <s|(X^p)^dagger Z X^p|s> = -(-1)^s pi sin(pi * p)
    """

    # Build inference object
    exp_infer = circuit_infer.QuantumInference(self.p_qnn)

    # Get all the bitstrings multiple times.
    initial_states_list = 5 * list(
        itertools.product([0, 1], repeat=self.num_bits))
    initial_states = tf.constant(initial_states_list, dtype=tf.int8)

    # Get true expectation values based on the bitstrings.
    expected_x_exps = []
    expected_x_exps_grad = []
    expected_y_exps = []
    expected_y_exps_grad = []
    expected_z_exps = []
    expected_z_exps_grad = []
    expected = [
        expected_x_exps,
        expected_y_exps,
        expected_z_exps,
    ]
    expected_grad = [
        expected_x_exps_grad,
        expected_y_exps_grad,
        expected_z_exps_grad,
    ]
    sin_pi_p = math.sin(math.pi * self.p_qnn.symbol_values[0])
    cos_pi_p = math.cos(math.pi * self.p_qnn.symbol_values[0])
    for bits in initial_states_list:
      for exps in expected + expected_grad:
        exps.append([])
      for s in bits:
        expected_x_exps[-1].append(0)
        expected_x_exps_grad[-1].append(0)
        expected_y_exps[-1].append(-((-1.0)**s) * sin_pi_p)
        expected_y_exps_grad[-1].append(-((-1.0)**s) * math.pi * cos_pi_p)
        expected_z_exps[-1].append(((-1.0)**s) * cos_pi_p)
        expected_z_exps_grad[-1].append(-((-1.0)**s) * math.pi * sin_pi_p)

    # Measure operators on every qubit.
    x_ops = tfq.convert_to_tensor([1 * cirq.X(q) for q in self.raw_qubits])
    y_ops = tfq.convert_to_tensor([1 * cirq.Y(q) for q in self.raw_qubits])
    z_ops = tfq.convert_to_tensor([1 * cirq.Z(q) for q in self.raw_qubits])
    all_ops = [x_ops, y_ops, z_ops]

    expectation_wrapper = tf.function(exp_infer.expectation)
    actual = []
    actual_grad = []
    for op in all_ops:
      with tf.GradientTape() as tape:
        current_exp = expectation_wrapper(initial_states, op)
      current_grad = tf.squeeze(
          tape.jacobian(current_exp, self.p_qnn.trainable_variables))
      actual.append(current_exp)
      actual_grad.append(current_grad)

    self.assertAllClose(actual, expected, rtol=self.close_rtol)
    self.assertAllClose(actual_grad, expected_grad, rtol=self.close_rtol)

  @test_util.eager_mode_toggle
  def test_expectation_cirq(self):
    """Compares library expectation values to those from Cirq."""
    # observable
    qubits = cirq.GridQubit.rect(1, self.num_bits)
    raw_ops = [
        cirq.PauliSum.from_pauli_strings(
            [cirq.PauliString(cirq.Z(q)) for q in qubits])
    ]
    ops = tfq.convert_to_tensor(raw_ops)

    # unitary
    batch_size = 1
    n_moments = 3
    act_fraction = 1.0
    num_symbols = 4
    symbols = set()
    for _ in range(num_symbols):
      symbols.add("".join(random.sample(string.ascii_letters, 10)))
    symbols = sorted(list(symbols))
    raw_circuits, _ = tfq_util.random_symbol_circuit_resolver_batch(
        qubits, symbols, batch_size, n_moments=n_moments, p=act_fraction)
    raw_circuit = raw_circuits[0]
    initial_random_values = tf.random.uniform([len(symbols)], -1, 1, tf.float32,
                                              self.tf_random_seed)
    random_values = tf.Variable(initial_random_values)

    all_bitstrings = list(itertools.product([0, 1], repeat=self.num_bits))
    bitstring_circuit = circuit_model_utils.bit_circuit(qubits)
    bitstring_symbols = sorted(tfq.util.get_circuit_symbols(bitstring_circuit))
    total_circuit = bitstring_circuit + raw_circuit

    def generate_resolvers():
      """Return the current resolver."""
      random_values_list = random_values.read_value().numpy().tolist()
      base_resolver = dict(zip(symbols, random_values_list))
      bitstring_resolvers = [
          dict(zip(bitstring_symbols, b)) for b in all_bitstrings
      ]
      return [{**r, **base_resolver} for r in bitstring_resolvers]

    def delta_expectations_func(k, var, delta):
      """Calculate the expectation with kth entry of `var` perturbed."""
      num_elts = tf.size(var)
      old_value = var.read_value()
      var.assign(old_value + delta * tf.one_hot(k, num_elts, 1.0, 0.0))
      total_resolvers = generate_resolvers()
      raw_delta_expectations = tf.constant([[
          cirq.Simulator().simulate_expectation_values(total_circuit, o,
                                                       r)[0].real
          for o in raw_ops
      ]
                                            for r in total_resolvers])
      delta_expectations = tf.constant(raw_delta_expectations)
      var.assign(old_value)
      return delta_expectations

    # hamiltonian model and inference
    circuit = circuit_model.QuantumCircuit(
        tfq.convert_to_tensor([raw_circuit]), qubits, tf.constant(symbols),
        [random_values], [[]])
    q_infer = circuit_infer.QuantumInference(circuit)

    # calculate expected values
    expected_expectations = delta_expectations_func(0, random_values, 0)
    print(f"expected_expectations: {expected_expectations}")

    # Check that expectations are a reasonable size
    self.assertAllGreater(
        tf.math.abs(expected_expectations), self.not_zero_atol)

    expectation_wrapper = tf.function(q_infer.expectation)
    with tf.GradientTape() as tape:
      actual_expectations = expectation_wrapper(all_bitstrings, ops)
    self.assertAllClose(
        actual_expectations, expected_expectations, rtol=self.close_rtol)

    def expectations_derivative(variables_list):
      """Approximately differentiates expectations with respect to the inputs"""
      derivatives = []
      for var in variables_list:
        var_derivative_list = []
        num_elts = tf.size(var)  # Assumes variable is 1D
        for n in range(num_elts):
          this_derivative = test_util.approximate_derivative_unsummed(
              functools.partial(delta_expectations_func, n, var))
          var_derivative_list.append(this_derivative.numpy())
        derivatives.append(tf.constant(var_derivative_list))
      return derivatives

    expected_expectations_derivative = tf.transpose(
        tf.squeeze(expectations_derivative(circuit.trainable_variables)))
    actual_expectations_derivative = tf.squeeze(
        tape.jacobian(actual_expectations, circuit.trainable_variables))

    self.assertNotAllClose(
        expected_expectations_derivative,
        tf.zeros_like(expected_expectations_derivative),
        atol=self.not_zero_atol)
    self.assertAllClose(
        expected_expectations_derivative,
        actual_expectations_derivative,
        atol=self.close_atol)

  @parameterized.parameters({
      "energy_class": energy_class,
      "energy_args": energy_args,
  } for energy_class, energy_args in zip(
      [energy_model.BernoulliEnergy, energy_model.KOBE], [[], [2]]))
  @test_util.eager_mode_toggle
  def test_expectation_modular_hamiltonian(self, energy_class, energy_args):
    """Confirm expectation of modular Hamiltonians works."""
    # set up the modular Hamiltonian to measure
    # EBM
    energy_h = energy_class(*([list(range(self.num_bits))] + energy_args))
    energy_h.build([None, self.num_bits])

    # QNN
    qubits = cirq.GridQubit.rect(1, self.num_bits)
    batch_size = 1
    n_moments = 4
    act_fraction = 1.0
    num_symbols = 4
    symbols = set()
    for _ in range(num_symbols):
      symbols.add("".join(random.sample(string.ascii_letters, 10)))
    symbols = sorted(list(symbols))
    raw_circuits, _ = tfq_util.random_symbol_circuit_resolver_batch(
        qubits, symbols, batch_size, n_moments=n_moments, p=act_fraction)
    raw_circuit_h = raw_circuits[0]
    circuit_h = circuit_model.DirectQuantumCircuit(raw_circuit_h)
    circuit_h.build([])
    initial_random_values = tf.random.uniform([len(symbols)], -1, 1, tf.float32,
                                              self.tf_random_seed)
    circuit_h.set_weights([initial_random_values])
    hamiltonian_measure = hamiltonian_model.Hamiltonian(energy_h, circuit_h)
    raw_shards = tfq.from_tensor(hamiltonian_measure.operator_shards)

    # set up the circuit to measure against
    model_raw_circuit = cirq.testing.random_circuit(qubits, n_moments,
                                                    act_fraction)
    model_circuit = circuit_model.DirectQuantumCircuit(model_raw_circuit)
    model_infer = circuit_infer.QuantumInference(model_circuit)

    # bitstring injectors
    all_bitstrings = list(itertools.product([0, 1], repeat=self.num_bits))
    bitstring_circuit = circuit_model_utils.bit_circuit(qubits)
    bitstring_symbols = sorted(tfq.util.get_circuit_symbols(bitstring_circuit))
    total_circuit = bitstring_circuit + model_raw_circuit + raw_circuit_h**-1

    def generate_resolvers():
      """Return the current resolver."""
      random_values_list = circuit_h.trainable_variables[0].read_value().numpy(
      ).tolist()
      base_resolver = dict(zip(symbols, random_values_list))
      bitstring_resolvers = [
          dict(zip(bitstring_symbols, b)) for b in all_bitstrings
      ]
      return [{**r, **base_resolver} for r in bitstring_resolvers]

    def delta_expectations_func(k, var, delta):
      """Calculate the expectation with kth entry of `var` perturbed."""
      num_elts = tf.size(var)
      old_value = var.read_value()
      var.assign(old_value + delta * tf.one_hot(k, num_elts, 1.0, 0.0))
      total_resolvers = generate_resolvers()
      raw_delta_expectations = tf.stack([
          tf.stack([
              hamiltonian_measure.energy.operator_expectation([
                  cirq.Simulator().simulate_expectation_values(
                      total_circuit, o, r)[0].real for o in raw_shards
              ])
          ]) for r in total_resolvers
      ])
      delta_expectations = tf.constant(raw_delta_expectations)
      var.assign(old_value)
      return delta_expectations

    expected_expectations = delta_expectations_func(
        0, circuit_h.trainable_variables[0], 0)
    self.assertNotAllClose(
        expected_expectations,
        tf.zeros_like(expected_expectations),
        atol=self.not_zero_atol)

    expectation_wrapper = tf.function(model_infer.expectation)
    with tf.GradientTape() as tape:
      actual_expectations = expectation_wrapper(all_bitstrings,
                                                hamiltonian_measure)
    self.assertAllClose(actual_expectations, expected_expectations)

    def expectations_derivative(variables_list):
      """Approximately differentiates expectations with respect to the inputs"""
      derivatives = []
      for var in variables_list:
        var_derivative_list = []
        num_elts = tf.size(var)  # Assumes variable is 1D
        for n in range(num_elts):
          this_derivative = test_util.approximate_derivative_unsummed(
              functools.partial(delta_expectations_func, n, var))
          var_derivative_list.append(this_derivative.numpy())
        derivatives.append(tf.constant(var_derivative_list))
      return derivatives

    expected_derivatives_thetas = tf.transpose(
        tf.squeeze(
            expectations_derivative(
                hamiltonian_measure.energy.trainable_variables)))
    self.assertNotAllClose(
        expected_derivatives_thetas,
        tf.zeros_like(expected_derivatives_thetas),
        atol=self.not_zero_atol)
    expected_derivatives_phis = tf.transpose(
        tf.squeeze(
            expectations_derivative(
                hamiltonian_measure.circuit.trainable_variables)))
    self.assertNotAllClose(
        expected_derivatives_phis,
        tf.zeros_like(expected_derivatives_phis),
        atol=self.not_zero_atol)
    actual_derivatives_thetas, actual_derivatives_phis = tape.jacobian(
        actual_expectations, (hamiltonian_measure.energy.trainable_variables,
                              hamiltonian_measure.circuit.trainable_variables))
    actual_derivatives_thetas = tf.squeeze(actual_derivatives_thetas)
    actual_derivatives_phis = tf.squeeze(actual_derivatives_phis)
    self.assertAllClose(
        actual_derivatives_phis,
        expected_derivatives_phis,
        rtol=self.close_rtol)
    self.assertAllClose(
        actual_derivatives_thetas,
        expected_derivatives_thetas,
        rtol=self.close_rtol)

  @test_util.eager_mode_toggle
  def test_sample_basic(self):
    """Confirms correct sampling from identity, bit flip, and GHZ QNNs."""
    bitstrings = tf.constant(
        list(itertools.product([0, 1], repeat=self.num_bits)), dtype=tf.int8)
    counts = tf.random.uniform([tf.shape(bitstrings)[0]], 10, 100, tf.int32)

    ident_qnn = circuit_model.DirectQuantumCircuit(
        cirq.Circuit(cirq.I(q) for q in self.raw_qubits), name="identity")
    q_infer = circuit_infer.QuantumInference(ident_qnn)
    sample_wrapper = tf.function(q_infer.sample)
    test_samples = sample_wrapper(bitstrings, counts)
    for i, (b, c) in enumerate(zip(bitstrings, counts)):
      self.assertEqual(tf.shape(test_samples[i].to_tensor())[0], c)
      for j in range(c):
        self.assertAllEqual(test_samples[i][j], b)

    flip_qnn = circuit_model.DirectQuantumCircuit(
        cirq.Circuit(cirq.X(q) for q in self.raw_qubits), name="flip")
    q_infer = circuit_infer.QuantumInference(flip_qnn)
    sample_wrapper = tf.function(q_infer.sample)
    test_samples = sample_wrapper(bitstrings, counts)
    for i, (b, c) in enumerate(zip(bitstrings, counts)):
      self.assertEqual(tf.shape(test_samples[i].to_tensor())[0], c)
      for j in range(c):
        self.assertAllEqual(
            test_samples[i][j],
            tf.cast(tf.math.logical_not(tf.cast(b, tf.bool)), tf.int8))

    ghz_param = sympy.Symbol("ghz")
    ghz_circuit = cirq.Circuit(cirq.X(
        self.raw_qubits[0])**ghz_param) + cirq.Circuit(
            cirq.CNOT(q0, q1)
            for q0, q1 in zip(self.raw_qubits, self.raw_qubits[1:]))
    ghz_qnn = circuit_model.DirectQuantumCircuit(
        ghz_circuit,
        initializer=tf.keras.initializers.Constant(value=0.5),
        name="ghz")
    q_infer = circuit_infer.QuantumInference(ghz_qnn)
    sample_wrapper = tf.function(q_infer.sample)
    test_samples = sample_wrapper(
        tf.expand_dims(tf.constant([0] * self.num_bits, dtype=tf.int8), 0),
        tf.expand_dims(counts[0], 0))[0].to_tensor()
    # Both |0...0> and |1...1> should be among the measured bitstrings
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0] * self.num_bits, dtype=tf.int8), test_samples))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1] * self.num_bits, dtype=tf.int8), test_samples))

  @test_util.eager_mode_toggle
  def test_sample_uneven(self):
    """Check for discrepancy in samples when count entries differ."""
    max_counts = int(1e7)
    counts = tf.constant([max_counts // 2, max_counts])
    test_qnn = circuit_model.DirectQuantumCircuit(
        cirq.Circuit(cirq.H(cirq.GridQubit(0, 0))))
    test_infer = circuit_infer.QuantumInference(test_qnn)
    sample_wrapper = tf.function(test_infer.sample)
    bitstrings = tf.constant([[0], [0]], dtype=tf.int8)
    _, samples_counts = sample_wrapper(bitstrings, counts)
    # QNN samples should be half 0 and half 1.
    self.assertAllClose(
        samples_counts[0], samples_counts[1], atol=max_counts // 1000)


if __name__ == "__main__":
  logging.info("Running circuit_infer_test.py ...")
  tf.test.main()
