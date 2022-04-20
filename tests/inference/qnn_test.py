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
"""Tests for qhbmlib.inference.qnn"""

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

from qhbmlib import inference
from qhbmlib import models
from qhbmlib import utils
from tests import test_util


class QuantumInferenceTest(parameterized.TestCase, tf.test.TestCase):
  """Tests the Analytic and Sampled QuantumInference classes."""

  def setUp(self):
    """Initializes test objects."""
    super().setUp()
    self.python_random_seed = 11
    self.tf_random_seed = 11
    tf.random.set_seed(self.tf_random_seed)
    self.tf_random_seed_alt = 201
    self.tfp_seed = tf.constant([5, 6], dtype=tf.int32)

    self.close_atol_analytic = 2e-3
    self.close_atol_sampled = 2e-2
    self.not_zero_atol = max(self.close_atol_analytic, self.close_atol_sampled)

    self.expectation_samples = int(1e6)

    # Build QNN representing X^p|s>
    self.num_bits = 3
    self.raw_qubits = cirq.GridQubit.rect(1, self.num_bits)
    p_param = sympy.Symbol("p")
    p_circuit = cirq.Circuit(cirq.X(q)**p_param for q in self.raw_qubits)
    self.p_qnn = models.DirectQuantumCircuit(
        p_circuit,
        initializer=tf.keras.initializers.RandomUniform(
            minval=-1.0, maxval=1.0, seed=self.tf_random_seed),
        name="p_qnn")

  def test_init(self):
    """Confirms QuantumInference classes are initialized correctly."""
    expected_name = "test_qnn_name"
    actual_exp = inference.AnalyticQuantumInference(
        self.p_qnn, name=expected_name)
    self.assertEqual(actual_exp.name, expected_name)
    self.assertEqual(actual_exp.circuit, self.p_qnn)

    expected_expectation_samples = 41827
    actual_exp = inference.SampledQuantumInference(
        self.p_qnn, expected_expectation_samples, name=expected_name)
    self.assertEqual(actual_exp.name, expected_name)
    self.assertEqual(actual_exp._expectation_samples,
                     expected_expectation_samples)
    self.assertEqual(actual_exp.circuit, self.p_qnn)

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

    qnn_analytic = inference.AnalyticQuantumInference(self.p_qnn)
    qnn_sampled = inference.SampledQuantumInference(self.p_qnn,
                                                    self.expectation_samples)

    for qnn, atol in [(qnn_analytic, self.close_atol_analytic),
                      (qnn_sampled, self.close_atol_sampled)]:
      expectation_wrapper = tf.function(qnn.expectation)
      actual = []
      actual_grad = []
      for op in all_ops:
        with tf.GradientTape() as tape:
          current_exp = expectation_wrapper(initial_states, op)
        current_grad = tf.squeeze(
            tape.jacobian(current_exp, self.p_qnn.trainable_variables))
        actual.append(current_exp)
        actual_grad.append(current_grad)

      for a, e in zip(actual, expected):
        self.assertAllClose(a, e, atol=atol)
      for a, e in zip(actual_grad, expected_grad):
        self.assertAllClose(a, e, atol=atol)

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
    bitstring_circuit = models.circuit_utils.bit_circuit(qubits)
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

    def expectation_func():
      """Computes the current expectation values."""
      total_resolvers = generate_resolvers()
      return tf.constant([[
          cirq.Simulator().simulate_expectation_values(total_circuit, o,
                                                       r)[0].real
          for o in raw_ops
      ]
                          for r in total_resolvers])

    # hamiltonian model and inference
    actual_circuit = models.QuantumCircuit(
        tfq.convert_to_tensor([raw_circuit]), qubits, tf.constant(symbols),
        [random_values], [[]])

    # calculate expected values
    expected_expectations = expectation_func()
    expected_jacobian = test_util.approximate_jacobian(
        expectation_func, actual_circuit.trainable_variables)

    # Check that expectations are a reasonable size
    self.assertAllGreater(
        tf.math.abs(expected_expectations), self.not_zero_atol)

    qnn_analytic = inference.AnalyticQuantumInference(actual_circuit)
    qnn_sampled = inference.SampledQuantumInference(actual_circuit,
                                                    self.expectation_samples)
    for qnn, atol in [(qnn_analytic, self.close_atol_analytic),
                      (qnn_sampled, self.close_atol_sampled)]:
      expectation_wrapper = tf.function(qnn.expectation)
      with tf.GradientTape() as tape:
        actual_expectations = expectation_wrapper(all_bitstrings, ops)
      self.assertAllClose(actual_expectations, expected_expectations, atol=atol)

      actual_jacobian = tape.jacobian(actual_expectations,
                                      actual_circuit.trainable_variables)

      self.assertNotAllClose(
          expected_jacobian,
          tf.zeros_like(expected_jacobian),
          atol=self.not_zero_atol)
      self.assertAllClose(expected_jacobian, actual_jacobian, atol=atol)

  @parameterized.parameters({
      "energy_class": energy_class,
      "energy_args": energy_args,
  } for energy_class, energy_args in zip([models.BernoulliEnergy, models.KOBE],
                                         [[], [2]]))
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
    circuit_h = models.DirectQuantumCircuit(raw_circuit_h)
    circuit_h.build([])
    initial_random_values = tf.random.uniform([len(symbols)], -1, 1, tf.float32,
                                              self.tf_random_seed)
    circuit_h.set_weights([initial_random_values])
    hamiltonian_measure = models.Hamiltonian(energy_h, circuit_h)
    raw_shards = tfq.from_tensor(hamiltonian_measure.operator_shards)

    # set up the circuit to measure against
    model_raw_circuit = cirq.testing.random_circuit(qubits, n_moments,
                                                    act_fraction)
    model_circuit = models.DirectQuantumCircuit(model_raw_circuit)

    # bitstring injectors
    all_bitstrings = list(itertools.product([0, 1], repeat=self.num_bits))
    bitstring_circuit = models.circuit_utils.bit_circuit(qubits)
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

    def expectation_func():
      """Returns the current expectation values."""
      total_resolvers = generate_resolvers()
      return tf.stack([
          tf.stack([
              hamiltonian_measure.energy.operator_expectation([
                  cirq.Simulator().simulate_expectation_values(
                      total_circuit, o, r)[0].real for o in raw_shards
              ])
          ]) for r in total_resolvers
      ])

    expected_expectations = expectation_func()
    self.assertNotAllClose(
        expected_expectations,
        tf.zeros_like(expected_expectations),
        atol=self.not_zero_atol)
    expected_jacobian_thetas = test_util.approximate_jacobian(
        expectation_func, hamiltonian_measure.energy.trainable_variables)
    self.assertNotAllClose(
        expected_jacobian_thetas,
        tf.zeros_like(expected_jacobian_thetas),
        atol=self.not_zero_atol)
    expected_jacobian_phis = test_util.approximate_jacobian(
        expectation_func, hamiltonian_measure.circuit.trainable_variables)
    self.assertNotAllClose(
        expected_jacobian_phis,
        tf.zeros_like(expected_jacobian_phis),
        atol=self.not_zero_atol)

    qnn_analytic = inference.AnalyticQuantumInference(model_circuit)
    qnn_sampled = inference.SampledQuantumInference(model_circuit,
                                                    self.expectation_samples)
    for qnn, atol in [(qnn_analytic, self.close_atol_analytic),
                      (qnn_sampled, self.close_atol_sampled)]:
      expectation_wrapper = tf.function(qnn.expectation)
      with tf.GradientTape() as tape:
        actual_expectations = expectation_wrapper(all_bitstrings,
                                                  hamiltonian_measure)
      self.assertAllClose(actual_expectations, expected_expectations, atol=atol)

      actual_jacobian_thetas, actual_jacobian_phis = tape.jacobian(
          actual_expectations,
          (hamiltonian_measure.energy.trainable_variables,
           hamiltonian_measure.circuit.trainable_variables))
      self.assertAllClose(
          actual_jacobian_phis, expected_jacobian_phis, atol=atol)
      self.assertAllClose(
          actual_jacobian_thetas, expected_jacobian_thetas, atol=atol)

  @test_util.eager_mode_toggle
  def test_expectation_bitstring_energy(self):
    """Tests Hamiltonian containing a general BitstringEnergy diagonal."""

    # Circuit preparation
    qubits = cirq.GridQubit.rect(1, self.num_bits)
    batch_size = 1
    n_moments = 3
    act_fraction = 1.0
    num_symbols = 4
    symbols = set()
    for _ in range(num_symbols):
      symbols.add("".join(random.sample(string.ascii_letters, 10)))
    symbols = sorted(list(symbols))

    # state circuit
    state_qnn_symbols = symbols[num_symbols // 2:]
    state_raw_circuits, _ = tfq_util.random_symbol_circuit_resolver_batch(
        qubits,
        state_qnn_symbols,
        batch_size,
        n_moments=n_moments,
        p=act_fraction)
    state_raw_circuit = state_raw_circuits[0]
    qnn_minval = 0.25
    qnn_maxval = 0.75
    state_circuit_initializer = tf.keras.initializers.RandomUniform(
        minval=qnn_minval, maxval=qnn_maxval, seed=self.tf_random_seed)
    state_circuit = models.DirectQuantumCircuit(state_raw_circuit,
                                                state_circuit_initializer)

    # state qnn
    expectation_samples = int(1e6)
    actual_qnn = inference.SampledQuantumInference(
        state_circuit, expectation_samples=expectation_samples)

    # hamiltonian circuit
    hamiltonian_qnn_symbols = symbols[:num_symbols // 2]
    hamiltonian_raw_circuits, _ = tfq_util.random_symbol_circuit_resolver_batch(
        qubits,
        hamiltonian_qnn_symbols,
        batch_size,
        n_moments=n_moments,
        p=act_fraction)
    hamiltonian_raw_circuit = hamiltonian_raw_circuits[0]
    # Note that this initializer uses a different seed than the `state_circuit`
    # initializer does.  This is to make sure `state_circuit` and
    # `hamiltonian_circuit` have different parameter values, to assuage a worry
    # about pathological behavior if
    # `state_circuit.circuit + hamiltonian.circuit_dagger` was possibly identity
    hamiltonian_circuit_initializer = tf.keras.initializers.RandomUniform(
        minval=qnn_minval, maxval=qnn_maxval, seed=self.tf_random_seed_alt)
    hamiltonian_circuit = models.DirectQuantumCircuit(
        hamiltonian_raw_circuit, hamiltonian_circuit_initializer)
    hamiltonian_circuit.build([])

    # Total circuit
    bitstring_circuit = models.circuit_utils.bit_circuit(qubits)
    measurement_circuit = cirq.Circuit(
        cirq.measure(q, key=f"measure_qubit_{n}") for n, q in enumerate(qubits))
    total_circuit = (
        bitstring_circuit + state_raw_circuit + hamiltonian_raw_circuit**-1 +
        measurement_circuit)

    # Resolvers for total circuit
    bitstring_symbols = sorted(tfq.util.get_circuit_symbols(bitstring_circuit))
    num_unique_bitstrings = 3
    num_repetitions = 2  # to ensure sending multiple identical inputs works
    initial_states_list = num_repetitions * random.sample(
        list(itertools.product([0, 1], repeat=self.num_bits)),
        num_unique_bitstrings)
    initial_states = tf.constant(initial_states_list, dtype=tf.int8)

    # TODO(#171): consider refactoring to accept symbol and variable tensors
    def generate_resolvers():
      """Return the current resolver."""
      state_values_list = state_circuit.trainable_variables[0].numpy().tolist()
      state_resolver = dict(zip(state_qnn_symbols, state_values_list))
      hamiltonian_values_list = hamiltonian_circuit.trainable_variables[
          0].numpy().tolist()
      hamiltonian_resolver = dict(
          zip(hamiltonian_qnn_symbols, hamiltonian_values_list))
      bitstring_resolvers = [
          dict(zip(bitstring_symbols, b)) for b in initial_states_list
      ]
      return [{
          **r,
          **state_resolver,
          **hamiltonian_resolver
      } for r in bitstring_resolvers]

    # hamiltonian energy
    num_layers = 2
    random.seed(self.python_random_seed)
    bits = random.sample(range(1000), self.num_bits)
    units = [2] * num_layers
    activations = random.sample([
        "elu", "exponential", "gelu", "hard_sigmoid", "linear", "relu", "selu",
        "sigmoid", "softmax", "softplus", "softsign", "swish", "tanh"
    ], num_layers)
    expected_layer_list = []
    min_val = -0.75
    max_val = 0.75
    for i in range(num_layers):
      kernel_initializer = tf.keras.initializers.RandomUniform(
          minval=min_val, maxval=max_val, seed=(self.tf_random_seed_alt + i))
      bias_initializer = tf.keras.initializers.RandomUniform(
          minval=min_val,
          maxval=max_val,
          seed=(self.tf_random_seed_alt + 2 * i + 1))
      expected_layer_list.append(
          tf.keras.layers.Dense(
              units[i],
              activation=activations[i],
              kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer))
    kernel_initializer = tf.keras.initializers.RandomUniform(
        minval=min_val, maxval=max_val, seed=(self.tf_random_seed_alt + 2 * i))
    bias_initializer = tf.keras.initializers.RandomUniform(
        minval=min_val,
        maxval=max_val,
        seed=(self.tf_random_seed_alt + 2 * i + 1))
    expected_layer_list.append(
        tf.keras.layers.Dense(
            1,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer))
    expected_layer_list.append(utils.Squeeze(-1))
    hamiltonian_energy = models.BitstringEnergy(bits, expected_layer_list)
    hamiltonian_energy.build([None, self.num_bits])
    # TODO(#209)
    _ = hamiltonian_energy(tf.constant([[0] * self.num_bits], tf.int8))
    hamiltonian = models.Hamiltonian(hamiltonian_energy, hamiltonian_circuit)

    # Get expectations
    total_resolvers = generate_resolvers()
    raw_expectations = []
    # Note the keys must match those used in `measurement_circuit`.
    qb_keys = [(q, f"measure_qubit_{i}") for i, q in enumerate(qubits)]
    for r in total_resolvers:
      samples_pd = cirq.Simulator().sample(
          total_circuit, repetitions=expectation_samples, params=r)
      samples = samples_pd[[x[1] for x in qb_keys]].to_numpy()
      current_energies = hamiltonian_energy(samples)
      raw_expectations.append(
          tf.math.reduce_mean(current_energies, keepdims=True))
    expected_expectations = tf.stack(raw_expectations)
    self.assertNotAllClose(
        expected_expectations,
        tf.zeros_like(expected_expectations),
        atol=self.not_zero_atol)

    # Compare
    expectation_wrapper = tf.function(actual_qnn.expectation)
    actual_expectations = expectation_wrapper(initial_states, hamiltonian)
    self.assertAllClose(
        actual_expectations,
        expected_expectations,
        atol=self.close_atol_sampled)
    self.assertAllEqual(
        tf.shape(actual_expectations), [len(initial_states_list), 1])

    expected_derivatives = test_util.approximate_jacobian(
        functools.partial(expectation_wrapper, initial_states, hamiltonian),
        hamiltonian.trainable_variables)
    for derivative in expected_derivatives:
      # Checks that at last one entry in each variable's derivative is
      # not too close to zero.
      self.assertNotAllClose(derivative, tf.zeros_like(derivative),
                             self.not_zero_atol)

    with tf.GradientTape() as tape:
      actual_expectations = expectation_wrapper(initial_states, hamiltonian)
    actual_derivatives = tape.jacobian(actual_expectations,
                                       hamiltonian.trainable_variables)
    self.assertEqual(len(actual_derivatives), len(expected_derivatives))
    for actual, expected in zip(actual_derivatives, expected_derivatives):
      # atol is ok here because we checked above derivatives are not all zero
      self.assertAllClose(actual, expected, atol=self.close_atol_sampled)

  @test_util.eager_mode_toggle
  def test_sample_basic(self):
    """Confirms correct sampling from identity, bit flip, and GHZ QNNs."""
    bitstrings = tf.constant(
        list(itertools.product([0, 1], repeat=self.num_bits)), dtype=tf.int8)
    counts = tf.random.uniform([tf.shape(bitstrings)[0]], 100, 1000, tf.int32)

    ident_qnn = models.DirectQuantumCircuit(
        cirq.Circuit(cirq.I(q) for q in self.raw_qubits), name="identity")
    q_infer = inference.SampledQuantumInference(ident_qnn,
                                                self.expectation_samples)
    sample_wrapper = tf.function(q_infer._sample)
    test_samples = sample_wrapper(bitstrings, counts)
    for i, (b, c) in enumerate(zip(bitstrings, counts)):
      self.assertEqual(tf.shape(test_samples[i].to_tensor())[0], c)
      for j in range(c):
        self.assertAllEqual(test_samples[i][j], b)

    flip_qnn = models.DirectQuantumCircuit(
        cirq.Circuit(cirq.X(q) for q in self.raw_qubits), name="flip")
    q_infer = inference.SampledQuantumInference(flip_qnn,
                                                self.expectation_samples)
    sample_wrapper = tf.function(q_infer._sample)
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
    ghz_qnn = models.DirectQuantumCircuit(
        ghz_circuit,
        initializer=tf.keras.initializers.Constant(value=0.5),
        name="ghz")
    q_infer = inference.SampledQuantumInference(ghz_qnn,
                                                self.expectation_samples)
    sample_wrapper = tf.function(q_infer._sample)
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
    test_qnn = models.DirectQuantumCircuit(
        cirq.Circuit(cirq.H(cirq.GridQubit(0, 0))))
    test_infer = inference.SampledQuantumInference(test_qnn,
                                                   self.expectation_samples)
    sample_wrapper = tf.function(test_infer._sample)
    bitstrings = tf.constant([[0], [0]], dtype=tf.int8)
    _, samples_counts = sample_wrapper(bitstrings, counts)
    # QNN samples should be half 0 and half 1.
    self.assertAllClose(
        samples_counts[0], samples_counts[1], atol=max_counts // 1000)


if __name__ == "__main__":
  logging.info("Running qnn_test.py ...")
  tf.test.main()
