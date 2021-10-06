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
"""Tests for the qnn module."""

import random
import itertools
from absl import logging

import cirq
import math
import numpy as np
import sympy
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_quantum as tfq

from qhbmlib import qnn
from tests import test_util

# Global tolerance, set for float32.
ATOL = 1e-5
GRAD_ATOL = 2e-4


class BitCircuitTest(tf.test.TestCase):
  """Test bit_circuit from the qhbm library."""

  def test_bit_circuit(self):
    """Confirm correct bit injector circuit creation."""
    my_qubits = [
        cirq.GridQubit(0, 2),
        cirq.GridQubit(1, 4),
        cirq.GridQubit(2, 2)
    ]
    identifier = "build_bit_test"
    test_circuit = qnn.bit_circuit(my_qubits, identifier)
    test_symbols = list(sorted(tfq.util.get_circuit_symbols(test_circuit)))
    expected_symbols = list(
        sympy.symbols(
            "build_bit_test_bit_0 build_bit_test_bit_1 build_bit_test_bit_2"))
    expected_circuit = cirq.Circuit(
        [cirq.X(q)**s for q, s in zip(my_qubits, expected_symbols)])
    self.assertAllEqual(test_symbols, expected_symbols)
    self.assertEqual(test_circuit, expected_circuit)


class QNNTest(tf.test.TestCase):
  """Tests the QNN class."""

  num_qubits = 5
  raw_symbols = [sympy.Symbol("s0"), sympy.Symbol("s1")]
  symbols = tf.constant([str(s) for s in raw_symbols])
  initializer = tf.keras.initializers.RandomUniform(-1.0, 1.0)
  raw_qubits = cirq.GridQubit.rect(1, num_qubits)
  backend = "noiseless"
  differentiator = None
  pqc = cirq.Circuit()
  for s in raw_symbols:
    for q in raw_qubits:
      pqc += cirq.X(q)**s
      pqc += cirq.Z(q)**s
  inverse_pqc = pqc**-1
  pqc_tfq = tfq.convert_to_tensor([pqc])
  inverse_pqc_tfq = tfq.convert_to_tensor([inverse_pqc])
  name = "TestOE"
  raw_bit_circuit = qnn.bit_circuit(raw_qubits, "bit_circuit")
  raw_bit_symbols = list(sorted(tfq.util.get_circuit_symbols(raw_bit_circuit)))
  bit_symbols = tf.constant([str(s) for s in raw_bit_symbols])
  bit_circuit = tfq.convert_to_tensor([raw_bit_circuit])

  def test_init(self):
    """Confirms QNN is initialized correctly."""
    test_qnn = qnn.QNN(
        self.pqc,
        backend=self.backend,
        differentiator=self.differentiator,
        is_analytic=True,
        name=self.name)
    self.assertEqual(self.name, test_qnn.name)
    self.assertAllEqual(self.symbols, test_qnn.symbols)
    self.assertAllEqual(self.backend, test_qnn.backend)
    self.assertAllEqual(self.differentiator, test_qnn.differentiator)
    self.assertAllEqual(True, test_qnn.is_analytic)
    self.assertAllEqual(
        tfq.from_tensor(self.pqc_tfq),
        tfq.from_tensor(test_qnn._pqc),
    )
    self.assertAllEqual(
        tfq.from_tensor(self.inverse_pqc_tfq),
        tfq.from_tensor(test_qnn._inverse_pqc),
    )
    self.assertAllEqual(self.bit_symbols, test_qnn._bit_symbols)
    self.assertEqual(
        tfq.from_tensor(self.bit_circuit),
        tfq.from_tensor(test_qnn._bit_circuit))

    self.assertEqual(
        tfq.from_tensor(test_qnn.pqc(resolve=True)),
        tfq.from_tensor(
            tfq.resolve_parameters(self.pqc_tfq, self.symbols,
                                   tf.expand_dims(test_qnn.values, 0))))
    self.assertEqual(
        tfq.from_tensor(test_qnn.inverse_pqc(resolve=True)),
        tfq.from_tensor(
            tfq.resolve_parameters(self.inverse_pqc_tfq, self.symbols,
                                   tf.expand_dims(test_qnn.values, 0))))

  def test_alternative_init(self):
    """Confirms that `symbols` and `values` get set correctly."""
    expected_values = self.initializer(shape=[self.num_qubits])
    actual_qnn = qnn.QNN(self.pqc, symbols=self.symbols, values=expected_values)
    self.assertAllEqual(actual_qnn.symbols, self.symbols)
    self.assertAllEqual(actual_qnn.values, expected_values)

  def test_add(self):
    """Confirms two QNNs are added successfully."""
    num_qubits = 5
    qubits = cirq.GridQubit.rect(1, num_qubits)

    pqc_1 = cirq.Circuit()
    symbols_1_str = ["s_1_{n}" for n in range(num_qubits)]
    symbols_1_sympy = [sympy.Symbol(s) for s in symbols_1_str]
    symbols_1 = tf.constant(symbols_1_str)
    for s, q in zip(symbols_1_sympy, qubits):
      pqc_1 += cirq.rx(s)(q)
    values_1 = self.initializer(shape=[num_qubits])

    pqc_2 = cirq.Circuit()
    symbols_2_str = ["s_2_{n}" for n in range(num_qubits)]
    symbols_2_sympy = [sympy.Symbol(s) for s in symbols_2_str]
    symbols_2 = tf.constant(symbols_2_str)
    for s, q in zip(symbols_2_sympy, qubits):
      pqc_2 += cirq.ry(s)(q)
    values_2 = self.initializer(shape=[num_qubits])

    qnn_1 = qnn.QNN(pqc_1, symbols=symbols_1, values=values_1)
    qnn_2 = qnn.QNN(pqc_2, symbols=symbols_2, values=values_2)
    actual_added = qnn_1 + qnn_2

    self.assertAllEqual(
        tfq.from_tensor(actual_added.pqc(False))[0],
        tfq.from_tensor(tfq.convert_to_tensor([pqc_1 + pqc_2]))[0])
    self.assertAllEqual(actual_added.symbols,
                        tf.concat([symbols_1, symbols_2], 0))
    self.assertAllEqual(actual_added.values, tf.concat([values_1, values_2], 0))

  def test_pow(self):
    """Confirms inverse works correctly."""
    actual_qnn = qnn.QNN(self.pqc)
    with self.assertRaisesRegex(ValueError, expected_regex="Only the inverse"):
      _ = actual_qnn**-2

    inverse_qnn = actual_qnn**-1
    actual_pqc = tfq.from_tensor(inverse_qnn.pqc(resolve=True))
    expected_pqc = tfq.from_tensor(
        tfq.resolve_parameters(self.inverse_pqc_tfq, self.symbols,
                               tf.expand_dims(actual_qnn._values, 0)))
    actual_inverse_pqc = tfq.from_tensor(inverse_qnn.inverse_pqc(resolve=True))
    expected_inverse_pqc = tfq.from_tensor(
        tfq.resolve_parameters(self.pqc_tfq, self.symbols,
                               tf.expand_dims(actual_qnn._values, 0)))
    self.assertEqual(actual_pqc, expected_pqc)
    self.assertEqual(actual_inverse_pqc, expected_inverse_pqc)
    # Ensure swapping circuits was actually meaningful
    self.assertNotEqual(actual_pqc, actual_inverse_pqc)

  def test_copy(self):
    """Confirms copied QNN has correct attributes."""
    test_qnn = qnn.QNN(
        self.pqc,
        initializer=self.initializer,
        backend=self.backend,
        differentiator=self.differentiator,
        name=self.name)
    test_qnn_copy = test_qnn.copy()
    self.assertEqual(test_qnn_copy.name, test_qnn.name)
    self.assertAllClose(test_qnn_copy.trainable_variables,
                        test_qnn.trainable_variables)
    self.assertAllEqual(test_qnn_copy.symbols, test_qnn.symbols)
    self.assertAllEqual(test_qnn_copy.backend, test_qnn.backend)
    self.assertAllEqual(test_qnn_copy.differentiator, test_qnn.differentiator)
    self.assertAllEqual(test_qnn_copy.is_analytic, test_qnn.is_analytic)
    self.assertAllEqual(
        tfq.from_tensor(test_qnn_copy._pqc),
        tfq.from_tensor(test_qnn._pqc),
    )
    self.assertAllEqual(
        tfq.from_tensor(test_qnn_copy._inverse_pqc),
        tfq.from_tensor(test_qnn._inverse_pqc),
    )
    self.assertAllEqual(test_qnn_copy.qubits, test_qnn.qubits)
    self.assertAllEqual(test_qnn_copy._bit_symbols, test_qnn._bit_symbols)
    self.assertEqual(
        tfq.from_tensor(test_qnn_copy._bit_circuit),
        tfq.from_tensor(test_qnn._bit_circuit))
    self.assertEqual(
        tfq.from_tensor(test_qnn_copy.pqc(resolve=True)),
        tfq.from_tensor(test_qnn.pqc(resolve=True)))
    self.assertEqual(
        tfq.from_tensor(test_qnn_copy.inverse_pqc(resolve=True)),
        tfq.from_tensor(test_qnn.inverse_pqc(resolve=True)))

  def test_circuits(self):
    """Confirms bitstring injectors are prepended to pqc."""
    bitstrings = 2 * list(itertools.product([0, 1], repeat=self.num_qubits))
    test_qnn = qnn.QNN(
        self.pqc,
        initializer=self.initializer,
        name=self.name,
    )
    test_circuits = test_qnn.circuits(
        tf.constant(bitstrings, dtype=tf.int8), resolve=True)
    test_circuits_deser = tfq.from_tensor(test_circuits)

    resolved_pqc = tfq.from_tensor(
        tfq.resolve_parameters(self.pqc_tfq, self.symbols,
                               tf.expand_dims(test_qnn.values, 0)))[0]
    bit_injectors = []
    for b in bitstrings:
      bit_injectors.append(
          cirq.Circuit(cirq.X(q)**b_i for q, b_i in zip(self.raw_qubits, b)))
    combined = [b + resolved_pqc for b in bit_injectors]

    for expected, test in zip(combined, test_circuits_deser):
      self.assertTrue(cirq.approx_eq(expected, test))

  def test_sample_basic(self):
    """Confirms correct sampling from identity, bit flip, and GHZ QNNs."""
    bitstrings = tf.constant(
        list(itertools.product([0, 1], repeat=self.num_qubits)), dtype=tf.int8)
    counts = tf.random.uniform([tf.shape(bitstrings)[0]],
                               minval=10,
                               maxval=100,
                               dtype=tf.int32)

    ident_qnn = qnn.QNN(
        cirq.Circuit(cirq.I(q) for q in self.raw_qubits), name="identity")
    test_samples = ident_qnn.sample(
        bitstrings, counts, reduce=False, unique=False)
    for i, (b, c) in enumerate(zip(bitstrings, counts)):
      self.assertEqual(tf.shape(test_samples[i].to_tensor())[0], c)
      for j in range(c):
        self.assertAllEqual(test_samples[i][j], b)

    flip_qnn = qnn.QNN(
        cirq.Circuit(cirq.X(q) for q in self.raw_qubits), name="flip")
    test_samples = flip_qnn.sample(
        bitstrings, counts, reduce=False, unique=False)
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
    ghz_qnn = qnn.QNN(
        ghz_circuit,
        initializer=tf.keras.initializers.Constant(value=0.5),
        name="ghz")
    test_samples = ghz_qnn.sample(
        tf.expand_dims(tf.constant([0] * self.num_qubits, dtype=tf.int8), 0),
        tf.expand_dims(counts[0], 0),
        reduce=False,
        unique=False)[0].to_tensor()
    # Both |0...0> and |1...1> should be among the measured bitstrings
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0] * self.num_qubits, dtype=tf.int8), test_samples))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1] * self.num_qubits, dtype=tf.int8), test_samples))

  def test_sample_uneven(self):
    """Check for discrepancy in samples when count entries differ."""
    max_counts = int(1e7)
    counts = tf.constant([max_counts // 2, max_counts])
    test_qnn = qnn.QNN(cirq.Circuit(cirq.H(cirq.GridQubit(0, 0))))
    bitstrings = tf.constant([[0], [0]], dtype=tf.int8)
    samples, samples_counts = test_qnn.sample(bitstrings, counts)
    # QNN samples should be half 0 and half 1.
    self.assertAllClose(
        samples_counts[0], samples_counts[1], atol=max_counts // 1000)

  def test_expectation(self):
    """Confirms basic correct expectation values and derivatives.

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
    # Build QNN representing X^p|s>
    p_param = sympy.Symbol("p")
    p_circuit = cirq.Circuit(cirq.X(q)**p_param for q in self.raw_qubits)
    p_qnn = qnn.QNN(
        p_circuit,
        initializer=tf.keras.initializers.RandomUniform(
            minval=-5.0, maxval=5.0),
        name="p_qnn")

    # Choose some bitstrings.
    num_bitstrings = 10
    bitstrings_raw = []
    for _ in range(num_bitstrings):
      bitstrings_raw.append([])
      for _ in range(self.num_qubits):
        if random.choice([True, False]):
          bitstrings_raw[-1].append(0)
        else:
          bitstrings_raw[-1].append(1)
    bitstrings = tf.constant(bitstrings_raw, dtype=tf.int8)
    counts = tf.random.uniform(
        shape=[num_bitstrings], minval=1, maxval=1000, dtype=tf.int32)

    # Get true expectation values based on the bitstrings.
    x_exps_true = []
    x_exps_grad_true = []
    y_exps_true = []
    y_exps_grad_true = []
    z_exps_true = []
    z_exps_grad_true = []
    sin_pi_p = math.sin(math.pi * p_qnn.values[0])
    cos_pi_p = math.cos(math.pi * p_qnn.values[0])
    for bits in bitstrings_raw:
      x_exps_true.append([])
      x_exps_grad_true.append([])
      y_exps_true.append([])
      y_exps_grad_true.append([])
      z_exps_true.append([])
      z_exps_grad_true.append([])
      for s in bits:
        x_exps_true[-1].append(0)
        x_exps_grad_true[-1].append(0)
        y_exps_true[-1].append(-((-1.0)**s) * sin_pi_p)
        y_exps_grad_true[-1].append(-((-1.0)**s) * math.pi * cos_pi_p)
        z_exps_true[-1].append(((-1.0)**s) * cos_pi_p)
        z_exps_grad_true[-1].append(-((-1.0)**s) * math.pi * sin_pi_p)
    e_counts = tf.cast(tf.expand_dims(counts, 1), tf.float32)
    total_counts = tf.cast(tf.reduce_sum(counts), tf.float32)
    x_exps_true_reduced = tf.reduce_sum(x_exps_true * e_counts,
                                        0) / total_counts
    x_exps_grad_true_reduced = tf.reduce_sum(x_exps_grad_true * e_counts,
                                             0) / total_counts
    y_exps_true_reduced = tf.reduce_sum(y_exps_true * e_counts,
                                        0) / total_counts
    y_exps_grad_true_reduced = tf.reduce_sum(y_exps_grad_true * e_counts,
                                             0) / total_counts
    z_exps_true_reduced = tf.reduce_sum(z_exps_true * e_counts,
                                        0) / total_counts
    z_exps_grad_true_reduced = tf.reduce_sum(z_exps_grad_true * e_counts,
                                             0) / total_counts

    # Measure operators on every qubit.
    x_ops = tfq.convert_to_tensor([1 * cirq.X(q) for q in self.raw_qubits])
    y_ops = tfq.convert_to_tensor([1 * cirq.Y(q) for q in self.raw_qubits])
    z_ops = tfq.convert_to_tensor([1 * cirq.Z(q) for q in self.raw_qubits])

    # Check with reduce True (this is the default)
    with tf.GradientTape(persistent=True) as tape:
      x_exps_test = p_qnn.expectation(bitstrings, counts, x_ops)
      y_exps_test = p_qnn.expectation(bitstrings, counts, y_ops)
      z_exps_test = p_qnn.expectation(bitstrings, counts, z_ops)
    x_exps_grad_test = tf.squeeze(tape.jacobian(x_exps_test, p_qnn.values))
    y_exps_grad_test = tf.squeeze(tape.jacobian(y_exps_test, p_qnn.values))
    z_exps_grad_test = tf.squeeze(tape.jacobian(z_exps_test, p_qnn.values))
    del (tape)
    self.assertAllClose(x_exps_test, x_exps_true_reduced, atol=ATOL)
    self.assertAllClose(
        x_exps_grad_test, x_exps_grad_true_reduced, atol=GRAD_ATOL)
    self.assertAllClose(y_exps_test, y_exps_true_reduced, atol=ATOL)
    self.assertAllClose(
        y_exps_grad_test, y_exps_grad_true_reduced, atol=GRAD_ATOL)
    self.assertAllClose(z_exps_test, z_exps_true_reduced, atol=ATOL)
    self.assertAllClose(
        z_exps_grad_test, z_exps_grad_true_reduced, atol=GRAD_ATOL)

    # Check with reduce False
    with tf.GradientTape(persistent=True) as tape:
      x_exps_test = p_qnn.expectation(bitstrings, counts, x_ops, reduce=False)
      y_exps_test = p_qnn.expectation(bitstrings, counts, y_ops, reduce=False)
      z_exps_test = p_qnn.expectation(bitstrings, counts, z_ops, reduce=False)
    x_exps_grad_test = tf.squeeze(tape.jacobian(x_exps_test, p_qnn.values))
    y_exps_grad_test = tf.squeeze(tape.jacobian(y_exps_test, p_qnn.values))
    z_exps_grad_test = tf.squeeze(tape.jacobian(z_exps_test, p_qnn.values))
    self.assertAllClose(x_exps_test, x_exps_true, atol=ATOL)
    self.assertAllClose(x_exps_test, x_exps_grad_true, atol=GRAD_ATOL)
    self.assertAllClose(y_exps_test, y_exps_true, atol=ATOL)
    self.assertAllClose(y_exps_grad_test, y_exps_grad_true, atol=GRAD_ATOL)
    self.assertAllClose(z_exps_test, z_exps_true, atol=ATOL)
    self.assertAllClose(z_exps_grad_test, z_exps_grad_true, atol=GRAD_ATOL)

  def test_pulled_back_circuits(self):
    """Confirms the pulled back circuits correct for a variety of inputs."""
    num_data_states = 100
    data_states, _ = tfq.util.random_circuit_resolver_batch(
        self.raw_qubits, num_data_states)
    data_states_t = tfq.convert_to_tensor(data_states)
    test_qnn = qnn.QNN(
        self.pqc,
        name=self.name,
    )
    test_circuits = test_qnn.pulled_back_circuits(data_states_t, resolve=True)
    test_circuits_deser = tfq.from_tensor(test_circuits)

    resolved_inverse_pqc = tfq.from_tensor(
        test_qnn.inverse_pqc(resolve=True))[0]
    combined = tfq.from_tensor(
        tfq.convert_to_tensor([d + resolved_inverse_pqc for d in data_states]))
    for expected, test in zip(combined, test_circuits_deser):
      self.assertTrue(cirq.approx_eq(expected, test))

  def test_pulled_back_sample_basic(self):
    """Confirms correct pulled back sampling from GHZ QNN.

    The state preparation circuit for GHZ is not equal to its inverse,
    so it tests that the dagger is taken correctly before appending.
    """
    ghz_param = sympy.Symbol("ghz")
    ghz_circuit = cirq.Circuit(cirq.X(
        self.raw_qubits[0])**ghz_param) + cirq.Circuit(
            cirq.CNOT(q0, q1)
            for q0, q1 in zip(self.raw_qubits, self.raw_qubits[1:]))
    ghz_qnn = qnn.QNN(
        ghz_circuit,
        initializer=tf.keras.initializers.Constant(value=0.5),
        name="ghz")
    flip_circuits = [cirq.Circuit(), cirq.Circuit(cirq.X(self.raw_qubits[0]))]
    flip_circuits_t = tfq.convert_to_tensor(flip_circuits)
    counts = tf.random.uniform([len(flip_circuits)],
                               minval=10,
                               maxval=100,
                               dtype=tf.int32)

    test_samples = ghz_qnn.pulled_back_sample(
        flip_circuits_t, counts, reduce=False, unique=False)
    # The first circuit leaves only the Hadamard to superpose the first qubit
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0] * self.num_qubits, dtype=tf.int8),
            test_samples[0].to_tensor()))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1] + [0] * (self.num_qubits - 1), dtype=tf.int8),
            test_samples[0].to_tensor()))
    # The second circuit causes an additional bit flip
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0] + [1] + [0] * (self.num_qubits - 2), dtype=tf.int8),
            test_samples[1].to_tensor()))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1, 1] + [0] * (self.num_qubits - 2), dtype=tf.int8),
            test_samples[1].to_tensor()))

  def test_pulled_back_expectation(self):
    """Confirms correct pulled back measurement."""
    #TODO(zaqqwerty)
    pass

  def test_trainable_variables(self):
    test_qnn = qnn.QNN(
        self.pqc,
        backend=self.backend,
        differentiator=self.differentiator,
        is_analytic=True,
        name=self.name)

    self.assertAllEqual(test_qnn.values, test_qnn.trainable_variables[0])

    values = tf.random.uniform(tf.shape(test_qnn.trainable_variables[0]))
    test_qnn.trainable_variables = [values]
    self.assertAllEqual(values, test_qnn.trainable_variables[0])

    values = tf.Variable(values)
    test_qnn.trainable_variables = [values]
    self.assertAllEqual(values, test_qnn.trainable_variables[0])


if __name__ == "__main__":
  logging.info("Running qnn_test.py ...")
  tf.test.main()
