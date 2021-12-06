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

from qhbmlib import circuit_infer
from qhbmlib import circuit_model
from tests import test_util

# Global tolerance, set for float32.
ATOL = 1e-5
GRAD_ATOL = 2e-4


def _pystr(x):
  return [str(y) for y in x]


class BitstringInjectorTest(tf.test.TestCase):
  """Tests BitstringInjector."""

  def test_init(self):
    """Confirm correct bit injector circuit creation."""
    expected_qubits = [
        cirq.GridQubit(0, 2),
        cirq.GridQubit(1, 4),
        cirq.GridQubit(2, 2)
    ]
    expected_name = "build_bit_test"
    actual_bit_injector = circuit_infer.BitstringInjector(expected_qubits, expected_name)
    expected_symbols = list(
        sympy.symbols(
            "build_bit_test_bit_0 build_bit_test_bit_1 build_bit_test_bit_2"))
    expected_circuit = tfq.convert_to_tensor([cirq.Circuit(
        [cirq.X(q)**s for q, s in zip(my_qubits, expected_symbols)])])
    self.assertAllEqual(actual_bit_injector.qubits, expected_qubits)
    self.assertAllEqual(_pystr(actual_bit_injector.bit_symbols), _pystr(expected_symbols))
    self.assertEqual(tfq.from_tensor(actual_bit_injector.bit_circuit), tfq.from_tensor(expected_circuit))

  def test_inject_bitstrings(self):
    """Confirms correct combination of bits and circuits."""
    num_qubits = 5
    qubits = cirq.GridQubit.rect(1, num_qubits)
    actual_bit_injector = circuit_infer.BitstringInjector(qubits)

    test_pqcs, _ = tfq.util.random_circuit_resolver_batch(qubits, 1)
    test_qnn = circuit_model.DirectQuantumCircuit(test_pqcs[0])

    bitstrings = 2 * list(itertools.product([0, 1], repeat=num_qubits))
    bit_injectors = []
    for b in bitstrings:
      bit_injectors.append(
          cirq.Circuit(cirq.X(q)**b_i for q, b_i in zip(qubits, b)))
    expected_circuits = tfq.convert_to_tensor([b + test_pqcs[0] for b in bit_injectors])
    self.assertAllEqual(tfq.from_tensor(actual_circuits), tfq.from_tensor(expected_circuits))


class ExpectationTest(tf.test.TestCase):
  """Tests the Expectation class."""

  def test_init(self):
    """Confirms Expectation is initialized correctly."""
    expected_qubits = cirq.GridQubit.rect(1, 100)
    expected_backend = "noiseless"
    expected_differentiator = None
    expected_name = "TestOE"
    actual_exp = circuit_infer.Expectation(
        expected_qubits,
        backend=expected_backend,
        differentiator=expected_differentiator,
        name=expected_name)
    self.assertAllEqual(actual_exp.qubits, expected_qubits)
    self.assertEqual(actual_exp.name, expected_name)
    self.assertEqual(actual_exp.backend, expected_backend)
    self.assertEqual(actual_exp.differentiator, expected_differentiator)

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
    num_qubits = 5
    qubits = cirq.GridQubit.rect(1, num_qubits)
    exp_infer = circuit_infer.Expectation(qubits)
    
    # Build QNN representing X^p|s>
    p_param = sympy.Symbol("p")
    p_circuit = cirq.Circuit(cirq.X(q)**p_param for q in qubits)
    p_qnn = circuit_model.DirectQuantumCircuit(
        p_circuit,
        initializer=tf.keras.initializers.RandomUniform(
            minval=-5.0, maxval=5.0),
        name="p_qnn")

    # Choose some bitstrings.
    num_bitstrings = 10
    bitstrings_raw = []
    for _ in range(num_bitstrings):
      bitstrings_raw.append([])
      for _ in range(num_qubits):
        if random.choice([True, False]):
          bitstrings_raw[-1].append(0)
        else:
          bitstrings_raw[-1].append(1)
    bitstrings = tf.constant(bitstrings_raw, dtype=tf.int8)
    counts = tf.random.uniform(
        shape=[num_bitstrings], minval=1, maxval=1000, dtype=tf.int32)

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
    sin_pi_p = math.sin(math.pi * p_qnn.values[0])
    cos_pi_p = math.cos(math.pi * p_qnn.values[0])
    for bits in bitstrings_raw:
      for exps in expected + expected_grads:
        exps.append([])
      for s in bits:
        expected_x_exps[-1].append(0)
        expected_x_exps_grad[-1].append(0)
        expected_y_exps[-1].append(-((-1.0)**s) * sin_pi_p)
        expected_y_exps_grad[-1].append(-((-1.0)**s) * math.pi * cos_pi_p)
        expected_z_exps[-1].append(((-1.0)**s) * cos_pi_p)
        expected_z_exps_grad[-1].append(-((-1.0)**s) * math.pi * sin_pi_p)
    e_counts = tf.cast(tf.expand_dims(counts, 1), tf.float32)
    total_counts = tf.cast(tf.reduce_sum(counts), tf.float32)
    expected_reduced = []
    for exps in expected:
      expected_reduced.append(tf.reduce_sum(exps * e_counts, 0) / total_counts)

    # Measure operators on every qubit.
    x_ops = tfq.convert_to_tensor([1 * cirq.X(q) for q in self.raw_qubits])
    y_ops = tfq.convert_to_tensor([1 * cirq.Y(q) for q in self.raw_qubits])
    z_ops = tfq.convert_to_tensor([1 * cirq.Z(q) for q in self.raw_qubits])
    all_ops = [x_ops, y_ops, z_ops]

    # Check with reduce True (this is the default)
    with tf.GradientTape(persistent=True) as tape:
      actual_exps = []
      for op in ops:
        actual_exps.append(exp_infer.expectation(p_qnn, bitstrings, counts, op))
    actual_exps_grad = [tf.squeeze(tape.jacobian(exps, p_qnn.values)) for exps in actual_exps]
    del (tape)
    for a, e in zip(actual_exps, expected):
      self.assertAllClose(a, e, atol=ATOL)
    for a, e in zip(actual_exps_grad, expected_grad):
      self.assertAllClose(a, e, atol=GRAD_ATOL)

    # Check with reduce False
    with tf.GradientTape(persistent=True) as tape:
      x_exps_test = exp_infer.expectation(p_qnn, bitstrings, counts, x_ops, reduce=False)
      y_exps_test = exp_infer.expectation(p_qnn, bitstrings, counts, y_ops, reduce=False)
      z_exps_test = exp_infer.expectation(p_qnn, bitstrings, counts, z_ops, reduce=False)
    x_exps_grad_test = tf.squeeze(tape.jacobian(x_exps_test, p_qnn.values))
    y_exps_grad_test = tf.squeeze(tape.jacobian(y_exps_test, p_qnn.values))
    z_exps_grad_test = tf.squeeze(tape.jacobian(z_exps_test, p_qnn.values))
    self.assertAllClose(x_exps_test, x_exps_true, atol=ATOL)
    self.assertAllClose(x_exps_test, x_exps_grad_true, atol=GRAD_ATOL)
    self.assertAllClose(y_exps_test, y_exps_true, atol=ATOL)
    self.assertAllClose(y_exps_grad_test, y_exps_grad_true, atol=GRAD_ATOL)
    self.assertAllClose(z_exps_test, z_exps_true, atol=ATOL)
    self.assertAllClose(z_exps_grad_test, z_exps_grad_true, atol=GRAD_ATOL)


if __name__ == "__main__":
  logging.info("Running circuit_infer_test.py ...")
  tf.test.main()
