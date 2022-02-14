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

import itertools
from absl import logging

import cirq
import math
import sympy
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_quantum as tfq

from qhbmlib import circuit_infer
from qhbmlib import circuit_model
from qhbmlib import utils
from tests import test_util

# Global tolerance, set for float32.
ATOL = 1e-5
GRAD_ATOL = 2e-4


class QuantumInferenceTest(tf.test.TestCase):
  """Tests the QuantumInference class."""

  def setUp(self):
    """Initializes test objects."""
    super().setUp()

    # Build QNN representing X^p|s>
    self.num_qubits = 5
    self.raw_qubits = cirq.GridQubit.rect(1, self.num_qubits)
    p_param = sympy.Symbol("p")
    p_circuit = cirq.Circuit(cirq.X(q)**p_param for q in self.raw_qubits)
    self.p_qnn = circuit_model.DirectQuantumCircuit(
        p_circuit,
        initializer=tf.keras.initializers.RandomUniform(
            minval=-5.0, maxval=5.0),
        name="p_qnn")

  def test_init(self):
    """Confirms QuantumInference is initialized correctly."""
    expected_backend = "noiseless"
    expected_differentiator = None
    expected_name = "TestOE"
    actual_exp = circuit_infer.QuantumInference(
        backend=expected_backend,
        differentiator=expected_differentiator,
        name=expected_name)
    self.assertEqual(actual_exp.name, expected_name)
    self.assertEqual(actual_exp.backend, expected_backend)
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
    exp_infer = circuit_infer.QuantumInference()

    # Choose some bitstrings.
    num_bitstrings = 10
    bitstrings = tfp.distributions.Bernoulli(
        probs=[0.5] * self.num_qubits, dtype=tf.int8).sample(num_bitstrings)
    counts = tf.random.uniform([num_bitstrings], 1, 1000, tf.int32)

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
    for bits in bitstrings.numpy().tolist():
      for exps in expected + expected_grad:
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
    expected_grad_reduced = []
    for exps in expected:
      expected_reduced.append(tf.reduce_sum(exps * e_counts, 0) / total_counts)
    for exps in expected_grad:
      expected_grad_reduced.append(
          tf.reduce_sum(exps * e_counts, 0) / total_counts)
    expected_reduced = tf.stack(expected_reduced)
    expected_grad_reduced = tf.stack(expected_grad_reduced)

    # Measure operators on every qubit.
    x_ops = tfq.convert_to_tensor([1 * cirq.X(q) for q in self.raw_qubits])
    y_ops = tfq.convert_to_tensor([1 * cirq.Y(q) for q in self.raw_qubits])
    z_ops = tfq.convert_to_tensor([1 * cirq.Z(q) for q in self.raw_qubits])
    all_ops = [x_ops, y_ops, z_ops]

    expectation_wrapper = tf.function(exp_infer.expectation)
    actual_reduced = []
    actual_grad_reduced = []
    for op in all_ops:
      with tf.GradientTape() as tape:
        current_exp = expectation_wrapper(initial_states, op)
        reduced_exp = tf.math.reduce_mean(current_exp, 0)
      reduced_grad = tf.squeeze(
          tape.jacobian(reduced_exp, self.p_qnn.trainable_variables))
      actual_reduced.append(reduced_exp)
      actual_grad_reduced.append(reduced_grad)
    actual_reduced = tf.stack(actual_reduced)
    actual_grad_reduced = tf.stack(actual_grad_reduced)

    self.assertAllClose(actual_reduced, expected_reduced, atol=ATOL)
    self.assertAllClose(
        actual_grad_reduced, expected_grad_reduced, atol=GRAD_ATOL)


  @test_util.eager_mode_toggle
  def test_sample_basic(self):
    """Confirms correct sampling from identity, bit flip, and GHZ QNNs."""
    bitstrings = tf.constant(
        list(itertools.product([0, 1], repeat=self.num_qubits)), dtype=tf.int8)
    counts = tf.random.uniform([tf.shape(bitstrings)[0]], 10, 100, tf.int32)

    q_infer = circuit_infer.QuantumInference()

    @tf.function
    def sample_wrapper(qnn, bitstrings, counts):
      return q_infer.sample(qnn, bitstrings, counts)

    ident_qnn = circuit_model.DirectQuantumCircuit(
        cirq.Circuit(cirq.I(q) for q in self.raw_qubits), name="identity")
    test_samples = sample_wrapper(ident_qnn, bitstrings, counts)
    for i, (b, c) in enumerate(zip(bitstrings, counts)):
      self.assertEqual(tf.shape(test_samples[i].to_tensor())[0], c)
      for j in range(c):
        self.assertAllEqual(test_samples[i][j], b)

    flip_qnn = circuit_model.DirectQuantumCircuit(
        cirq.Circuit(cirq.X(q) for q in self.raw_qubits), name="flip")
    test_samples = sample_wrapper(flip_qnn, bitstrings, counts)
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
    test_samples = sample_wrapper(
        ghz_qnn,
        tf.expand_dims(tf.constant([0] * self.num_qubits, dtype=tf.int8), 0),
        tf.expand_dims(counts[0], 0))[0].to_tensor()
    # Both |0...0> and |1...1> should be among the measured bitstrings
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0] * self.num_qubits, dtype=tf.int8), test_samples))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1] * self.num_qubits, dtype=tf.int8), test_samples))

  @test_util.eager_mode_toggle
  def test_sample_uneven(self):
    """Check for discrepancy in samples when count entries differ."""
    max_counts = int(1e7)
    counts = tf.constant([max_counts // 2, max_counts])
    test_qnn = circuit_model.DirectQuantumCircuit(
        cirq.Circuit(cirq.H(cirq.GridQubit(0, 0))))
    test_infer = circuit_infer.QuantumInference()

    @tf.function
    def sample_wrapper(qnn, bitstrings, counts):
      return test_infer.sample(qnn, bitstrings, counts)

    bitstrings = tf.constant([[0], [0]], dtype=tf.int8)
    _, samples_counts = sample_wrapper(test_qnn, bitstrings, counts)
    # QNN samples should be half 0 and half 1.
    self.assertAllClose(
        samples_counts[0], samples_counts[1], atol=max_counts // 1000)


if __name__ == "__main__":
  logging.info("Running circuit_infer_test.py ...")
  tf.test.main()
