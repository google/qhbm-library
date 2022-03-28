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
"""Tests for qhbmlib.models.circuit"""

import absl
import itertools
import random
import string

import cirq
import sympy
import tensorflow as tf
from tensorflow.python.framework import errors
import tensorflow_quantum as tfq
from tensorflow_quantum.python import util as tfq_util

from qhbmlib import models
from qhbmlib import utils
from tests import test_util


class QuantumCircuitTest(tf.test.TestCase):
  """Tests the QuantumCircuit class."""

  def setUp(self):
    """Initializes test objects."""
    super().setUp()
    self.num_qubits = 5
    self.expected_qubits = cirq.GridQubit.rect(1, self.num_qubits)
    self.raw_symbols = [sympy.Symbol("s0"), sympy.Symbol("s1")]
    self.expected_symbol_names = tf.constant([str(s) for s in self.raw_symbols])
    self.raw_pqc = cirq.Circuit()
    for s in self.raw_symbols:
      for q in self.expected_qubits:
        self.raw_pqc += cirq.X(q)**s
        self.raw_pqc += cirq.Z(q)**s
    self.expected_pqc = tfq.convert_to_tensor([self.raw_pqc])
    init_const = tf.random.uniform([1, 42], dtype=tf.float32)
    self.expected_value_layers_inputs = [tf.Variable(init_const)]
    value_layer_0 = tf.keras.layers.Dense(5)
    value_layer_1 = tf.keras.layers.Dense(len(self.raw_symbols))
    value_layer_2 = utils.Squeeze(0)
    self.expected_value_layers = [[value_layer_0, value_layer_1, value_layer_2]]
    self.expected_symbol_values = value_layer_2(
        value_layer_1(value_layer_0(self.expected_value_layers_inputs[0])))
    self.expected_name = "TestOE"
    self.actual_layer = models.QuantumCircuit(self.expected_pqc,
                                              self.expected_qubits,
                                              self.expected_symbol_names,
                                              self.expected_value_layers_inputs,
                                              self.expected_value_layers,
                                              self.expected_name)

  def test_init(self):
    """Tests initialization."""
    self.assertAllEqual(self.actual_layer.qubits, self.expected_qubits)
    self.assertAllEqual(self.actual_layer.symbol_names,
                        self.expected_symbol_names)
    self.assertAllClose(self.actual_layer.value_layers_inputs,
                        self.expected_value_layers_inputs)
    self.assertAllEqual(self.actual_layer.value_layers,
                        self.expected_value_layers)
    self.assertAllEqual(self.actual_layer.symbol_values,
                        self.expected_symbol_values)
    self.assertAllEqual(
        tfq.from_tensor(self.actual_layer.pqc),
        tfq.from_tensor(self.expected_pqc))
    self.assertEqual(self.actual_layer.name, self.expected_name)

  @test_util.eager_mode_toggle
  def test_call(self):
    """Confirms calling on bitstrings yields correct circuits."""
    inputs = tf.cast(
        tf.random.shuffle(
            list(itertools.product([0, 1], repeat=self.num_qubits))), tf.int8)
    bit_injectors = []
    for b in inputs.numpy().tolist():
      bit_injectors.append(
          cirq.Circuit(
              cirq.X(q)**b_i for q, b_i in zip(self.expected_qubits, b)))
    combined = [b + self.raw_pqc for b in bit_injectors]
    expected_outputs = tfq.convert_to_tensor(combined)

    @tf.function
    def wrapper(inputs):
      return self.actual_layer(inputs)

    actual_outputs = wrapper(inputs)
    self.assertAllEqual(
        tfq.from_tensor(actual_outputs), tfq.from_tensor(expected_outputs))

  def test_add(self):
    """Confirms addition of QuantumCircuits works correctly."""
    other_qubits = cirq.GridQubit.rect(1, 2 * self.num_qubits)
    expected_qubits = other_qubits  # since is superset of self.expected_qubits
    other_symbols = [sympy.Symbol("other_symbol")]
    expected_symbol_names = tf.constant(
        [str(s) for s in self.raw_symbols + other_symbols])
    other_pqc = cirq.Circuit()
    for s in other_symbols:
      for q in other_qubits:
        other_pqc += cirq.Y(q)**s
    raw_expected_pqc = self.raw_pqc + other_pqc
    expected_pqc = tfq.convert_to_tensor([raw_expected_pqc])
    other_value_layers_inputs = [[
        tf.Variable(tf.random.uniform([1], dtype=tf.float32))
    ]]
    expected_value_layers_inputs = (
        self.expected_value_layers_inputs + other_value_layers_inputs)
    other_value_layers = [[utils.Squeeze(0)]]
    expected_value_layers = self.expected_value_layers + other_value_layers
    expected_symbol_values = tf.concat(
        [self.expected_symbol_values, other_value_layers_inputs[0][0]], 0)
    other_name = "the_other_layer"
    expected_name = self.expected_name + "_" + other_name
    other_layer = models.QuantumCircuit(
        tfq.convert_to_tensor([other_pqc]), other_pqc.all_qubits(),
        tf.constant([str(s) for s in other_symbols]), other_value_layers_inputs,
        other_value_layers, other_name)
    actual_add = self.actual_layer + other_layer
    self.assertAllEqual(actual_add.qubits, expected_qubits)
    self.assertAllEqual(actual_add.symbol_names, expected_symbol_names)
    self.assertAllClose(actual_add.value_layers_inputs,
                        expected_value_layers_inputs)
    self.assertAllEqual(actual_add.value_layers, expected_value_layers)
    self.assertAllEqual(actual_add.symbol_values, expected_symbol_values)
    self.assertAllEqual(
        tfq.from_tensor(actual_add.pqc), tfq.from_tensor(expected_pqc))
    self.assertEqual(actual_add.name, expected_name)

    # Confirm that tf.Variables in the sum are the same as in the addends.
    var_1 = tf.Variable([5.0])
    pqc_1 = cirq.Circuit(cirq.X(cirq.GridQubit(0, 0))**sympy.Symbol("a"))
    qnn_1 = models.QuantumCircuit(
        tfq.convert_to_tensor([pqc_1]), pqc_1.all_qubits(), tf.constant(["a"]),
        [var_1], [[]])
    var_2 = tf.Variable([-7.0])
    pqc_2 = cirq.Circuit(cirq.X(cirq.GridQubit(0, 0))**sympy.Symbol("b"))
    qnn_2 = models.QuantumCircuit(
        tfq.convert_to_tensor([pqc_2]), pqc_2.all_qubits(), tf.constant(["b"]),
        [var_2], [[]])
    actual_sum = qnn_1 + qnn_2
    # modify individual variables and confirm changes in the sum
    var_1.assign([-3.0])
    var_2.assign([12.0])
    self.assertAllClose(actual_sum.trainable_variables[0], var_1)
    self.assertAllClose(actual_sum.trainable_variables[1], var_2)

  @test_util.eager_mode_toggle
  def test_trace_add(self):
    """Confirm addition works under tf.function tracing."""
    var_1 = tf.Variable([5.0])
    pqc_1 = cirq.Circuit(cirq.X(cirq.GridQubit(0, 0))**sympy.Symbol("a"))
    qnn_1 = models.QuantumCircuit(
        tfq.convert_to_tensor([pqc_1]), pqc_1.all_qubits(), tf.constant(["a"]),
        [var_1], [[]])
    var_2 = tf.Variable([-7.0])
    pqc_2 = cirq.Circuit(cirq.X(cirq.GridQubit(0, 0))**sympy.Symbol("b"))
    qnn_2 = models.QuantumCircuit(
        tfq.convert_to_tensor([pqc_2]), pqc_2.all_qubits(), tf.constant(["b"]),
        [var_2], [[]])

    @tf.function
    def add_traced(qnn_a, qnn_b):
      actual_sum = qnn_a + qnn_b
      return actual_sum.pqc

    actual_sum_pqc = add_traced(qnn_1, qnn_2)
    expected_sum_pqc = tfq.from_tensor(tfq.convert_to_tensor([pqc_1 + pqc_2]))
    self.assertAllEqual(tfq.from_tensor(actual_sum_pqc), expected_sum_pqc)

  def test_add_error(self):
    """Confirms bad inputs to __add__ are rejected."""
    qubit_1 = cirq.GridQubit(0, 0)
    qubit_2 = cirq.GridQubit(1, 0)
    shared_symbol_name = "the_same_symbol"
    symbol_names_1 = ["first", "test_1", shared_symbol_name]
    symbol_names_2 = ["second", shared_symbol_name, "something"]
    symbols_1 = [sympy.Symbol(s) for s in symbol_names_1]
    symbols_2 = [sympy.Symbol(s) for s in symbol_names_2]
    pqc_1 = cirq.Circuit([cirq.X(qubit_1)**s for s in symbols_1])
    pqc_2 = cirq.Circuit([cirq.Y(qubit_2)**s for s in symbols_2])
    qnn_1 = models.QuantumCircuit(
        tfq.convert_to_tensor([pqc_1]), pqc_1.all_qubits(),
        tf.constant([str(s) for s in symbol_names_1]),
        [tf.random.uniform([len(symbol_names_1)], dtype=tf.float32)], [[]])
    qnn_2 = models.QuantumCircuit(
        tfq.convert_to_tensor([pqc_2]), pqc_2.all_qubits(),
        tf.constant([str(s) for s in symbol_names_2]),
        [tf.random.uniform([len(symbol_names_2)], dtype=tf.float32)], [[]])
    with self.assertRaises(TypeError):
      _ = qnn_1 + 1
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        expected_regex="must not have symbols in common"):
      _ = qnn_1 + qnn_2


# TODO(#130)

  def test_pow(self):
    """Confirms inversion of QuantumCircuit works correctly."""
    actual_inverse = self.actual_layer**-1
    expected_pqc = tfq.convert_to_tensor([self.raw_pqc**-1])
    self.assertNotAllEqual(
        tfq.from_tensor(self.actual_layer.pqc),
        tfq.from_tensor(actual_inverse.pqc))
    self.assertAllEqual(
        tfq.from_tensor(actual_inverse.pqc), tfq.from_tensor(expected_pqc))

    # Confirm that tf.Variables in the inverse are the same as self.
    var_1 = tf.Variable([2.5])
    pqc_1 = cirq.Circuit(cirq.X(cirq.GridQubit(0, 0))**sympy.Symbol("a"))
    qnn_1 = models.QuantumCircuit(
        tfq.convert_to_tensor([pqc_1]), pqc_1.all_qubits(), tf.constant(["a"]),
        [var_1], [[]])
    actual_inverse = qnn_1**-1
    var_1.assign([-3.0])
    self.assertAllClose(actual_inverse.trainable_variables[0], var_1)

  def test_pow_error(self):
    """Confirms bad inputs to __pow__ are rejected."""
    with self.assertRaisesRegex(ValueError, expected_regex="Only the inverse"):
      _ = self.actual_layer**2


class DirectQuantumCircuitTest(tf.test.TestCase):
  """Tests the DirectQuantumCircuit class."""

  def test_init(self):
    """Tests initialization."""
    raw_symbol_names = ["test_symbol", "s_2222", "where"]
    expected_symbol_names = tf.constant(sorted(raw_symbol_names))
    symbols = [sympy.Symbol(s) for s in raw_symbol_names]
    num_qubits = 5
    expected_qubits = cirq.GridQubit.rect(num_qubits, 1)
    expected_pqc = cirq.Circuit()
    for q in expected_qubits:
      for s in symbols:
        random_gate = random.choice([cirq.X, cirq.Y, cirq.Z])
        expected_pqc += random_gate(q)**s
    init_const = tf.random.uniform([len(raw_symbol_names)], dtype=tf.float32)
    expected_value_layers_inputs = [tf.Variable(init_const)]
    expected_value_layers = [[]]
    actual_qnn = models.DirectQuantumCircuit(
        expected_pqc, initializer=tf.keras.initializers.Constant(init_const))
    self.assertAllEqual(actual_qnn.qubits, expected_qubits)
    self.assertAllEqual(actual_qnn.symbol_names, expected_symbol_names)
    self.assertAllClose(actual_qnn.value_layers_inputs,
                        expected_value_layers_inputs)
    self.assertAllEqual(actual_qnn.value_layers, expected_value_layers)
    self.assertAllClose(actual_qnn.symbol_values,
                        expected_value_layers_inputs[0])
    self.assertAllEqual(
        tfq.from_tensor(actual_qnn.pqc),
        tfq.from_tensor(tfq.convert_to_tensor([expected_pqc])))

  def test_default_init(self):
    """Confirms default initializer sets values in expected range."""
    num_qubits = 10
    qubits = cirq.GridQubit.rect(num_qubits, 1)
    symbols = set()
    num_symbols = 100
    for _ in range(num_symbols):
      symbols.add("".join(random.sample(string.ascii_letters, 10)))
    pqc = tfq_util.random_symbol_circuit(qubits, symbols)
    actual_circuit = models.DirectQuantumCircuit(pqc)
    actual_circuit.build([])
    self.assertAllInRange(actual_circuit.symbol_values, 0, 2)


class QAIATest(tf.test.TestCase):
  """Tests the DirectQuantumCircuit class."""

  def test_init(self):
    """Tests initialization."""
    num_qubits = 3
    expected_qubits = cirq.GridQubit.rect(1, num_qubits)
    classical_h_terms = [
        cirq.Z(q0) * cirq.Z(q1)
        for q0, q1 in zip(expected_qubits, expected_qubits[1:])
    ]
    x_terms = cirq.PauliSum()
    y_terms = cirq.PauliSum()
    for q in expected_qubits:
      x_terms += cirq.X(q)
      y_terms += cirq.Y(q)
    quantum_h_terms = [x_terms, y_terms]

    num_layers = 2
    expected_symbol_names = []
    expected_pqc = cirq.Circuit()
    for p in range(num_layers):
      for k, q in enumerate(quantum_h_terms):
        symbol_name = f"gamma_{p}_{k}"
        expected_symbol_names.append(symbol_name)
        expected_pqc += tfq.util.exponential([q], [symbol_name])
      for k, c in enumerate(classical_h_terms):
        symbol_name = f"eta_{p}_{k}"
        expected_symbol_names.append(symbol_name)
        expected_pqc += tfq.util.exponential([c], [symbol_name])

    eta_const = random.uniform(-1, 1)
    theta_const = random.uniform(-1, 1)
    gamma_const = random.uniform(-1, 1)
    expected_value_layers_inputs = [[
        tf.Variable([eta_const] * num_layers),
        tf.Variable([theta_const] * len(classical_h_terms)),
        tf.Variable([[gamma_const] * len(quantum_h_terms)] * num_layers)
    ]]
    expected_symbol_values = []
    for _ in range(num_layers):
      expected_symbol_values += (
          ([eta_const * theta_const] * len(classical_h_terms)) +
          ([gamma_const] * len(quantum_h_terms)))
    actual_qnn = models.QAIA(quantum_h_terms, classical_h_terms, num_layers)
    self.assertAllEqual(actual_qnn.qubits, expected_qubits)
    self.assertAllEqual(actual_qnn.symbol_names, expected_symbol_names)
    self.assertAllEqual(
        tfq.from_tensor(actual_qnn.pqc),
        tfq.from_tensor(tfq.convert_to_tensor([expected_pqc])))

    actual_qnn.value_layers_inputs[0][0].assign([eta_const] * num_layers)
    actual_qnn.value_layers_inputs[0][1].assign([theta_const] *
                                                len(classical_h_terms))
    actual_qnn.value_layers_inputs[0][2].assign(
        [[gamma_const] * len(quantum_h_terms)] * num_layers)
    self.assertAllClose(actual_qnn.value_layers_inputs,
                        expected_value_layers_inputs)
    self.assertAllClose(actual_qnn.symbol_values, expected_symbol_values)


if __name__ == "__main__":
  absl.logging.info("Running circuit_test.py ...")
  tf.test.main()
