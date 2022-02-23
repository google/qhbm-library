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
"""Tests for the test_util module."""

from absl.testing import parameterized

import cirq
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


class EagerModeToggle(tf.test.TestCase):
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
