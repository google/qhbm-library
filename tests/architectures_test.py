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
"""Tests of the architectures module."""

from absl.testing import parameterized
import random

import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib import architectures


class RPQCTest(tf.test.TestCase, parameterized.TestCase):
  """Test RPQC functions in the architectures module."""

  def test_get_xz_rotation(self):
    """Confirm an XZ rotation is returned."""
    q = cirq.GridQubit(7, 9)
    a, b = sympy.symbols("a b")
    expected_circuit = cirq.Circuit(cirq.X(q)**a, cirq.Z(q)**b)
    actual_circuit = architectures.get_xz_rotation(q, a, b)
    self.assertEqual(actual_circuit, expected_circuit)

  def test_get_cz_exp(self):
    """Confirm an exponentiated CNOT is returned."""
    q0 = cirq.GridQubit(4, 1)
    q1 = cirq.GridQubit(2, 5)
    a = sympy.Symbol("a")
    expected_circuit = cirq.Circuit(cirq.CZ(q0, q1)**a)
    actual_circuit = architectures.get_cz_exp(q0, q1, a)
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
    actual_circuit = architectures.get_xz_rotation_layer(
        qubits, layer_num, name)
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
    actual_circuit = architectures.get_cz_exp_layer(qubits, layer_num, name)
    self.assertEqual(actual_circuit, expected_circuit)

  @parameterized.parameters([{"n_qubits": 11}, {"n_qubits": 12}])
  def test_get_hardware_efficient_model_unitary(self, n_qubits):
    """Confirm a multi-layered circuit is returned."""
    qubits = cirq.GridQubit.rect(1, n_qubits)
    name = "test_hardware_efficient_model"
    expected_circuit = cirq.Circuit()
    this_circuit = architectures.get_xz_rotation_layer(qubits, 0, name)
    expected_circuit += this_circuit
    this_circuit = architectures.get_cz_exp_layer(qubits, 0, name)
    expected_circuit += this_circuit
    this_circuit = architectures.get_xz_rotation_layer(qubits, 1, name)
    expected_circuit += this_circuit
    this_circuit = architectures.get_cz_exp_layer(qubits, 1, name)
    expected_circuit += this_circuit
    actual_circuit = architectures.get_hardware_efficient_model_unitary(
        qubits, 2, name)
    self.assertEqual(actual_circuit, expected_circuit)

  def test_get_hardware_efficient_model_unitary_1q(self):
    """Confirm the correct model is returned when there is only one qubit."""
    qubits = [cirq.GridQubit(2, 3)]
    name = "test_harware_efficient_model_1q"
    expected_circuit = cirq.Circuit()
    this_circuit = architectures.get_xz_rotation_layer(qubits, 0, name)
    expected_circuit += this_circuit
    this_circuit = architectures.get_xz_rotation_layer(qubits, 1, name)
    expected_circuit += this_circuit
    actual_circuit = architectures.get_hardware_efficient_model_unitary(
        qubits, 2, name)
    self.assertEqual(actual_circuit, expected_circuit)


class HEA2dTest(tf.test.TestCase):
  """Test 2D HEA functions in the architectures module."""

  def test_get_2d_xz_rotation_layer(self):
    """Confirms the xz rotations are correct on a 2x3 grid."""
    rows = 2
    cols = 3
    name = "test_xz"
    layer_num = 7
    expected_circuit = cirq.Circuit()
    for r in range(rows):
      for c in range(cols):
        q = cirq.GridQubit(r, c)
        s = sympy.Symbol(f"sx_{name}_{layer_num}_{r}_{c}")
        x_gate = cirq.X(q)**s
        s = sympy.Symbol(f"sz_{name}_{layer_num}_{r}_{c}")
        z_gate = cirq.Z(q)**s
        expected_circuit += cirq.Circuit(x_gate, z_gate)
    actual_circuit = architectures.get_2d_xz_rotation_layer(
        rows, cols, layer_num, name)
    self.assertEqual(actual_circuit, expected_circuit)

  def test_get_2d_xz_rotation_layer_small(self):
    """Confirms the xz rotation layer on one qubit is just a single xz."""
    name = "test_small_xz"
    layer_num = 29
    expected_circuit = cirq.Circuit()
    q = cirq.GridQubit(0, 0)
    s = sympy.Symbol(f"sx_{name}_{layer_num}_{0}_{0}")
    x_gate = cirq.X(q)**s
    s = sympy.Symbol(f"sz_{name}_{layer_num}_{0}_{0}")
    z_gate = cirq.Z(q)**s
    expected_circuit += cirq.Circuit(x_gate, z_gate)
    actual_circuit = architectures.get_2d_xz_rotation_layer(
        1, 1, layer_num, name)
    self.assertEqual(actual_circuit, expected_circuit)

  def test_get_2d_cz_exp_layer(self):
    """Confirms the cz exponentials are correct on a 2x3 grid."""
    name = "test_cz"
    layer_num = 19
    expected_circuit = cirq.Circuit()

    # Apply horizontal bonds
    s = sympy.Symbol(f"scz_{name}_{layer_num}_row{0}_{0}_{1}")
    expected_circuit += cirq.Circuit(
        cirq.CZPowGate(exponent=s)(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)))
    s = sympy.Symbol(f"scz_{name}_{layer_num}_row{0}_{1}_{2}")
    expected_circuit += cirq.Circuit(
        cirq.CZPowGate(exponent=s)(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)))
    s = sympy.Symbol(f"scz_{name}_{layer_num}_row{1}_{0}_{1}")
    expected_circuit += cirq.Circuit(
        cirq.CZPowGate(exponent=s)(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)))
    s = sympy.Symbol(f"scz_{name}_{layer_num}_row{1}_{1}_{2}")
    expected_circuit += cirq.Circuit(
        cirq.CZPowGate(exponent=s)(cirq.GridQubit(1, 1), cirq.GridQubit(1, 2)))

    # Apply vertical bonds
    s = sympy.Symbol(f"scz_{name}_{layer_num}_col{0}_{0}_{1}")
    expected_circuit += cirq.Circuit(
        cirq.CZPowGate(exponent=s)(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)))
    s = sympy.Symbol(f"scz_{name}_{layer_num}_col{1}_{0}_{1}")
    expected_circuit += cirq.Circuit(
        cirq.CZPowGate(exponent=s)(cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)))
    s = sympy.Symbol(f"scz_{name}_{layer_num}_col{2}_{0}_{1}")
    expected_circuit += cirq.Circuit(
        cirq.CZPowGate(exponent=s)(cirq.GridQubit(0, 2), cirq.GridQubit(1, 2)))

    actual_circuit = architectures.get_2d_cz_exp_layer(2, 3, layer_num, name)
    self.assertEqual(actual_circuit, expected_circuit)

  def test_get_2d_cz_exp_layer_empty(self):
    """On single qubit, no gates should be returned."""
    actual_circuit = architectures.get_2d_cz_exp_layer(1, 1, 1, "")
    self.assertEqual(actual_circuit, cirq.Circuit())

  def test_get_2d_cz_exp_layer_small(self):
    """Tests on 2 qubits."""
    name = "small"
    layer_num = 51
    s = sympy.Symbol(f"scz_{name}_{layer_num}_row{0}_{0}_{1}")
    expected_circuit = cirq.Circuit(
        cirq.CZPowGate(exponent=s)(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)))
    actual_circuit = architectures.get_2d_cz_exp_layer(1, 2, layer_num, name)
    self.assertEqual(actual_circuit, expected_circuit)

  def test_get_2d_hea(self):
    """Confirms the hea is correct on a 2x3 grid."""
    num_layers = 2
    name = "test_hea"
    expected_circuit = cirq.Circuit()
    for layer in range(num_layers):
      xz_circuit = architectures.get_2d_xz_rotation_layer(2, 3, layer, name)
      cz_circuit = architectures.get_2d_cz_exp_layer(2, 3, layer, name)
      expected_circuit += xz_circuit
      expected_circuit += cz_circuit
    actual_circuit = architectures.get_2d_hea(2, 3, 2, name)
    self.assertEqual(actual_circuit, expected_circuit)


class TrotterTest(tf.test.TestCase, parameterized.TestCase):
  """Test trotter functions in the architectures module."""

  def test_get_trotter_model_unitary(self):
    """Confirm correct trotter unitary and parameters are returned."""
    n_qubits = 4
    qubits = cirq.GridQubit.rect(1, n_qubits)
    p = 7
    hz = cirq.PauliSum()
    hx = cirq.PauliSum()
    test_name = "test_trotter"
    for q in qubits:
      hz += cirq.PauliString(random.uniform(-4.5, 4.5), cirq.Z(q))
      hx += cirq.PauliString(cirq.X(q))
    for q0, q1 in zip(qubits[:-1], qubits[1:]):
      hz += cirq.PauliString(random.uniform(-4.5, 4.5), cirq.Z(q0), cirq.Z(q1))
    gammas = [
        sympy.Symbol("phi_test_trotter_L{}_H0".format(j)) for j in range(p)
    ]
    betas = [
        sympy.Symbol("phi_test_trotter_L{}_H1".format(j)) for j in range(p)
    ]
    expected_circuit = cirq.Circuit()
    x_circuit = cirq.PauliSum()
    for q in qubits:
      x_circuit += cirq.X(q)
    for j in range(p):
      expected_circuit += tfq.util.exponential([hz], coefficients=[gammas[j]])
      expected_circuit += tfq.util.exponential([x_circuit],
                                               coefficients=[betas[j]])
    actual_circuit = architectures.get_trotter_model_unitary(
        p, [hz, hx], test_name)
    self.assertEqual(actual_circuit, expected_circuit)
