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
"""Tests for qhbmlib.models.hamiltonian"""

import absl

import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib import models


class HamiltonianTest(tf.test.TestCase):
  """Tests the Hamiltonian class."""

  def setUp(self):
    """Initializes test objects."""
    super().setUp()
    self.expected_name = "this_IS_theTestHam42"
    self.num_bits = 3
    self.expected_energy = models.BernoulliEnergy(list(range(self.num_bits)))
    self.expected_energy.build([None, self.num_bits])
    qubits = cirq.GridQubit.rect(1, self.num_bits)
    symbols = [sympy.Symbol(str(n)) for n in range(self.num_bits)]
    pqc = cirq.Circuit(cirq.X(q)**s for q, s in zip(qubits, symbols))
    self.expected_circuit = models.DirectQuantumCircuit(pqc)
    self.expected_circuit.build([])
    self.expected_operator_shards = self.expected_energy.operator_shards(
        self.expected_circuit.qubits)
    self.actual_hamiltonian = models.Hamiltonian(self.expected_energy,
                                                 self.expected_circuit,
                                                 self.expected_name)

  def test_init(self):
    """Tests Hamiltonian initialization.

    The first three equality checks are by python id.
    """
    self.assertEqual(self.actual_hamiltonian.name, self.expected_name)
    self.assertEqual(self.actual_hamiltonian.energy, self.expected_energy)
    self.assertEqual(self.actual_hamiltonian.circuit, self.expected_circuit)
    self.assertEqual(
        tfq.from_tensor(self.actual_hamiltonian.circuit_dagger.pqc),
        tfq.from_tensor((self.expected_circuit**-1).pqc))
    self.assertAllEqual(
        tfq.from_tensor(self.actual_hamiltonian.operator_shards),
        self.expected_operator_shards)
    expected_variables = (
        self.expected_energy.trainable_variables +
        self.expected_circuit.trainable_variables)
    self.assertNotEmpty(expected_variables)
    self.assertAllClose(self.actual_hamiltonian.trainable_variables,
                        expected_variables)

    # check None operator shards.
    pqc = cirq.Circuit(cirq.X(cirq.GridQubit(0, 0))**sympy.Symbol("a"))
    qnn = models.DirectQuantumCircuit(pqc)
    actual_energy = models.BitstringEnergy([1], [])
    actual_ham = models.Hamiltonian(actual_energy, qnn)
    self.assertIsNone(actual_ham.operator_shards)

  def test_init_error(self):
    """Confirms initialization fails for mismatched energy and circuit."""
    small_energy = models.BernoulliEnergy(list(range(self.num_bits - 1)))
    with self.assertRaisesRegex(
        ValueError, expected_regex="same number of bits"):
      _ = models.Hamiltonian(small_energy, self.expected_circuit)


if __name__ == "__main__":
  absl.logging.info("Running hamiltonian_test.py ...")
  tf.test.main()
