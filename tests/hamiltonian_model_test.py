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
"""Tests for the hamiltonian_model module."""

import absl
import itertools
import random

import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib import circuit_model
from qhbmlib import energy_model
from qhbmlib import hamiltonian_model
from qhbmlib import utils


class HamiltonianTest(tf.test.TestCase):
  """Tests the Hamiltonian class."""

  def setUp(self):
    """Initializes test objects."""
    super().setUp()
    self.expected_name = "this_IS_theTestHam42"
    self.num_bits = 3
    self.expected_energy = energy_model.BernoulliEnergy(list(range(self.num_bits)))
    self.expected_energy.build([None, self.num_bits])
    qubits = cirq.GridQubit.rect(1, self.num_bits)
    symbols = [sympy.Symbol(str(n)) for n in range(self.num_bits)]
    pqc = cirq.Circuit(cirq.X(q) ** s for q, s in zip(qubits, symbols)) 
    self.expected_circuit = circuit_model.DirectQuantumCircuit(pqc)
    self.expected_circuit.build([])
    self.actual_hamiltonian = hamiltonian_model.Hamiltonian(
        self.expected_energy,
        self.expected_circuit,
        self.expected_name)

  def test_init(self):
    """Tests Hamiltonian initialization.

    The first three equality checks are by python id.
    """
    self.assertEqual(self.actual_hamiltonian.name, self.expected_name)
    self.assertEqual(self.actual_hamiltonian.energy, self.expected_energy)
    self.assertEqual(self.actual_hamiltonian.circuit, self.expected_circuit)
    self.assertEqual(tfq.from_tensor(
      self.actual_hamiltonian.circuit_dagger.pqc), tfq.from_tensor((self.expected_circuit ** -1).pqc))
    expected_variables = self.expected_energy.trainable_variables + self.expected_circuit.trainable_variables
    self.assertNotEmpty(expected_variables)
    self.assertAllClose(self.actual_hamiltonian.trainable_variables, expected_variables)

  def test_init_error(self):
    """Confirms initialization fails for mismatched energy and circuit."""
    small_energy = energy_model.BernoulliEnergy(list(range(self.num_bits - 1)))
    with self.assertRaisesRegex(ValueError, expected_regex="same number of bits"):
      _ = hamiltonian_model.Hamiltonian(small_energy, self.expected_circuit)

  def test_add(self):
    """Tests Hamiltonian addition."""
    pass


class HamiltonianSumTest(tf.test.TestCase):
  """Tests the HamiltonianSum class."""

  def setUp(self):
    """Initializes test objects."""
    super().setUp()
    self.expected_name = "SumThing"
    self.num_bits_list = [2, 3]
    self.expected_energy_list = [energy_model.BernoulliEnergy(list(range(num_bits))) for num_bits in self.num_bits_list]
    for energy, num_bits in zip(self.expected_energy_list, self.num_bits_list):
      energy.build([None, num_bits])
    symbols_list = [[sympy.Symbol(str(n)) for n in range(num_bits)] for num_bits in self.num_bits_list]
    qubits_list = [cirq.GridQubit.rect(1, num_bits) for num_bits in self.num_bits_list]
    pqc_list = [cirq.Circuit(cirq.X(q) ** s for q, s in zip(qubits, symbols)) for qubits, symbols in zip(qubits_list, symbols_list)]
    self.expected_circuit_list = [circuit_model.DirectQuantumCircuit(pqc) for pqc in pqc_list]
    for circuit in self.expected_circuit_list:
      circuit.build([])
    self.expected_terms = [hamiltonian_model.Hamiltonian(e, c) for e, c in zip(self.expected_energy_list, self.expected_circuit_list)]
    self.actual_hamiltonian_sum = hamiltonian_model.HamiltonianSum(self.expected_terms, self.expected_name)

  def test_init(self):
    """Tests HamiltonianSum initialization."""
    self.assertEqual(self.actual_hamiltonian_sum.name, self.expected_name)
    expected_variables = []
    for e, c in zip(self.expected_energy_list, self.expected_circuit_list):
      expected_variables += e.trainable_variables
      expected_variables += c.trainable_variables
    self.assertNotEmpty(expected_variables)
    self.assertAllClose(self.actual_hamiltonian_sum.trainable_variables, expected_variables)

  def test_add(self):
    """Tests HamiltonianSum addition."""
    pass


if __name__ == "__main__":
  absl.logging.info("Running hamiltonian_model_test.py ...")
  tf.test.main()
