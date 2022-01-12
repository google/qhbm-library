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
"""Tests for the hamiltonian_infer module."""

import absl
import random
import string

import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
from tensorflow_quantum.python import util as tfq_util

from qhbmlib import circuit_infer
from qhbmlib import circuit_model
from qhbmlib import energy_infer
from qhbmlib import energy_model
from qhbmlib import hamiltonian_model
from qhbmlib import hamiltonian_infer
from tests import test_util


class QHBMTest(tf.test.TestCase):
  """Tests the QHBM class."""

  def setUp(self):
    """Initializes test objects."""
    super().setUp()

    # Model hamiltonian
    num_bits = 3
    self.energy = energy_model.BernoulliEnergy(list(range(num_bits)))
    self.energy.build([None, num_bits])
    # pin first and last bits, middle bit free.
    self.energy.set_weights([tf.constant([-23, 0, 17])])
    qubits = cirq.GridQubit.rect(1, num_bits)
    symbols = set()
    num_symbols = 20
    for _ in range(num_symbols):
      symbols.add("".join(random.sample(string.ascii_letters, 10)))
    self.pqc = tfq_util.random_symbol_circuit(qubits, symbols)
    circuit = circuit_model.DirectQuantumCircuit(self.pqc)
    circuit.build([])
    self.model = hamiltonian_model.Hamiltonian(self.energy, circuit)

    # Inference
    self.expected_e_inference = energy_infer.AnalyticEnergyInference(3)
    self.expected_q_inference = circuit_infer.QuantumInference()
    self.expected_name = "nameforaQHBM"
    self.actual_qhbm = hamiltonian_infer.QHBM(self.expected_e_inference,
                                              self.expected_q_inference,
                                              self.expected_name)

  def test_init(self):
    """Tests QHBM initialization."""
    self.assertEqual(self.actual_qhbm.e_inference, self.expected_e_inference)
    self.assertEqual(self.actual_qhbm.q_inference, self.expected_q_inference)
    self.assertEqual(self.actual_qhbm.name, self.expected_name)

  def test_circuits(self):
    """Confirms correct circuits are sampled."""
    num_samples = int(1e7)
    actual_circuits, actual_counts = self.actual_qhbm.circuits(
        self.model, num_samples)

    # Circuits with the allowed-to-be-sampled bitstrings prepended.
    u = tfq.from_tensor(self.model.circuit.pqc)[0]
    qubits = self.model.circuit.qubits
    expected_circuits_deser = [
        cirq.Circuit(
            cirq.X(qubits[0])**0,
            cirq.X(qubits[1])**0,
            cirq.X(qubits[2]),
        ) + u,
        cirq.Circuit(
            cirq.X(qubits[0])**0,
            cirq.X(qubits[1]),
            cirq.X(qubits[2]),
        ) + u,
    ]
    # Check that both circuits are generated.
    actual_circuits_deser = tfq.from_tensor(actual_circuits)
    self.assertTrue(
        any([
            expected_circuits_deser[0] == actual_circuits_deser[0],
            expected_circuits_deser[0] == actual_circuits_deser[1],
        ]))
    self.assertTrue(
        any([
            expected_circuits_deser[1] == actual_circuits_deser[0],
            expected_circuits_deser[1] == actual_circuits_deser[1],
        ]))
    # Check that the fraction is approximately 0.5 (equal counts)
    self.assertAllClose(
        actual_counts[0], actual_counts[1], atol=num_samples / 1000)

  def test_expectation_cirq(self):
    """Compares library expectation values to those from Cirq."""
    num_bits = 3
    qubits = cirq.GridQubit.rect(1, num_bits)
    raw_op = test_util.get_random_pauli_sum(qubits)
    ops = tfq.convert_to_tensor([raw_op])

    symbols = set()
    num_symbols = 10
    for _ in range(num_symbols):
      symbols.add("".join(random.sample(string.ascii_letters, 10)))
    symbols = list(symbols)
    circuits, resolvers = tfq.util.random_symbol_circuit_resolver_batch(
        qubits, symbols, 1)

    energy = energy_model.BernoulliEnergy(list(range(num_bits)))
    energy.build([None, num_bits])
    circuit = circuit_model.QuantumCircuit(
        circuits[0], tf.constant(symbols), [tf.Variable([resolvers[0][s] for s in symbols])], [[]])
    circuit.build([])
    actual_hamiltonian = hamiltonian_model.Hamiltonian(energy, circuit)

    e_infer = energy_infer.BernoulliEnergyInference()
    q_infer = circuit_infer.QuantumInference()
    actual_h_infer = hamiltonian_infer.QHBM(e_infer, q_infer)

    num_samples = 1e7
    actual_expectation = actual_h_infer.expectation(
        actual_hamiltonian, ops, num_samples)

    total_circuits = 
    expected_expectation = cirq.Simulator().simulate_expectation_values(
        circuits[0], [raw_op], resolvers[0])
    
    self.assertAllClose(actual_expectation, expected_expectation)

  def test_sample(self):
    """Compares library samples to those from Cirq."""
    pass


if __name__ == "__main__":
  absl.logging.info("Running hamiltonian_infer_test.py ...")
  tf.test.main()
