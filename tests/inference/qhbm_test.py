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
"""Tests for qhbmlib.inference.qhbm"""

import absl
from absl.testing import parameterized
import random
import string

import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
from tensorflow_quantum.python import util as tfq_util

from qhbmlib import inference
from qhbmlib import models
from qhbmlib.models import energy
from qhbmlib.models import hamiltonian
from qhbmlib import utils
from tests import test_util


class QHBMTest(parameterized.TestCase, tf.test.TestCase):
  """Tests the QHBM class."""

  def setUp(self):
    """Initializes test objects."""
    super().setUp()
    self.num_samples = int(1e7)

    # Model hamiltonian
    num_bits = 3
    self.actual_energy = models.BernoulliEnergy(list(range(num_bits)))
    self.expected_ebm = inference.AnalyticEnergyInference(
        self.actual_energy, self.num_samples)
    # pin first and last bits, middle bit free.
    self.actual_energy.set_weights([tf.constant([-23, 0, 17])])
    qubits = cirq.GridQubit.rect(1, num_bits)
    symbols = set()
    num_symbols = 20
    for _ in range(num_symbols):
      symbols.add("".join(random.sample(string.ascii_letters, 10)))
    self.pqc = tfq_util.random_symbol_circuit(qubits, symbols)
    actual_circuit = models.DirectQuantumCircuit(self.pqc)
    self.expected_qnn = inference.AnalyticQuantumInference(actual_circuit)

    # Inference
    self.expected_name = "nameforaQHBM"
    self.actual_qhbm = inference.QHBM(self.expected_ebm, self.expected_qnn,
                                      self.expected_name)

    self.tfp_seed = tf.constant([5, 1], tf.int32)

  def test_init(self):
    """Tests QHBM initialization."""
    self.assertEqual(self.actual_qhbm.e_inference, self.expected_ebm)
    self.assertEqual(self.actual_qhbm.q_inference, self.expected_qnn)
    self.assertEqual(self.actual_qhbm.name, self.expected_name)

  @test_util.eager_mode_toggle
  def test_circuits(self):
    """Confirms correct circuits are sampled."""
    circuits_wrapper = tf.function(self.actual_qhbm.circuits)
    actual_circuits, actual_counts = circuits_wrapper(self.num_samples)

    # Circuits with the allowed-to-be-sampled bitstrings prepended.
    u = tfq.from_tensor(self.actual_qhbm.q_inference.circuit.pqc)[0]
    qubits = self.actual_qhbm.q_inference.circuit.qubits
    expected_circuits_deserialized = [
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
    actual_circuits_deserialized = tfq.from_tensor(actual_circuits)
    self.assertTrue(
        any([
            expected_circuits_deserialized[0] ==
            actual_circuits_deserialized[0],
            expected_circuits_deserialized[0] ==
            actual_circuits_deserialized[1],
        ]))
    self.assertTrue(
        any([
            expected_circuits_deserialized[1] ==
            actual_circuits_deserialized[0],
            expected_circuits_deserialized[1] ==
            actual_circuits_deserialized[1],
        ]))
    # Check that the fraction is approximately 0.5 (equal counts)
    self.assertAllClose(
        actual_counts[0], actual_counts[1], atol=self.num_samples / 1000)

  def test_circuit_param_update(self):
    """Confirm circuits are different after updating energy model parameters."""
    num_bits = 2
    actual_energy = models.BernoulliEnergy(list(range(num_bits)))
    e_infer = inference.BernoulliEnergyInference(actual_energy,
                                                 self.num_samples)

    qubits = cirq.GridQubit.rect(1, num_bits)
    pqc = cirq.Circuit(cirq.Y(q) for q in qubits)
    actual_circuit = models.DirectQuantumCircuit(pqc)
    q_infer = inference.AnalyticQuantumInference(actual_circuit)

    h_infer = inference.QHBM(e_infer, q_infer)
    circuits_wrapper = tf.function(h_infer.circuits)

    # Pin Bernoulli to [0, 1]
    actual_energy.set_weights([tf.constant([-1000, 1000])])
    expected_circuits_1 = tfq.from_tensor(
        tfq.convert_to_tensor(
            [cirq.Circuit(cirq.X(qubits[0])**0, cirq.X(qubits[1])) + pqc]))
    output_circuits, _ = circuits_wrapper(self.num_samples)
    actual_circuits_1 = tfq.from_tensor(output_circuits)
    self.assertAllEqual(actual_circuits_1, expected_circuits_1)

    # Change pin to [1, 0]
    actual_energy.set_weights([tf.constant([1000, -1000])])
    expected_circuits_2 = tfq.from_tensor(
        tfq.convert_to_tensor(
            [cirq.Circuit(cirq.X(qubits[0]),
                          cirq.X(qubits[1])**0) + pqc]))
    output_circuits, _ = circuits_wrapper(self.num_samples)
    actual_circuits_2 = tfq.from_tensor(output_circuits)
    self.assertAllEqual(actual_circuits_2, expected_circuits_2)

    # Assumption check, that circuits are actually different
    self.assertNotAllEqual(actual_circuits_1, actual_circuits_2)

  @test_util.eager_mode_toggle
  def test_expectation_pauli(self):
    """Compares QHBM expectation value to manual expectation."""
    # observable
    num_bits = 4
    qubits = cirq.GridQubit.rect(1, num_bits)
    raw_ops = [
        cirq.PauliSum.from_pauli_strings(
            [cirq.PauliString(cirq.Z(q)) for q in qubits])
    ]
    ops = tfq.convert_to_tensor(raw_ops)

    # unitary
    num_layers = 3
    actual_h, actual_h_infer = test_util.get_random_hamiltonian_and_inference(
        qubits,
        num_layers,
        "expectation_test",
        self.num_samples,
        ebm_seed=self.tfp_seed)

    # sample bitstrings
    samples = actual_h_infer.e_inference.sample(self.num_samples)
    bitstrings, _, counts = utils.unique_bitstrings_with_counts(samples)

    # calculate expected values
    raw_expectations = actual_h_infer.q_inference.expectation(bitstrings, ops)
    expected_expectations = utils.weighted_average(counts, raw_expectations)
    # Check that expectations are a reasonable size
    self.assertAllGreater(tf.math.abs(expected_expectations), 1e-3)

    expectation_wrapper = tf.function(actual_h_infer.expectation)
    actual_expectations = expectation_wrapper(ops)
    self.assertAllClose(actual_expectations, expected_expectations, rtol=1e-6)

    # Ensure energy parameter update changes the expectation value.
    old_energy_weights = actual_h.energy.get_weights()
    actual_h.energy.set_weights([tf.ones_like(w) for w in old_energy_weights])
    altered_energy_expectations = actual_h_infer.expectation(ops)
    self.assertNotAllClose(
        altered_energy_expectations, actual_expectations, rtol=1e-5)
    actual_h.energy.set_weights(old_energy_weights)

    # Ensure circuit parameter update changes the expectation value.
    old_circuit_weights = actual_h.circuit.get_weights()
    actual_h.circuit.set_weights([tf.ones_like(w) for w in old_circuit_weights])
    altered_circuit_expectations = expectation_wrapper(ops)
    self.assertNotAllClose(
        altered_circuit_expectations, actual_expectations, rtol=1e-5)
    actual_h.circuit.set_weights(old_circuit_weights)

    # Check that values return to start.
    reset_expectations = expectation_wrapper(ops)
    self.assertAllClose(reset_expectations, actual_expectations, rtol=1e-6)

  @parameterized.parameters({
      "energy_class": energy_class,
      "energy_args": energy_args,
  } for energy_class, energy_args in zip([models.BernoulliEnergy, models.KOBE],
                                         [[], [2]]))
  @test_util.eager_mode_toggle
  def test_expectation_modular_hamiltonian(self, energy_class, energy_args):
    """Confirm expectation of modular Hamiltonians works."""
    # set up the modular Hamiltonian to measure
    num_bits = 3
    n_moments = 5
    act_fraction = 1.0
    qubits = cirq.GridQubit.rect(1, num_bits)
    energy_h = energy_class(*([list(range(num_bits))] + energy_args))
    energy_h.build([None, num_bits])
    raw_circuit_h = cirq.testing.random_circuit(qubits, n_moments, act_fraction)
    circuit_h = models.DirectQuantumCircuit(raw_circuit_h)
    hamiltonian_measure = models.Hamiltonian(energy_h, circuit_h)

    # unitary
    num_layers = 3
    _, actual_h_infer = test_util.get_random_hamiltonian_and_inference(
        qubits,
        num_layers,
        "expectation_test",
        self.num_samples,
        ebm_seed=self.tfp_seed)

    # sample bitstrings
    samples = actual_h_infer.e_inference.sample(self.num_samples)
    bitstrings, _, counts = utils.unique_bitstrings_with_counts(samples)

    # calculate expected values
    raw_expectations = actual_h_infer.q_inference.expectation(
        bitstrings, hamiltonian_measure)
    expected_expectations = utils.weighted_average(counts, raw_expectations)
    # Check that expectations are a reasonable size
    self.assertAllGreater(tf.math.abs(expected_expectations), 1e-3)

    expectation_wrapper = tf.function(actual_h_infer.expectation)
    actual_expectations = expectation_wrapper(hamiltonian_measure)
    self.assertAllClose(actual_expectations, expected_expectations)


if __name__ == "__main__":
  absl.logging.info("Running qhbm_test.py ...")
  tf.test.main()
