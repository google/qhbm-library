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
from absl.testing import parameterized
import random
import string

import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
from tensorflow_quantum.python import util as tfq_util

from qhbmlib import circuit_infer
from qhbmlib import circuit_model
from qhbmlib import circuit_model_utils
from qhbmlib import energy_infer
from qhbmlib import energy_model
from qhbmlib import hamiltonian_model
from qhbmlib import hamiltonian_infer
from qhbmlib import utils
from tests import test_util


class QHBMTest(parameterized.TestCase, tf.test.TestCase):
  """Tests the QHBM class."""

  def setUp(self):
    """Initializes test objects."""
    super().setUp()
    self.num_samples = int(1e7)
    self.tfp_seed = tf.constant([5, 6], dtype=tf.int32)

    # EBM
    num_bits = 3
    self.energy = energy_model.BernoulliEnergy(list(range(num_bits)))
    self.expected_e_inference = energy_infer.AnalyticEnergyInference(
        self.energy, self.num_samples)
    # pin first and last bits, middle bit free.
    self.energy.set_weights([tf.constant([-23, 0, 17])])

    # QNN
    qubits = cirq.GridQubit.rect(1, num_bits)
    symbols = set()
    num_symbols = 20
    for _ in range(num_symbols):
      symbols.add("".join(random.sample(string.ascii_letters, 10)))
    self.pqc = tfq_util.random_symbol_circuit(qubits, symbols)
    circuit = circuit_model.DirectQuantumCircuit(self.pqc)
    self.expected_q_inference = circuit_infer.QuantumInference(circuit)

    # QHBM
    self.expected_name = "nameforaQHBM"
    self.actual_qhbm = hamiltonian_infer.QHBM(self.expected_e_inference,
                                              self.expected_q_inference,
                                              self.expected_name)    
    self.model = self.actual_qhbm.hamiltonian

  def test_init(self):
    """Tests QHBM initialization."""
    self.assertEqual(self.actual_qhbm.e_inference, self.expected_e_inference)
    self.assertEqual(self.actual_qhbm.q_inference, self.expected_q_inference)
    self.assertEqual(self.actual_qhbm.name, self.expected_name)

  @test_util.eager_mode_toggle
  def test_circuits(self):
    """Confirms correct circuits are sampled."""
    circuits_wrapper = tf.function(self.actual_qhbm.circuits)
    actual_circuits, actual_counts = circuits_wrapper(self.num_samples)

    # Circuits with the allowed-to-be-sampled bitstrings prepended.
    u = tfq.from_tensor(self.model.circuit.pqc)[0]
    qubits = self.model.circuit.qubits
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
    # EBM
    energy = energy_model.BernoulliEnergy(list(range(num_bits)))
    e_infer = energy_infer.BernoulliEnergyInference(energy, self.num_samples)

    # QNN
    qubits = cirq.GridQubit.rect(1, num_bits)
    pqc = cirq.Circuit(cirq.Y(q) for q in qubits)
    circuit = circuit_model.DirectQuantumCircuit(pqc)    
    q_infer = circuit_infer.QuantumInference(circuit)

    # QHBM
    h_infer = hamiltonian_infer.QHBM(e_infer, q_infer)
    circuits_wrapper = tf.function(h_infer.circuits)
    model = h_infer.hamiltonian

    # Pin Bernoulli to [0, 1]
    energy.set_weights([tf.constant([-1000, 1000])])
    expected_circuits_1 = tfq.from_tensor(
        tfq.convert_to_tensor(
            [cirq.Circuit(cirq.X(qubits[0])**0, cirq.X(qubits[1])) + pqc]))
    output_circuits, _ = circuits_wrapper(self.num_samples)
    actual_circuits_1 = tfq.from_tensor(output_circuits)
    self.assertAllEqual(actual_circuits_1, expected_circuits_1)

    # Change pin to [1, 0]
    energy.set_weights([tf.constant([1000, -1000])])
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
  def test_expectation_cirq(self):
    """Compares library expectation values to those from Cirq."""
    # observable
    num_bits = 4
    qubits = cirq.GridQubit.rect(1, num_bits)
    raw_ops = [
        cirq.PauliSum.from_pauli_strings(
            [cirq.PauliString(cirq.Z(q)) for q in qubits])
    ]
    ops = tfq.convert_to_tensor(raw_ops)

    # unitary
    batch_size = 1
    n_moments = 10
    act_fraction = 0.9
    num_symbols = 2
    symbols = set()
    for _ in range(num_symbols):
      symbols.add("".join(random.sample(string.ascii_letters, 10)))
    symbols = sorted(list(symbols))
    raw_circuits, raw_resolvers = tfq_util.random_symbol_circuit_resolver_batch(
        qubits, symbols, batch_size, n_moments=n_moments, p=act_fraction)
    raw_circuit = raw_circuits[0]
    resolver = {k: raw_resolvers[0].value_of(k) for k in raw_resolvers[0]}

    # EBM
    energy = energy_model.BernoulliEnergy(list(range(num_bits)))
    e_infer = energy_infer.BernoulliEnergyInference(energy, self.num_samples,
                                                    self.tfp_seed)

    # QNN
    circuit = circuit_model.QuantumCircuit(
        tfq.convert_to_tensor([raw_circuit]), qubits, tf.constant(symbols),
        [tf.Variable([resolver[s] for s in symbols])], [[]])
    q_infer = circuit_infer.QuantumInference(circuit)
    actual_h_infer = hamiltonian_infer.QHBM(e_infer, q_infer)
    actual_hamiltonian = actual_h_infer.hamiltonian

    # sample bitstrings
    samples = e_infer.sample(self.num_samples)
    bitstrings, _, counts = utils.unique_bitstrings_with_counts(samples)
    bit_list = bitstrings.numpy().tolist()

    # bitstring injectors
    bitstring_circuit = circuit_model_utils.bit_circuit(qubits)
    bitstring_symbols = sorted(tfq.util.get_circuit_symbols(bitstring_circuit))
    bitstring_resolvers = [
        dict(zip(bitstring_symbols, bstr)) for bstr in bit_list
    ]

    # calculate expected values
    total_circuit = bitstring_circuit + raw_circuit
    total_resolvers = [{**r, **resolver} for r in bitstring_resolvers]
    raw_expectations = tf.constant([[
        cirq.Simulator().simulate_expectation_values(total_circuit, o,
                                                     r)[0].real for o in raw_ops
    ] for r in total_resolvers])
    expected_expectations = utils.weighted_average(counts, raw_expectations)
    # Check that expectations are a reasonable size
    self.assertAllGreater(tf.math.abs(expected_expectations), 1e-3)

    expectation_wrapper = tf.function(actual_h_infer.expectation)
    actual_expectations = expectation_wrapper(ops)
    self.assertAllClose(actual_expectations, expected_expectations, rtol=1e-6)

    # Ensure energy parameter update changes the expectation value.
    old_energy_weights = energy.get_weights()
    energy.set_weights([tf.ones_like(w) for w in old_energy_weights])
    altered_energy_expectations = actual_h_infer.expectation(ops)
    self.assertNotAllClose(
        altered_energy_expectations, actual_expectations, rtol=1e-5)
    energy.set_weights(old_energy_weights)

    # Ensure circuit parameter update changes the expectation value.
    old_circuit_weights = circuit.get_weights()
    circuit.set_weights([tf.ones_like(w) for w in old_circuit_weights])
    altered_circuit_expectations = expectation_wrapper(ops)
    self.assertNotAllClose(
        altered_circuit_expectations, actual_expectations, rtol=1e-5)
    circuit.set_weights(old_circuit_weights)

    # Check that values return to start.
    reset_expectations = expectation_wrapper(ops)
    self.assertAllClose(reset_expectations, actual_expectations, rtol=1e-6)

  @parameterized.parameters({
      "energy_class": energy_class,
      "energy_args": energy_args,
  } for energy_class, energy_args in zip(
      [energy_model.BernoulliEnergy, energy_model.KOBE], [[], [2]]))
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
    circuit_h = circuit_model.DirectQuantumCircuit(raw_circuit_h)
    circuit_h.build([])
    hamiltonian_measure = hamiltonian_model.Hamiltonian(energy_h, circuit_h)
    raw_shards = tfq.from_tensor(hamiltonian_measure.operator_shards)

    # hamiltonian model and inference
    seed = tf.constant([5, 6], dtype=tf.int32)
    model_energy = energy_model.BernoulliEnergy(list(range(num_bits)))
    model_energy.build([None, num_bits])
    model_raw_circuit = cirq.testing.random_circuit(qubits, n_moments,
                                                    act_fraction)
    model_circuit = circuit_model.DirectQuantumCircuit(model_raw_circuit)
    model_circuit.build([])
    model_hamiltonian = hamiltonian_model.Hamiltonian(model_energy,
                                                      model_circuit)
    e_infer = energy_infer.BernoulliEnergyInference(num_bits, self.num_samples,
                                                    seed)
    q_infer = circuit_infer.QuantumInference()
    model_h_infer = hamiltonian_infer.QHBM(e_infer, q_infer)

    # sample bitstrings
    e_infer.infer(model_energy)
    samples = e_infer.sample(self.num_samples)
    bitstrings, _, counts = utils.unique_bitstrings_with_counts(samples)
    bit_list = bitstrings.numpy().tolist()

    # bitstring injectors
    bitstring_circuit = circuit_model_utils.bit_circuit(qubits)
    bitstring_symbols = sorted(tfq.util.get_circuit_symbols(bitstring_circuit))
    bitstring_resolvers = [
        dict(zip(bitstring_symbols, bstr)) for bstr in bit_list
    ]

    # calculate expected values
    total_circuit = bitstring_circuit + model_raw_circuit + raw_circuit_h**-1
    raw_expectations = tf.stack([
        tf.stack([
            hamiltonian_measure.energy.operator_expectation([
                cirq.Simulator().simulate_expectation_values(
                    total_circuit, o, r)[0].real for o in raw_shards
            ])
        ]) for r in bitstring_resolvers
    ])
    expected_expectations = utils.weighted_average(counts, raw_expectations)

    expectation_wrapper = tf.function(model_h_infer.expectation)
    actual_expectations = expectation_wrapper(model_hamiltonian,
                                              hamiltonian_measure)
    self.assertAllClose(actual_expectations, expected_expectations)


if __name__ == "__main__":
  absl.logging.info("Running hamiltonian_infer_test.py ...")
  tf.test.main()
