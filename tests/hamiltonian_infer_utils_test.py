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
"""Tests for the hamiltonian_infer_utils module."""

import cirq
import tensorflow as tf

from qhbmlib import circuit_model
from qhbmlib import energy_model
from qhbmlib import hamiltonian_model
from qhbmlib import hamiltonian_infer_utils
from tests import test_util


class DensityMatrixTest(tf.test.TestCase):
  """Tests the density_matrix function."""

  @test_util.eager_mode_toggle
  def test_density_matrix(self):
    """Confirms the density matrix represented by the QHBM is correct."""

    # Check density matrix of Bell state.
    num_bits = 2
    energy = energy_model.BernoulliEnergy(list(range(num_bits)))
    energy.build([None, num_bits])
    energy.set_weights([tf.constant([-10.0, -10.0])])  # pin at |00>

    qubits = cirq.GridQubit.rect(1, num_bits)
    test_u = cirq.Circuit([cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1])])
    circuit = circuit_model.DirectQuantumCircuit(test_u)
    circuit.build([])
    model = hamiltonian_model.Hamiltonian(energy, circuit)

    expected_dm = tf.constant(
        [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]],
        tf.complex64,
    )

    density_matrix_wrapper = tf.function(hamiltonian_infer_utils.density_matrix)
    actual_dm = density_matrix_wrapper(model)
    self.assertAllClose(actual_dm, expected_dm)


class FidelityTest(tf.test.TestCase):
  """Tests the fidelity function."""

  def setUp(self):
    """Initializes test objects."""
    super().setUp()
    self.close_rtol = 1e-4

  @test_util.eager_mode_toggle
  def test_fidelity_self(self):
    """Confirms the fidelity of a model with itself is 1."""
    num_bits = 4
    num_samples = 1  # required but not used
    qubits = cirq.GridQubit.rect(1, num_bits)
    num_layers = 3
    model, _ = test_util.get_random_hamiltonian_and_inference(
        qubits, num_layers, "test_fidelity", num_samples)

    density_matrix_wrapper = tf.function(hamiltonian_infer_utils.density_matrix)
    model_dm = density_matrix_wrapper(model)
    fidelity_wrapper = tf.function(hamiltonian_infer_utils.fidelity)
    actual_fidelity = fidelity_wrapper(model, model_dm)

    expected_fidelity = 1.0
    self.assertAllClose(
        actual_fidelity, expected_fidelity, rtol=self.close_rtol)


if __name__ == "__main__":
  print("Running hamiltonian_infer_utils_test.py ...")
  tf.test.main()
