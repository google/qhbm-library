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
"""Tests for qhbmlib.data.qhbm_data"""

import cirq
import tensorflow as tf

from qhbmlib import data
from tests import test_util


class QHBMDataTest(tf.test.TestCase):
  """Tests the QHBM data class."""

  def setUp(self):
    """Initializes test objects."""
    self.num_qubits_list = [1, 2, 3]
    self.tfp_seed = tf.constant([7, 8], tf.int32)
    self.num_samples = int(1e6)
    self.close_rtol = 3e-2

  def test_expectation(self):
    """Confirms initialization."""
    for num_qubits in self.num_qubits_list:
      qubits = cirq.GridQubit.rect(1, num_qubits)
      num_layers = 5
      _, qhbm = test_util.get_random_hamiltonian_and_inference(
          qubits,
          num_layers,
          f"data_objects_{num_qubits}",
          self.num_samples,
          ebm_seed=self.tfp_seed)
      hamiltonian, _ = test_util.get_random_hamiltonian_and_inference(
          qubits, num_layers, f"observable_{num_qubits}", self.num_samples)
      expected_expectation = tf.squeeze(qhbm.expectation(hamiltonian))

      actual_data = data.QHBMData(qhbm)
      actual_expectation = actual_data.expectation(hamiltonian)

      self.assertAllClose(actual_expectation, expected_expectation)
