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
    self.expected_energy =
    self.expected_circuit = 
    self.actual_hamiltonian = hamiltonian_model.Hamiltonian(
        self.expected_energy,
        self.expected_circuit,
        self.expected_name)

  def test_init(self):
    """Tests Hamiltonian initialization."""
    pass

  def test_add(self):
    """Tests Hamiltonian addition."""
    pass


class HamiltonianSumTest(tf.test.TestCase):
  """Tests the HamiltonianSum class."""

  def setUp(self):
    """Initializes test objects."""
    super().setUp()

  def test_init(self):
    """Tests HamiltonianSum initialization."""
    pass

  def test_add(self):
    """Tests HamiltonianSum addition."""
    pass


if __name__ == "__main__":
  absl.logging.info("Running hamiltonian_model_test.py ...")
  tf.test.main()
