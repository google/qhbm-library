# pylint: skip-file
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
"""Tests for hamiltonian.py"""

import cirq
import tensorflow as tf

from qhbmlib import hamiltonian


class QubitGridTest(tf.test.TestCase):
  """Test the qubit_grid function."""

  def test_qubit_grid(self):
    """Confirm correct grid."""
    r = 2
    c = 3
    test_qubits = hamiltonian._qubit_grid(r, c)
    for test_r, test_row in enumerate(test_qubits):
      for test_c, q in enumerate(test_row):
        self.assertEqual(cirq.GridQubit(test_r, test_c), q)


class HeisenbergTest(tf.test.TestCase):
  """Test components related to the 2D Heisenberg Hamiltonian."""

  def test_heisenberg_bond(self):
    """Test heisenberg bonds."""
    q0 = cirq.GridQubit(3, 7)
    q1 = cirq.GridQubit(2, 1)
    for t, pauli in enumerate([cirq.X, cirq.Y, cirq.Z], start=1):
      test_bond = hamiltonian.heisenberg_bond(q0, q1, t)
      self.assertEqual(
          cirq.PauliSum.from_pauli_strings([pauli(q0) * pauli(q1)]), test_bond)
    with self.assertRaises(ValueError):
      _ = hamiltonian.heisenberg_bond(q0, q1, 4)

  def test_heisenberg_hamiltonian_shard(self):
    """Test heisenberg shards."""
    rows_list = [2, 3, 4, 5]
    columns_list = [2, 3, 4, 5]
    for rows in rows_list:
      for columns in columns_list:
        jh = -1.5
        jv = 2.3
        horizontal_qubit_pairs = []
        for this_r in range(rows):
          current_qubit_pairs = []
          for this_c in range(columns - 1):
            current_qubit_pairs.append((
                cirq.GridQubit(this_r, this_c),
                cirq.GridQubit(this_r, this_c + 1),
            ))
          horizontal_qubit_pairs += current_qubit_pairs

        vertical_qubit_pairs = []
        for this_c in range(columns):
          current_qubit_pairs = []
          for this_r in range(rows - 1):
            current_qubit_pairs.append((
                cirq.GridQubit(this_r, this_c),
                cirq.GridQubit(this_r + 1, this_c),
            ))
          vertical_qubit_pairs += current_qubit_pairs

        for t, pauli in enumerate([cirq.X, cirq.Y, cirq.Z], start=1):
          test_shard = hamiltonian.heisenberg_hamiltonian_shard(
              rows, columns, jh, jv, t)
          actual_shard = cirq.PauliSum()
          for q0, q1 in horizontal_qubit_pairs:
            actual_shard += jh * pauli(q0) * pauli(q1)
          for q0, q1 in vertical_qubit_pairs:
            actual_shard += jv * pauli(q0) * pauli(q1)
          self.assertEqual(actual_shard, test_shard)


class TFIMTest(tf.test.TestCase):
  """Test components related to the 2D Transverse Field Ising Model."""

  def test_tfim_x_shard(self):
    """Test that every qubit gets an X term at critical field value."""
    rows_list = [2, 3, 4, 5]
    columns_list = [2, 3, 4, 5]
    for rows in rows_list:
      for columns in columns_list:
        lambda_crit = 3.05
        qubits = cirq.GridQubit.rect(rows, columns)
        test_shard = hamiltonian.tfim_x_shard(rows, columns)
        actual_shard = cirq.PauliSum()
        for q in qubits:
          actual_shard += lambda_crit * cirq.X(q)
        self.assertEqual(actual_shard, test_shard)

  def test_tfim_zz_shard(self):
    """Test that ZZ interactions show up on a toroidal grid."""
    rows_list = [2, 3, 4, 5]
    columns_list = [2, 3, 4, 5]
    for rows in rows_list:
      for columns in columns_list:
        horizontal_qubit_pairs = []
        for this_r in range(rows):
          current_qubit_pairs = []
          for this_c in range(columns):
            current_qubit_pairs.append((
                cirq.GridQubit(this_r, this_c),
                cirq.GridQubit(this_r, (this_c + 1) % columns),
            ))
          horizontal_qubit_pairs += current_qubit_pairs

        vertical_qubit_pairs = []
        for this_c in range(columns):
          current_qubit_pairs = []
          for this_r in range(rows):
            current_qubit_pairs.append((
                cirq.GridQubit(this_r, this_c),
                cirq.GridQubit((this_r + 1) % rows, this_c),
            ))
          vertical_qubit_pairs += current_qubit_pairs

        test_shard = hamiltonian.tfim_zz_shard(rows, columns)
        actual_shard = cirq.PauliSum()
        for q0, q1 in horizontal_qubit_pairs:
          actual_shard += cirq.Z(q0) * cirq.Z(q1)
        for q0, q1 in vertical_qubit_pairs:
          actual_shard += cirq.Z(q0) * cirq.Z(q1)
        self.assertEqual(actual_shard, test_shard)


if __name__ == "__main__":
  print("Running hamiltonian_test.py ...")
  tf.test.main()
