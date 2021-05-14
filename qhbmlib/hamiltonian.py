# Copyright 2021 The QHBM Library Authors.
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
"""Hamiltonians to use with QHBMs."""

import random

import cirq


def _qubit_grid(rows, cols):
    """Rectangle of qubits returned as a nested list."""
    qubits = []
    for r in range(rows):
        qubits.append([cirq.GridQubit(r, c) for c in range(cols)])
    return qubits


def heisenberg_bond(q0, q1, bond_type):
    """Given two Cirq qubits, return the PauliSum that bonds them."""
    if bond_type == 1:
        bond = cirq.X
    elif bond_type == 2:
        bond = cirq.Y
    elif bond_type == 3:
        bond = cirq.Z
    else:
        raise ValueError("Unknown bond type.")
    return cirq.PauliSum.from_pauli_strings([cirq.PauliString(bond(q0), bond(q1))])


def heisenberg_hamiltonian_shard(rows, columns, jh, jv, bond_type):
    """Returns a commuting subset of the 2D Heisenberg Hamiltonian."""
    qubits = _qubit_grid(rows, columns)
    heisenberg = cirq.PauliSum()
    # Apply horizontal bonds
    for r in qubits:
        for q0, q1 in zip(r, r[1::]):
            heisenberg += jh * heisenberg_bond(q0, q1, bond_type)
    # Apply vertical bonds
    for r0, r1 in zip(qubits, qubits[1::]):
        for q0, q1 in zip(r0, r1):
            heisenberg += jv * heisenberg_bond(q0, q1, bond_type)
    return heisenberg


def tfim_x_shard(rows, columns):
    """Build the X component of a rectangular toroid TFIM at critical field."""
    lambda_crit = 3.05  # see https://arxiv.org/abs/cond-mat/0703788
    qubits = cirq.GridQubit.rect(rows, columns)
    x_sum = cirq.PauliSum()
    for q in qubits:
        x_sum += lambda_crit * cirq.X(q)
    return x_sum


def tfim_zz_shard(rows, columns):
    """Uniform ferromagnetic interaction on a rectangular toroid."""
    qubits = _qubit_grid(rows, columns)
    extended_qubits = _qubit_grid(rows, columns)
    for r, row in enumerate(qubits):
        extended_qubits[r].append(row[0])
    extended_qubits.append(qubits[0])
    zz_sum = cirq.PauliSum()
    # Horizontal interactions.
    for row in extended_qubits[:-1]:
        for q0, q1 in zip(row, row[1:]):
            zz_sum += cirq.Z(q0) * cirq.Z(q1)
    # Vertical interactions.
    for row_0, row_1 in zip(extended_qubits, extended_qubits[1:]):
        for q0, q1 in zip(row_0[:-1], row_1):
            zz_sum += cirq.Z(q0) * cirq.Z(q1)
    return zz_sum


def tfim_x_shard_random(rows, columns, mean, std):
    """Build the X component of a rectangular toroid TFIM with random fields."""
    qubits = cirq.GridQubit.rect(rows, columns)
    x_sum = cirq.PauliSum()
    for q in qubits:
        x_sum += random.normalvariate(mean, std) * cirq.X(q)
    return x_sum


def tfim_zz_shard_random(rows, columns, mean, std):
    """Random ferromagnetic interaction on a rectangular toroid."""
    qubits = _qubit_grid(rows, columns)
    extended_qubits = _qubit_grid(rows, columns)
    for r, row in enumerate(qubits):
        extended_qubits[r].append(row[0])
    extended_qubits.append(qubits[0])
    zz_sum = cirq.PauliSum()
    # Horizontal interactions.
    for row in extended_qubits[:-1]:
        for q0, q1 in zip(row, row[1:]):
            zz_sum += random.normalvariate(mean, std) * cirq.Z(q0) * cirq.Z(q1)
    # Vertical interactions.
    for row_0, row_1 in zip(extended_qubits, extended_qubits[1:]):
        for q0, q1 in zip(row_0[:-1], row_1):
            zz_sum += random.normalvariate(mean, std) * cirq.Z(q0) * cirq.Z(q1)
    return zz_sum
