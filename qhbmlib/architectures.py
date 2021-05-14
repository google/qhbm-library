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
"""Selection of quantum circuits architectures used in defining QHBMs."""

import bisect

import cirq
import sympy
import tensorflow_quantum as tfq

from qhbmlib import ebm


# ============================================================================ #
# HEA components.
# ============================================================================ #


def get_xz_rotation(q, a, b):
    """Two-axis single qubit rotation."""
    return cirq.Circuit(cirq.X(q) ** a, cirq.Z(q) ** b)


def get_xyz_rotation(q, a, b, c):
    """General single qubit rotation."""
    return cirq.Circuit(cirq.X(q) ** a, cirq.Y(q) ** b, cirq.Z(q) ** c)


def get_cz_exp(q0, q1, a):
    """Exponent of entangling CZ gate."""
    return cirq.Circuit(cirq.CZPowGate(exponent=a)(q0, q1))


def get_zz_exp(q0, q1, a):
    """Exponent of entangling ZZ gate."""
    return cirq.Circuit(cirq.ZZPowGate(exponent=a)(q0, q1))


def get_cnot_exp(q0, q1, a):
    """Exponent of entangling CNot gate."""
    return cirq.Circuit(cirq.CNotPowGate(exponent=a)(q0, q1))


def get_xz_rotation_layer(qubits, layer_num, name):
    """Apply two-axis single qubit rotations to all the given qubits."""
    layer_symbols = []
    circuit = cirq.Circuit()
    for n, q in enumerate(qubits):
        sx, sz = sympy.symbols(
            "sx_{0}_{1}_{2} sz_{0}_{1}_{2}".format(name, layer_num, n)
        )
        layer_symbols += [sx, sz]
        circuit += get_xz_rotation(q, sx, sz)
    return circuit, layer_symbols


def get_cz_exp_layer(qubits, layer_num, name):
    """Apply parameterized CZ gates to all pairs of nearest-neighbor qubits."""
    layer_symbols = []
    circuit = cirq.Circuit()
    for n, (q0, q1) in enumerate(zip(qubits[::2], qubits[1::2])):
        a = sympy.symbols("sc_{0}_{1}_{2}".format(name, layer_num, 2 * n))
        layer_symbols += [a]
        circuit += get_cz_exp(q0, q1, a)
    shifted_qubits = qubits[1::]
    for n, (q0, q1) in enumerate(zip(shifted_qubits[::2], shifted_qubits[1::2])):
        a = sympy.symbols("sc_{0}_{1}_{2}".format(name, layer_num, 2 * n + 1))
        layer_symbols += [a]
        circuit += get_cz_exp(q0, q1, a)
    return circuit, layer_symbols


def get_hardware_efficient_model_unitary(qubits, num_layers, name):
    """Build our full parameterized model unitary."""
    circuit = cirq.Circuit()
    all_symbols = []
    for layer_num in range(num_layers):
        new_circ, new_symb = get_xz_rotation_layer(qubits, layer_num, name)
        circuit += new_circ
        all_symbols += new_symb
        if len(qubits) > 1:
            new_circ, new_symb = get_cz_exp_layer(qubits, layer_num, name)
            circuit += new_circ
            all_symbols += new_symb
    return circuit, all_symbols


def get_cnot_exp_layer(qubits, layer_num, name):
    """Apply CNot gates to all pairs of nearest-neighbor qubits."""
    layer_symbols = []
    circuit = cirq.Circuit()
    for n, (q0, q1) in enumerate(zip(qubits[::2], qubits[1::2])):
        a = sympy.symbols("sc_{0}_{1}_{2}".format(name, layer_num, 2 * n))
        layer_symbols += [a]
        circuit += get_cnot_exp(q0, q1, a)
    shifted_qubits = qubits[1::]
    for n, (q0, q1) in enumerate(zip(shifted_qubits[::2], shifted_qubits[1::2])):
        a = sympy.symbols("sc_{0}_{1}_{2}".format(name, layer_num, 2 * n + 1))
        layer_symbols += [a]
        circuit += get_cnot_exp(q0, q1, a)
    return circuit, layer_symbols


def hea_1d_cnot(qubits, num_layers, name):
    """Build our full parameterized model unitary."""
    circuit = cirq.Circuit()
    all_symbols = []
    for layer_num in range(num_layers):
        new_circ, new_symb = get_xz_rotation_layer(qubits, layer_num, name)
        circuit += new_circ
        all_symbols += new_symb
        if len(qubits) > 1:
            new_circ, new_symb = get_cnot_exp_layer(qubits, layer_num, name)
            circuit += new_circ
            all_symbols += new_symb
    return circuit, all_symbols


def get_zz_exp_layer(qubits, layer_num, name):
    """Apply ZZ gates to all pairs of nearest-neighbor qubits."""
    layer_symbols = []
    circuit = cirq.Circuit()
    for n, (q0, q1) in enumerate(zip(qubits[::2], qubits[1::2])):
        a = sympy.symbols("sc_{0}_{1}_{2}".format(name, layer_num, 2 * n))
        layer_symbols += [a]
        circuit += get_zz_exp(q0, q1, a)
    shifted_qubits = qubits[1::]
    for n, (q0, q1) in enumerate(zip(shifted_qubits[::2], shifted_qubits[1::2])):
        a = sympy.symbols("sc_{0}_{1}_{2}".format(name, layer_num, 2 * n + 1))
        layer_symbols += [a]
        circuit += get_zz_exp(q0, q1, a)
    return circuit, layer_symbols


def hea_1d_zz(qubits, num_layers, name):
    """Build our full parameterized model unitary."""
    circuit = cirq.Circuit()
    all_symbols = []
    for layer_num in range(num_layers):
        new_circ, new_symb = get_xz_rotation_layer(qubits, layer_num, name)
        circuit += new_circ
        all_symbols += new_symb
        if len(qubits) > 1:
            new_circ, new_symb = get_zz_exp_layer(qubits, layer_num, name)
            circuit += new_circ
            all_symbols += new_symb
    return circuit, all_symbols


# ============================================================================ #
# 2D HEA.
# ============================================================================ #


def get_2d_xz_rotation_layer(rows, cols, layer_num, name):
    """Apply single qubit rotations on a grid of qubits."""
    layer_symbols = []
    circuit = cirq.Circuit()
    for r in range(rows):
        for c in range(cols):
            sx = sympy.Symbol(f"sx_{name}_{layer_num}_{r}_{c}")
            sz = sympy.Symbol(f"sz_{name}_{layer_num}_{r}_{c}")
            layer_symbols += [sx, sz]
            circuit += get_xz_rotation(cirq.GridQubit(r, c), sx, sz)
    return circuit, layer_symbols


def get_2d_cz_exp_layer(rows, cols, layer_num, name):
    """Apply CZ gates to all pairs of nearest-neighbor qubits on a grid."""
    layer_symbols = []
    circuit = cirq.Circuit()
    # Apply horizontal bonds
    for r in range(rows):
        for par in [0, 1]:
            for q_c_0, q_c_1 in zip(range(par, cols, 2), range(par + 1, cols, 2)):
                scz = sympy.Symbol(f"scz_{name}_{layer_num}_row{r}_{q_c_0}_{q_c_1}")
                layer_symbols += [scz]
                circuit += get_cz_exp(
                    cirq.GridQubit(r, q_c_0), cirq.GridQubit(r, q_c_1), scz
                )
    # Apply vertical bonds
    for c in range(cols):
        for par in [0, 1]:
            for q_r_0, q_r_1 in zip(range(par, rows, 2), range(par + 1, rows, 2)):
                scz = sympy.Symbol(f"scz_{name}_{layer_num}_col{c}_{q_r_0}_{q_r_1}")
                layer_symbols += [scz]
                circuit += get_cz_exp(
                    cirq.GridQubit(q_r_0, c), cirq.GridQubit(q_r_1, c), scz
                )
    return circuit, layer_symbols


def get_2d_hea(rows, cols, num_layers, name):
    """Build a 2D HEA ansatz.

    Args:
      rows: int specifying the number of rows in the ansatz.
      cols: int specifying the number of columns in the ansatz.
      num_layers: int specifying how many layers of 2D HEA to apply.
      name: string which will be included in the parameters of the ansatz.

    Returns:
      circuit: `cirq.Circuit` which is the ansatz.
      symbols: list of `sympy.Symbol`s which are the parameters of the model.
    """
    symbols = []
    circuit = cirq.Circuit()
    for layer in range(num_layers):
        xz_circuit, xz_symbols = get_2d_xz_rotation_layer(rows, cols, layer, name)
        circuit += xz_circuit
        symbols += xz_symbols
        cz_circuit, cz_symbols = get_2d_cz_exp_layer(rows, cols, layer, name)
        circuit += cz_circuit
        symbols += cz_symbols
    return circuit, symbols


def get_2d_xyz_rotation_layer(rows, cols, layer_num, name):
    """Apply single qubit rotations on a grid of qubits."""
    layer_symbols = []
    circuit = cirq.Circuit()
    for r in range(rows):
        for c in range(cols):
            sx = sympy.Symbol(f"sx_{name}_{layer_num}_{r}_{c}")
            sy = sympy.Symbol(f"sy_{name}_{layer_num}_{r}_{c}")
            sz = sympy.Symbol(f"sz_{name}_{layer_num}_{r}_{c}")
            layer_symbols += [sx, sy, sz]
            circuit += get_xyz_rotation(cirq.GridQubit(r, c), sx, sy, sz)
    return circuit, layer_symbols


def get_2d_hea_y(rows, cols, num_layers, name):
    """Build a 2D HEA ansatz.

    Args:
      rows: int specifying the number of rows in the ansatz.
      cols: int specifying the number of columns in the ansatz.
      num_layers: int specifying how many layers of 2D HEA to apply.
      name: string which will be included in the parameters of the ansatz.

    Returns:
      circuit: `cirq.Circuit` which is the ansatz.
      symbols: list of `sympy.Symbol`s which are the parameters of the model.
    """
    symbols = []
    circuit = cirq.Circuit()
    for layer in range(num_layers):
        xyz_circuit, xyz_symbols = get_2d_xyz_rotation_layer(rows, cols, layer, name)
        circuit += xyz_circuit
        symbols += xyz_symbols
        cz_circuit, cz_symbols = get_2d_cz_exp_layer(rows, cols, layer, name)
        circuit += cz_circuit
        symbols += cz_symbols
    return circuit, symbols


def get_2d_cnot_exp_layer(rows, cols, layer_num, name):
    """Apply CNot gates to all pairs of nearest-neighbor qubits on a grid."""
    layer_symbols = []
    circuit = cirq.Circuit()
    # Apply horizontal bonds
    for r in range(rows):
        for par in [0, 1]:
            for q_c_0, q_c_1 in zip(range(par, cols, 2), range(par + 1, cols, 2)):
                scnot = sympy.Symbol(f"scnot_{name}_{layer_num}_row{r}_{q_c_0}_{q_c_1}")
                layer_symbols += [scnot]
                circuit += get_cnot_exp(
                    cirq.GridQubit(r, q_c_0), cirq.GridQubit(r, q_c_1), scnot
                )
    # Apply vertical bonds
    for c in range(cols):
        for par in [0, 1]:
            for q_r_0, q_r_1 in zip(range(par, rows, 2), range(par + 1, rows, 2)):
                scnot = sympy.Symbol(f"scnot_{name}_{layer_num}_col{c}_{q_r_0}_{q_r_1}")
                layer_symbols += [scnot]
                circuit += get_cnot_exp(
                    cirq.GridQubit(q_r_0, c), cirq.GridQubit(q_r_1, c), scnot
                )
    return circuit, layer_symbols


def get_2d_hea_cnot(rows, cols, num_layers, name):
    """Build a 2D HEA ansatz.

    Args:
      rows: int specifying the number of rows in the ansatz.
      cols: int specifying the number of columns in the ansatz.
      num_layers: int specifying how many layers of 2D HEA to apply.
      name: string which will be included in the parameters of the ansatz.

    Returns:
      circuit: `cirq.Circuit` which is the ansatz.
      symbols: list of `sympy.Symbol`s which are the parameters of the model.
    """
    symbols = []
    circuit = cirq.Circuit()
    for layer in range(num_layers):
        xz_circuit, xz_symbols = get_2d_xz_rotation_layer(rows, cols, layer, name)
        circuit += xz_circuit
        symbols += xz_symbols
        cnot_circuit, cnot_symbols = get_2d_cnot_exp_layer(rows, cols, layer, name)
        circuit += cnot_circuit
        symbols += cnot_symbols
    return circuit, symbols


# ============================================================================ #
# Trotter components.
# ============================================================================ #


def get_trotter_model_unitary(p, h_list, name):
    """Get a trotterized ansatz.

    Args:
      p: integer representing the number of QAOA steps.
      h_list: List of `cirq.PauliSum`s representing the Hamiltonians to
          exponentiate to build the circuit.
      name: string used to make symbols unique to this call.

    Returns:
      circuit: `cirq.Circuit` representing the parameterized QAOA ansatz.
      all_symbols: Python `list` of `sympy.Symbol`s containing all the
          parameters of the circuit.
    """
    circuit = cirq.Circuit()
    all_symbols = []
    for j in range(p):
        for n, h in enumerate(h_list):
            new_symb = sympy.Symbol("phi_{0}_L{1}_H{2}".format(name, j, n))
            circuit += tfq.util.exponential([h], coefficients=[new_symb])
            all_symbols.append(new_symb)
    return circuit, all_symbols


# ============================================================================ #
# QNHF components.
# ============================================================================ #


def get_general_trotter_unitary(symbol_array, h_list):
    """
    Args:
      symbol_array: 2-D array (list of lists) of `sympy.Symbol`s
        to use when exponentiating the Hamiltonians.  The first index is the
        trotter layer, the second index is corresponding Hamiltonian in `h_list`.
      h_list: List of `cirq.PauliSum`s representing all the Hamiltonians to
        exponentiate in a single trotter step.

    Returns:
      circuit: `cirq.Circuit` representing the parameterized trotter ansatz.
    """
    if len(symbol_array[0]) != len(h_list):
        raise ValueError(
            "Must have the same number of symbols as Hamiltonians in each layer."
        )
    circuit = cirq.Circuit()
    for s_layer in symbol_array:
        for n, h in enumerate(h_list):
            circuit += tfq.util.exponential([h], coefficients=[s_layer[n]])
    return circuit


def get_qnhf_symbols(p, n_bits, n_h, max_k, name):
    """Get all the symbols used by QNHF QHBMs.

    Args:
      p: the number of trotter steps.
      n_bits: the number of bits in the discrete sample space.
      max_k: the maximum locality of interactions in the classical model.
      n_h: the number of non-classical hamiltonian terms.
      name: string appended to symbols to uniqueify across QHBMs.

    Returns:
      eta_theta_symbols: 2-D array of `sympy.Symbol`s, where the first index is the
        trotter step and the second index is the particular symbol at that layer.
      phis_symbols: 2-D array of `sympy.Symbol`s, where the first index is the
        trotter step and the second index is the hamiltonian term at that layer.
    """
    eta_theta_symbols = []
    phis_symbols = []
    num_thetas_per_layer = ebm.get_klocal_energy_function_num_values(n_bits, max_k)
    for t_step in range(p):
        eta_theta_symbols.append([])
        for j in range(num_thetas_per_layer):
            eta_theta_symbols[-1].append(
                sympy.Symbol("eta_L{1}_theta_T{2}_{0}".format(name, t_step, j))
            )
        phis_symbols.append([])
        for m in range(n_h):
            phis_symbols[-1].append(
                sympy.Symbol("phi_L{1}_H{2}_{0}".format(name, t_step, m))
            )
    return eta_theta_symbols, phis_symbols


def get_qnhf_diagonal_operators(qubits, max_k):
    diag_op_list = []
    for k in range(1, max_k + 1):
        index_list = ebm.get_parity_index_list(len(qubits), k)
        for this_index_list in index_list:
            this_z_list = [cirq.Z(qubits[i]) for i in this_index_list]
            diag_op_list.append(
                cirq.PauliSum.from_pauli_strings(cirq.PauliString(*this_z_list))
            )
    return diag_op_list


def get_qnhf_model_unitary(p, qubits, max_k, h_list, name):
    """Get the QNHF unitary corresponding to the given Hamiltonians.

    Args:
      p: the number of trotter steps.
      qubits: list of `cirq.GridQubit`s on which to build KOBE.
      max_k: the maximum locality of interactions in the classical model.
      h_list: list of `cirq.PauliSum`s representing the hamiltonian terms.
      name: string appended to symbols to uniqueify across QHBMs.

    Returns:
      circuit: `cirq.Circuit` representing the QNHF ansatz.
    """
    eta_theta_symbols, phis_symbols = get_qnhf_symbols(
        p, len(qubits), len(h_list), max_k, name
    )
    total_symbols = []
    for t_layer, p_layer in zip(eta_theta_symbols, phis_symbols):
        total_symbols.append(t_layer + p_layer)
    classical_h_list = get_qnhf_diagonal_operators(qubits, max_k)
    total_h_list = classical_h_list + h_list
    return get_general_trotter_unitary(total_symbols, total_h_list)


# ============================================================================ #
# Convolutional components.
# ============================================================================ #


def qubits_to_grid(qubits):
    qubit_grid = []
    for q in qubits:
        if q.row > len(qubit_grid) - 1:
            qubit_grid.append([])
        bisect.insort(qubit_grid[q.row], q)
    return qubit_grid


def one_qubit_unitary(q, symbols):
    """Make a Cirq circuit for an arbitrary one qubit unitary."""
    return cirq.Circuit(
        cirq.X(q) ** symbols[0], cirq.Y(q) ** symbols[1], cirq.Z(q) ** symbols[2]
    )


def two_qubit_unitary(q0, q1, symbols):
    """Make a Cirq circuit for an arbitrary two qubit unitary."""
    circuit = cirq.Circuit()
    circuit += one_qubit_unitary(q0, symbols[0:3])
    circuit += one_qubit_unitary(q1, symbols[3:6])
    circuit += [cirq.ZZ(q0, q1) ** symbols[6]]
    circuit += [cirq.YY(q0, q1) ** symbols[7]]
    circuit += [cirq.XX(q0, q1) ** symbols[8]]
    circuit += one_qubit_unitary(q0, symbols[9:12])
    circuit += one_qubit_unitary(q1, symbols[12:])
    return circuit


def two_qubit_pool(source_qubit, sink_qubit, symbols):
    """Make a Cirq circuit to do a parameterized 'pooling' operation, which
    attempts to reduce entanglement down from two qubits to just one."""
    pool_circuit = cirq.Circuit()
    sink_basis_selector = one_qubit_unitary(sink_qubit, symbols[0:3])
    source_basis_selector = one_qubit_unitary(source_qubit, symbols[3:6])
    pool_circuit.append(sink_basis_selector)
    pool_circuit.append(source_basis_selector)
    pool_circuit.append(cirq.CNOT(control=source_qubit, target=sink_qubit))
    pool_circuit.append(sink_basis_selector ** -1)
    return pool_circuit


# TODO(#19)
# def quantum_convolutional_layer(qubits, layer_num, name):
#     """Assumes the qubits are arranged on a grid."""
#     qubit_grid = qubits_to_grid(qubits)
#     layer_symbols = []
#     circuit = cirq.Circuit()
#     tied_2q_symbols = [
#         sympy.Symbol("s_conv_I{0}_L{1}_N{2}".format(name, layer_num, s_num))
#         for s_num in range(15)
#     ]
#     # Apply horizontal bonds
#     for r in qubit_grid:
#         r_clipped = r[1:]
#         for alt_r in [r, r_clipped]:
#             for q0, q1 in zip(alt_r[::2], alt_r[1::2]):
#                 circuit += two_qubit_unitary(q0, q1, tied_2q_symbols)
#     # Apply vertical bonds
#     grid_clipped = qubit_grid[1:]
#     for r0, r1 in zip(qubit_grid[::2], qubit_grid[1::2]):
#         for q0, q1 in zip(r0, r1):
#             circuit += two_qubit_unitary(q0, q1, tied_2q_symbols)
#     for r0, r1 in zip(grid_clipped[::2], grid_clipped[1::2]):
#         for q0, q1 in zip(r0, r1):
#             circuit += two_qubit_unitary(q0, q1, tied_2q_symbols)
#     return circuit, tied_2q_symbols


# def quantum_pool_layer(qubits, layer_num, name, direction_flag):
#     """Assumes the qubits are arranged on a grid."""
#     qubit_grid = qubits_to_grid(qubits)
#     layer_symbols = []
#     circuit = cirq.Circuit()
#     tied_pool_symbols = [
#         sympy.Symbol("s_pool_I{0}_L{1}_N{2}".format(name, layer_num, s_num))
#         for s_num in range(6)
#     ]
