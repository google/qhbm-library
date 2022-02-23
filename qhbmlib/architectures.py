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
"""Selection of quantum circuits architectures used in defining QHBMs."""

import cirq
import sympy
import tensorflow_quantum as tfq

# ============================================================================ #
# HEA components.
# ============================================================================ #


def get_xz_rotation(q, a, b):
  """Two-axis single qubit rotation."""
  return cirq.Circuit(cirq.X(q)**a, cirq.Z(q)**b)


def get_xyz_rotation(q, a, b, c):
  """General single qubit rotation."""
  return cirq.Circuit(cirq.X(q)**a, cirq.Y(q)**b, cirq.Z(q)**c)


def get_cz_exp(q0, q1, a):
  """Exponent of entangling CZ gate."""
  return cirq.Circuit(cirq.CZPowGate(exponent=a)(q0, q1))


def get_zz_exp(q0, q1, a):
  """Exponent of entangling ZZ gate."""
  return cirq.Circuit(cirq.ZZPowGate(exponent=a)(q0, q1))


def get_xz_rotation_layer(qubits, layer_num, name):
  """Apply two-axis single qubit rotations to all the given qubits."""
  circuit = cirq.Circuit()
  for n, q in enumerate(qubits):
    sx, sz = sympy.symbols("sx_{0}_{1}_{2} sz_{0}_{1}_{2}".format(
        name, layer_num, n))
    circuit += get_xz_rotation(q, sx, sz)
  return circuit


def get_cz_exp_layer(qubits, layer_num, name):
  """Apply parameterized CZ gates to all pairs of nearest-neighbor qubits."""
  circuit = cirq.Circuit()
  for n, (q0, q1) in enumerate(zip(qubits[::2], qubits[1::2])):
    a = sympy.symbols("sc_{0}_{1}_{2}".format(name, layer_num, 2 * n))
    circuit += get_cz_exp(q0, q1, a)
  shifted_qubits = qubits[1::]
  for n, (q0, q1) in enumerate(zip(shifted_qubits[::2], shifted_qubits[1::2])):
    a = sympy.symbols("sc_{0}_{1}_{2}".format(name, layer_num, 2 * n + 1))
    circuit += get_cz_exp(q0, q1, a)
  return circuit


def get_hardware_efficient_model_unitary(qubits, num_layers, name):
  """Build our full parameterized model unitary."""
  circuit = cirq.Circuit()
  for layer_num in range(num_layers):
    new_circ = get_xz_rotation_layer(qubits, layer_num, name)
    circuit += new_circ
    if len(qubits) > 1:
      new_circ = get_cz_exp_layer(qubits, layer_num, name)
      circuit += new_circ
  return circuit


def get_zz_exp_layer(qubits, layer_num, name):
  """Apply ZZ gates to all pairs of nearest-neighbor qubits."""
  circuit = cirq.Circuit()
  for n, (q0, q1) in enumerate(zip(qubits[::2], qubits[1::2])):
    a = sympy.symbols("sc_{0}_{1}_{2}".format(name, layer_num, 2 * n))
    circuit += get_zz_exp(q0, q1, a)
  shifted_qubits = qubits[1::]
  for n, (q0, q1) in enumerate(zip(shifted_qubits[::2], shifted_qubits[1::2])):
    a = sympy.symbols("sc_{0}_{1}_{2}".format(name, layer_num, 2 * n + 1))
    circuit += get_zz_exp(q0, q1, a)
  return circuit


def hea_1d_zz(qubits, num_layers, name):
  """Build our full parameterized model unitary."""
  circuit = cirq.Circuit()
  for layer_num in range(num_layers):
    new_circ = get_xz_rotation_layer(qubits, layer_num, name)
    circuit += new_circ
    if len(qubits) > 1:
      new_circ = get_zz_exp_layer(qubits, layer_num, name)
      circuit += new_circ
  return circuit


# ============================================================================ #
# 2D HEA.
# ============================================================================ #


def get_2d_xz_rotation_layer(rows, cols, layer_num, name):
  """Apply single qubit rotations on a grid of qubits."""
  circuit = cirq.Circuit()
  for r in range(rows):
    for c in range(cols):
      sx = sympy.Symbol(f"sx_{name}_{layer_num}_{r}_{c}")
      sz = sympy.Symbol(f"sz_{name}_{layer_num}_{r}_{c}")
      circuit += get_xz_rotation(cirq.GridQubit(r, c), sx, sz)
  return circuit


def get_2d_cz_exp_layer(rows, cols, layer_num, name):
  """Apply CZ gates to all pairs of nearest-neighbor qubits on a grid."""
  circuit = cirq.Circuit()
  # Apply horizontal bonds
  for r in range(rows):
    for par in [0, 1]:
      for q_c_0, q_c_1 in zip(range(par, cols, 2), range(par + 1, cols, 2)):
        scz = sympy.Symbol(f"scz_{name}_{layer_num}_row{r}_{q_c_0}_{q_c_1}")
        circuit += get_cz_exp(
            cirq.GridQubit(r, q_c_0), cirq.GridQubit(r, q_c_1), scz)
  # Apply vertical bonds
  for c in range(cols):
    for par in [0, 1]:
      for q_r_0, q_r_1 in zip(range(par, rows, 2), range(par + 1, rows, 2)):
        scz = sympy.Symbol(f"scz_{name}_{layer_num}_col{c}_{q_r_0}_{q_r_1}")
        circuit += get_cz_exp(
            cirq.GridQubit(q_r_0, c), cirq.GridQubit(q_r_1, c), scz)
  return circuit


def get_2d_hea(rows, cols, num_layers, name):
  """Build a 2D HEA ansatz.

    Args:
      rows: int specifying the number of rows in the ansatz.
      cols: int specifying the number of columns in the ansatz.
      num_layers: int specifying how many layers of 2D HEA to apply.
      name: string which will be included in the parameters of the ansatz.

    Returns:
      circuit: `cirq.Circuit` which is the ansatz.
    """
  circuit = cirq.Circuit()
  for layer in range(num_layers):
    xz_circuit = get_2d_xz_rotation_layer(rows, cols, layer, name)
    circuit += xz_circuit
    cz_circuit = get_2d_cz_exp_layer(rows, cols, layer, name)
    circuit += cz_circuit
  return circuit


def get_2d_xyz_rotation_layer(rows, cols, layer_num, name):
  """Apply single qubit rotations on a grid of qubits."""
  circuit = cirq.Circuit()
  for r in range(rows):
    for c in range(cols):
      sx = sympy.Symbol(f"sx_{name}_{layer_num}_{r}_{c}")
      sy = sympy.Symbol(f"sy_{name}_{layer_num}_{r}_{c}")
      sz = sympy.Symbol(f"sz_{name}_{layer_num}_{r}_{c}")
      circuit += get_xyz_rotation(cirq.GridQubit(r, c), sx, sy, sz)
  return circuit


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
    """
  circuit = cirq.Circuit()
  for j in range(p):
    for n, h in enumerate(h_list):
      new_symb = sympy.Symbol("phi_{0}_L{1}_H{2}".format(name, j, n))
      circuit += tfq.util.exponential([h], coefficients=[new_symb])
  return circuit
