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


def get_xz_rotation(q, a, b):
  """Two-axis single qubit rotation."""
  return cirq.Circuit(cirq.X(q)**a, cirq.Z(q)**b)


def get_cz_exp(q0, q1, a):
  """Exponent of entangling CZ gate."""
  return cirq.Circuit(cirq.CZPowGate(exponent=a)(q0, q1))


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
