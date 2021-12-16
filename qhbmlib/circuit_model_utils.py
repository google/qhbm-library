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
"""Utilities for the circuit_model module."""

from typing import List

import cirq
import sympy


def bit_circuit(qubits: List[cirq.GridQubit], name="bit_circuit"):
  """Returns exponentiated X gate on each qubit and the exponent symbols."""
  circuit = cirq.Circuit()
  for n, q in enumerate(qubits):
    bit = sympy.Symbol("{0}_bit_{1}".format(name, n))
    circuit += cirq.X(q)**bit
  return circuit
