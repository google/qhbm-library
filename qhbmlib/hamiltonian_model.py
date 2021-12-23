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
"""Tools for defining quantum Hamiltonian-based models."""

import tensorflow as tf

from qhbmlib import circuit_model
from qhbmlib import energy_model


class Hamiltonian(tf.keras.layers.Layer):
  """Hamiltonian representation of a QHBM."""

  def __init__(self,
               energy: energy_model.BitstringEnergy,
               circuit: circuit_model.QuantumCircuit):
    """Initializes a Hamiltonian."""
    if energy.num_bits != len(circuit.qubits):
      raise ValueError(
        "`energy` and `circuit` must act on the same number of bits.")
    self.energy = energy
    self.circuit = circuit


class CirqHamiltonian(Hamiltonian):
  """Wraps a cirq.PauliSum as a Hamiltonian object."""

  def __init__(self, ham: cirq.PauliSum):
    """Initializes a CirqHamiltonian."""
    
