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

from typing import Union

import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib import circuit_model
from qhbmlib import energy_model


class Hamiltonian(tf.keras.layers.Layer):
  """Diagonalized (spectral) representation of a Hermitian operator."""

  def __init__(self,
               energy: energy_model.BitstringEnergy,
               circuit: circuit_model.QuantumCircuit,
               name: Union[None, str] = None):
    """Initializes a Hamiltonian.

    Args:
      energy: Represents the eigenvalues of this operator.
      circuit: Represents the eigenvectors of this operator.
      name: Optional name for the model.
    """
    super().__init__(name=name)
    if energy.num_bits != len(circuit.qubits):
      raise ValueError(
          "`energy` and `circuit` must act on the same number of bits.")
    self.energy = energy
    self.circuit = circuit
    self.circuit_dagger = circuit**-1

    self.operator_shards = None
    if isinstance(self.energy, energy_model.PauliMixin):
      self.operator_shards = tfq.convert_to_tensor(
          self.energy.operator_shards(self.circuit.qubits))
