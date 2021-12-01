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
"""Tools for working with Modular Hamiltonians."""


from qhbmlib import energy
from qhbmlib import circuit


class ModHam(tf.keras.Model):
  """Represents a Modular Hamiltonian."""

  def __init__(self, distribution: energy.BitstringDistribution, basis_transform: circuit.QuantumCircuit):
    if not len(basis_transform.raw_qubits) == energy_function.num_bits:
      raise ValueError("The energy function and basis transformation "
                       "must act on the same set of qubits.")
    self.distribution = distribution
    self.basis_transform = basis_transform

  @property
  def trainable_variables(self):
    return self.energy_function.trainable_variables + self.basis_transform.trainable_variables

  @trainable_variables.setter
  def trainable_variables(self, value):
    self.energy_function.trainable_variables = value[:len(self.ebm.trainable_variables)]
    self.basis_transform.trainable_variables = value[len(self.ebm.trainable_variables):]

  def __add__(self, other):
    if isinstance(other, ModHam):
      return ModHamSum([self, other])
    # TODO(zaqqwerty): enable ModHamSums here
    raise TypeError(
      "ModHam can only be added to another ModHam.")
  
  def __sub__(self, other):
    new_other = -1 * other
    return self + new_other


class ModHamSum(tf.keras.Model):
  """Represents a sum of Modular Hamiltonians."""

  def __init__(self, summands: list[ModHam]):
    self.summands = summands


class Expectation:
  """Manages taking expectations of circuits against modular Hamiltonians."""
