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
"""Tools defining energy functions."""


class BitstringDistribution(tf.keras.Model, abc.ABC):
  """Class for representing a probability distribution over bitstrings."""

  def __init__(self, name=None):
    super().__init__(name=name)

  @property
  @abc.abstractmethod
  def bits(self):
    """Integer labels for the bits on which this distribution acts."""
    raise NotImplementedError()

  @property
  def trainable_variables(self):
    return super().trainable_variables

  @trainable_variables.setter
  @abc.abstractmethod
  def trainable_variables(self, value):
    raise NotImplementedError()

  @abc.abstractmethod
  def energy(self, bitstrings):
    raise NotImplementedError()


class PauliBitstringDistribution(BitstringDistribution):
  """Augments a BitstringDistribution with a Pauli Z representation."""

  @abc.abstractmethod
  def operator_shards(self, qubits: list[cirq.GridQubit]) -> list[cirq.PauliString]:
    raise NotImplementedError()

  @abc.abstractmethod
  def operator_expectation(self, expectation_shards: list[float]) -> float:
    raise NotImplementedError()
