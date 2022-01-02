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
"""Tools for inference on quantum Hamiltonians."""

from typing import List, Type

import cirq
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib import circuit_infer
from qhbmlib import circuit_model
from qhbmlib import energy_infer
from qhbmlib import energy_model
from qhbmlib import hamiltonian_model
from qhbmlib import util


class QHBM(tf.keras.layers.Layer):
  """Methods for inference on Hamiltonian objects."""

  def __init__(self,
               e_inference: energy_infer.InferenceLayer,
               q_inference: circuit_infer.QuantumInference):
    """Initializes a QHBM."""
    self._e_inference = e_inference
    self._q_inference = q_inference

  @property
  def e_inference(self):
    return self._e_inference

  @property
  def q_inference(self):
    return self._q_inference

  def circuits(self,
               model: hamiltonian_model.Hamiltonian,
               num_samples: int):
    self.e_inference.infer(model.energy)
    samples = self.e_inference.sample(num_samples)
    bitstrings, counts = util.unique_bitstrings_with_counts(samples)
    return model.circuit(bitstrings), counts

  def expectation(self,
                  model: hamiltonian_model.Hamiltonian,
                  ops: Union[tf.Tensor, hamiltonian_model.Hamiltonian],
                  num_samples: int,
                  reduce: bool=True):
    self.e_inference.infer(model.energy)
    samples = self.e_inference.sample(num_samples)
    bitstrings, counts = util.unique_bitstrings_with_counts(samples)
    if isinstance(ops, tf.Tensor):
      return self.q_inference.expectation(
          model.circuit, bitstrings, counts, ops, reduce=reduce)
    elif isinstance(model.energy, energy_model.PauliMixin):
      u_dagger_u = model.circuit + op.circuit_dagger
      operator_shards = model.energy.operator_shards(model.circuit.qubits)
      expectation_shards = self.q_inference.expectation(
          u_dagger_u, bitstrings, counts, operator_shards, reduce=reduce)
      return model.energy.operator_expectation(expectation_shards)
    else:
      raise NotImplementedError(
          "General `BitstringEnergy` models not yet supported.")

  def sample(self,
             model: hamiltonian_model.Hamiltonian,
             num_samples: int):
    self.e_inference.infer(model.energy)
    samples = self.e_inference.sample(num_samples)
    bitstrings, counts = util.unique_bitstrings_with_counts(samples)
    return self.q_inference.sample(model.circuit, bitstrings, counts)
