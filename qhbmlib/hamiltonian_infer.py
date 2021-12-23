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
"""Tools for inference on Hamiltonians objects."""

from typing import List, Type

import cirq
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib import circuit_infer
from qhbmlib import energy_infer
from qhbmlib import hamiltonian_model
from qhbmlib import util


class QHBM(tf.keras.layers.Layer):
  """Methods for inference on Hamiltonian objects."""

  def __init__(self,
               e_inference: energy_infer.InferenceLayer,
               q_inference: circuit_infer.QuantumInference):
    """Initializes a QHBM."""
    self._ebm = e_inference
    self._qnn = q_inference

  @property
  def ebm(self):
    return self._ebm

  @property
  def qnn(self):
    return self._qnn

  def circuits(self, num_samples):
    samples = self.ebm.sample(num_samples)
    bitstrings, counts = util.unique_bitstrings_with_counts(samples)
    return self.qnn.qnn(bitstrings), counts

  def expectation(self,
                  ops: List[cirq.PauliSum],
                  num_samples: int,
                  reduce: bool=True):
    samples = self.ebm.sample(num_samples)
    bitstrings, counts = util.unique_bitstrings_with_counts(samples)
    return self.qnn.expectation(bitstrings, counts, operators, reduce=reduce)

  def expectation_hamiltonian(self,
                              ops: List[hamiltonian_model.Hamiltonian],
                              num_samples: int,
                              reduce: bool=True):
    
