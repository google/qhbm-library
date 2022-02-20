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
"""Utilities for metrics on QuantumCircuit."""

import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib import circuit_model


def unitary(circuit: circuit_model.QuantumCircuit):
  """Returns the unitary matrix corresponding to the given circuit.

  Args:
    circuit: Quantum circuit whose unitary matrix is to be calculated.
  """
  return tfq.layers.Unitary()(
      circuit.pqc,
      symbol_names=circuit.symbol_names,
      symbol_values=tf.expand_dims(circuit.symbol_values, 0)).to_tensor()[0]
