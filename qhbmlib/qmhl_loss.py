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
"""Implementation of the QMHL loss function."""

from qhbmlib import hamiltonian_infer
from qhbmlib import hamiltonian_model
from qhbmlib import quantum_data


def qmhl(data: quantum_data.QuantumData,
         infer: hamiltonian_infer.QHBM,
         model: hamiltonian_model.Hamiltonian):
  """Calculate the QMHL loss of the QHBM against the quantum data.
  See equation 21 in the appendix.
  Args:
    data: The data mixed state to learn.
    infer: Inference engine for the model.
    model: Hamiltonian whose normalized exponential approximates `data`.
  Returns:
    The quantum cross-entropy between the data and the model.
  """
  infer.e_inference.infer(model.energy)
  return data.expectation(model) + infer.e_inference.log_partition()
