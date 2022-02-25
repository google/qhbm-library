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

from qhbmlib.data import quantum_data
from qhbmlib.inference import qhbm


def qmhl(data: quantum_data.QuantumData, input_qhbm: qhbm.QHBM):
  """Calculate the QMHL loss of the QHBM against the quantum data.

  See equation 21 in the appendix.

  Args:
    data: The data mixed state to learn.
    input_qhbm: QHBM being trained to approximate `data`.

  Returns:
    The quantum cross-entropy between the data and the model.
  """
  return (data.expectation(input_qhbm.modular_hamiltonian) +
          input_qhbm.e_inference.log_partition())
