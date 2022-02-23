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
"""Interface to quantum data sources defined by QHBMs."""

from typing import Union

import tensorflow as tf

from qhbmlib.data import quantum_data
from qhbmlib.inference import qhbm
from qhbmlib.models import hamiltonian


class QHBMData(quantum_data.QuantumData):
  """QuantumData defined by a QHBM."""

  def __init__(self, input_qhbm: qhbm.QHBM):
    """Initializes a QHBMData.

    Args:
      qhbm: An inference engine for a QHBM.
    """
    self.qhbm = input_qhbm

  def expectation(self, observable: Union[tf.Tensor, hamiltonian.Hamiltonian]):
    """See base class docstring."""
    return tf.squeeze(self.qhbm.expectation(observable), 0)
