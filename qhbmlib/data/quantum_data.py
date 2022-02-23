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
"""Interface to quantum data sources."""

import abc
from typing import Union

import tensorflow as tf

from qhbmlib.models import hamiltonian


class QuantumData(abc.ABC):
  """Interface for quantum datasets."""

  @abc.abstractmethod
  def expectation(self, observable: Union[tf.Tensor, hamiltonian.Hamiltonian]):
    """Take the expectation value of an observable against this dataset.

    Args:
      observable: Hermitian operator to measure.  If `tf.Tensor`, it is of type
        `tf.string` with shape [1], result of  calling `tfq.convert_to_tensor`
        on a list of `cirq.PauliSum`, `[op]`.  Otherwise, a Hamiltonian.

    Returns:
      Scalar `tf.Tensor` which is the expectation value of `observable` against
        this quantum data source.
    """
    raise NotImplementedError()
