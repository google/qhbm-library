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

import tensorflow as tf

from qhbmlib import hamiltonian_infer
from qhbmlib import hamiltonian_model


class QuantumData(abc.ABC):
  """Interface for quantum datasets."""

  @abc.abstractmethod
  def expectation(self, ops):
    """Take the expectation value of ops against this dataset."""
    raise NotImplementedError()

  @abc.abstractmethod
  def sample(self, ops, num_samples):
    """Take measurements according to ops."""
    raise NotImplementedError()


class QHBMData(QuantumData):
  """QuantumData defined by a QHBM."""

  def __init__(self, infer, model, num_expectation_samples):
    """Initializes a QHBMData."""
    self.infer = infer
    self.model = model
    self.num_expectation_samples = num_expectation_samples

  def expectation(self, ops):
    """See base class docstring."""
    return self.infer.expectation(self.model, ops, self.num_expectation_samples)

  def sample(self, ops, num_samples):
    """See base class docstring."""
    return self.infer.sample(self.model, ops, num_samples)
