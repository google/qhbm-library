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

from typing import List, Union

import cirq
import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib import circuit_infer
from qhbmlib import energy_infer
from qhbmlib import energy_model
from qhbmlib import hamiltonian_model
from qhbmlib import util


class QHBM(tf.keras.layers.Layer):
  """Methods for inference on Hamiltonian objects."""

  def __init__(self,
               e_inference: energy_infer.EnergyInference,
               q_inference: circuit_infer.QuantumInference,
               name: Union[None, str] = None):
    """Initializes a QHBM.

    Args:
      e_inference: Attends to density operator eigenvalues.
      q_inference: Attends to density operator eigenvectors.
      name: Optional name for the model.
    """
    super().__init__(name=name)
    self._e_inference = e_inference
    self._q_inference = q_inference

  @property
  def e_inference(self):
    """The object used for inference on density operator eigenvalues."""
    return self._e_inference

  @property
  def q_inference(self):
    """The object used for inference on density operator eigenvectors."""
    return self._q_inference

  def circuits(self, model: hamiltonian_model.Hamiltonian, num_samples: int):
    """Draws pure states from the density operator.

    Args:
      model: The modular Hamiltonian whose normalized exponential is the
        density operator from which states will be approximately sampled.
      num_samples: Number of states to draw from the density operator.

    Returns:
      states: 1D `tf.Tensor` of dtype `tf.string`.  Each entry is a TFQ string
        representation of a state drawn from the density operator represented by
        the input `model`.
      counts: 1D `tf.Tensor` of dtype `tf.int32`.  `counts[i]` is the number of
        times `states[i]` was drawn from the density operator.
    """
    self.e_inference.infer(model.energy)
    samples = self.e_inference.sample(num_samples)
    bitstrings, counts = util.unique_bitstrings_with_counts(samples)
    states = model.circuit(bitstrings)
    return states, counts

  def expectation(self,
                  model: hamiltonian_model.Hamiltonian,
                  ops: Union[tf.Tensor, hamiltonian_model.Hamiltonian],
                  num_samples: int,
                  reduce: bool = True):
    """Estimates observable expectation values against the density operator.

    Args:
      model: The modular Hamiltonian whose normalized exponential is the
        density operator against which expectation values will be estimated.
      ops: The observable to measure.  If `tf.Tensor`, strings with shape
        [n_ops], result of calling `tfq.convert_to_tensor` on a list of
        cirq.PauliSum, `[op1, op2, ...]`. Else, a Hamiltonian.  Tiled to measure
        `<op_j>_((qnn)|initial_states[i]>)` for each bitstring i and op j.
      reduce: bool flag for whether or not to average over i.

    Returns:
      If `reduce` is true, a `tf.Tensor` with shape [n_ops] whose entries are
      are the batch-averaged expectation values of `operators`.
      Else, a `tf.Tensor` with shape [batch_size, n_ops] whose entries are the
      unaveraged expectation values of each `operator` against each `circuit`.
    """
    self.e_inference.infer(model.energy)
    samples = self.e_inference.sample(num_samples)
    bitstrings, counts = util.unique_bitstrings_with_counts(samples)
    if isinstance(ops, tf.Tensor):
      return self.q_inference.expectation(
          model.circuit, bitstrings, counts, ops, reduce=reduce)
    elif isinstance(ops.energy, energy_model.PauliMixin):
      u_dagger_u = model.circuit + ops.circuit_dagger
      operator_shards = ops.energy.operator_shards(ops.circuit.qubits)
      expectation_shards = self.q_inference.expectation(
          u_dagger_u, bitstrings, counts, operator_shards, reduce=reduce)
      return ops.energy.operator_expectation(expectation_shards)
    else:
      raise NotImplementedError(
          "General `BitstringEnergy` models not yet supported.")

  def sample(self, model: hamiltonian_model.Hamiltonian, num_samples: int):
    """Repeatedly measures the density operator in the computational basis."""
    self.e_inference.infer(model.energy)
    samples = self.e_inference.sample(num_samples)
    bitstrings, counts = util.unique_bitstrings_with_counts(samples)
    return self.q_inference.sample(model.circuit, bitstrings, counts)
