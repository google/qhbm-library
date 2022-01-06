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
                  ops: tf.Tensor,
                  num_samples: int):
    """Estimates observable expectation values against the density operator.

    TODO(#119): add expectation and derivative equations and discussions
                from updated paper.

    Args:
      model: The modular Hamiltonian whose normalized exponential is the
        density operator against which expectation values will be estimated.
      ops: The observables to measure.  If `tf.Tensor`, strings with shape
        [n_ops], result of calling `tfq.convert_to_tensor` on a list of
        cirq.PauliSum, `[op1, op2, ...]`.  Tiled to measure `<op_j>_((qnn)|x>)`
        once for each  unique sample `x` drawn from the EBM associated with
        `model`.
      num_samples: Number of draws from the EBM associated with `model` to
        average over.

      Returns:
        `tf.Tensor` with shape [n_ops] whose entries are are the sample averaged
        expectation values of each entry in `ops`.
      """
    self.e_inference.infer(model.energy)
    samples = self.e_inference.sample(num_samples)
    bitstrings, counts = util.unique_bitstrings_with_counts(samples)
    return self.q_inference.expectation(
        model.circuit, bitstrings, counts, ops, reduce=True)

  def sample(self, model: hamiltonian_model.Hamiltonian, num_samples: int):
    """Repeatedly measures the density operator in the computational basis.

    # TODO(#119) align the notation and discussion below with updated paper.

    Here we discuss the algorithm used to approximately sample from the density
    operator represented by `model`.

    Consider the probability distribution p implicitly represented by
    `model.energy`, and assume we can approximately sample x ~ X, where X is the
    random variable X controlled by p.  Draw a list of samples {x} of size
    `num_samples`.  From {x}, build a list of unique samples {y} and a list of
    counts {c} such that {c}[i] is the number of times {y}[i] appears in {x}.

    Next, for each index `i`, realize the quantum state `u |{y}[i]>` where u is
    the quantum circuit represented by `model.circuit`.  Then, measure the state
    `u |{y}[i]>` in the computational basis {c}[i] times, yielding a list of
    bitstrings {b_i}.  This function returns the concatenation of {b_i} over
    all indices `i`.

    Args:
      model: The modular Hamiltonian whose normalized exponential is the
        density operator against which expectation values will be estimated.
      num_samples: Number of draws from the EBM associated with `model` to
        draw as inputs to the quantum circuit of `model`. and use for sampling.

    Returns:
      
    """
    self.e_inference.infer(model.energy)
    samples = self.e_inference.sample(num_samples)
    bitstrings, counts = util.unique_bitstrings_with_counts(samples)
    return self.q_inference.sample(model.circuit, bitstrings, counts)
