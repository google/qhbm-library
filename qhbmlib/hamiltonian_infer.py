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

from typing import Union

import tensorflow as tf

from qhbmlib import circuit_infer
from qhbmlib import energy_infer
from qhbmlib import hamiltonian_model
from qhbmlib import utils


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
    r"""Draws thermally distributed eigenstates from the model Hamiltonian.

    We define some terms that will be used to explain the algorithm.
    `model` is some modular Hamiltonian
    $$K_{\theta\phi} = U_\phi K_\theta U_\phi^\dagger.$$
    The [thermal state][1] corresponding to the model is
    $$ \rho_T = Z^{-1} e^{-\beta K_{\theta\phi}}.$$
    For QHBMs, we fix $\beta = 1$, effectively absorbing it into the definition
    of the modular Hamiltonian.  Then $\rho_T$ can be expanded as
    $$\rho_T = \sum_x p_\theta(x)U_\phi\ket{x}\bra{x}U_\phi^\dagger,$$
    where the probability is given by
    $$p_\theta(x) = \tr[\exp(-K_\theta)]\bra{x}\exp(-K_\theta)\ket{x}$$
    for $x\in\{1, \ldots, \dim(K_{\theta\phi})\} = \mathcal{X}$. Note that each
    $U_\phi\ket{x}$ is an eigenvector of both $\rho_T$ and $K_{\theta\phi}$.
    Corresponding to this operator is the [ensemble of quantum states][2]
    $$\mathcal{E} = \{p_\theta(x), U_\phi\ket{x}\}_{x\in\mathcal{X}}.$$
    This function returns pure state samples from the ensemble.

    We now explain the algorithm.  First, construct $X$ to be a classical
    random variable with probability distribution $p_\theta(x)$ set by
    `model.energy`.  Then, draw $n = $`num\_samples` bitstrings,
    $S=\{x_1, \ldots, x_n\}$, from $X$.  For each unique $x_i\in S$, set
    `states[i]` to the TFQ string representation of $U_\phi\ket{x_i}$, where
    $U_\phi$ is set by `model.circuit`.  Finally, set `counts[i]` equal to the
    number of times $x_i$ occurs in $S$.

    #### References
    [1]: Nielsen, Michael A. and Chuang, Isaac L. (2010).
         Quantum Computation and Quantum Information.
         Cambridge University Press.
    [2]: Wilde, Mark M. (2017).
         Quantum Information Theory (second edition).
         Cambridge University Press.

    Args:
      model: The modular Hamiltonian whose normalized exponential is the
        density operator governing the ensemble of states from which to sample.
      num_samples: Number of states to draw from the ensemble.

    Returns:
      states: 1D `tf.Tensor` of dtype `tf.string`.  Each entry is a TFQ string
        representation of an eigenstate of the Hamiltonian `model`.
      counts: 1D `tf.Tensor` of dtype `tf.int32`.  `counts[i]` is the number of
        times `states[i]` was drawn from the ensemble.
    """
    self.e_inference.infer(model.energy)
    samples = self.e_inference.sample(num_samples)
    bitstrings, counts = utils.unique_bitstrings_with_counts(samples)
    states = model.circuit(bitstrings)
    return states, counts
