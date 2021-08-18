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
"""Utility functions for running tests."""

import random

import cirq
import numpy as np
import scipy

import tensorflow as tf
import tensorflow_probability as tfp

from qhbmlib import architectures
from qhbmlib import ebm
from qhbmlib import qhbm
from qhbmlib import qnn


def get_random_qhbm(
    qubits,
    num_layers,
    identifier,
    minval_thetas=-1.0,
    maxval_thetas=1.0,
    minval_phis=-6.2,
    maxval_phis=6.2,
):
  """Create a random QHBM for use in testing."""
  num_qubits = len(qubits)
  this_ebm = ebm.Bernoulli(num_qubits, tf.keras.initializers.RandomUniform(minval=minval_thetas, maxval=maxval_thetas), analytic=False)
  unitary, phis_symbols = architectures.get_hardware_efficient_model_unitary(
      qubits, num_layers, identifier)
  this_qnn = qnn.QNN(unitary, phis_symbols, tf.keras.initializers.RandomUniform(minval=minval_phis, maxval=maxval_phis))
  return qhbm.QHBM(this_ebm, this_qnn, identifier)


def get_ebm_functions(num_bits):
  """EBM functions to use in a test QHBM.

    The test EBM will be a simple case where the parameters are bias energies
    for uncoupled bits.

    Args:
      num_bits: number of bits on which this EBM is defined.

    Returns:
      tuple of functions required as input args for analytic QHBMs.
    """

  def energy(thetas, bitstring):
    """Computes the energy of a bitstring."""
    spins = tf.subtract(
        tf.ones(num_bits, dtype=tf.int8),
        tf.constant(2, dtype=tf.int8) * tf.cast(bitstring, tf.int8),
    )
    return tf.reduce_sum(tf.cast(spins, tf.float32) * thetas)

  def sampler(thetas, num_samples):
    r"""Fairly samples from the EBM defined by `energy`.

        For Bernoulli distribution, we let $p$ be the probability of being `1`
        bit.
        In this case, $p = \frac{e^{theta}}{{e^{theta}+e^{-theta}}}$.
        Therefore, each independent logit is:
          $logit = \log\frac{p}{1-p} = \log\frac{e^{theta}}{e^{-theta}}
                 = \log{e^{2*theta}} = 2*theta$

        Args:
          thetas: a `tf.Tensor` of dtype `tf.float32` representing classical
            model parameters for classical energies. `tf.shape(thetas)[0] ==
            num_bits`.
          num_samples: a `tf.Tensor` of dtype `tf.int32` representing the number
            of samples from given Bernoulli distribition.

        Returns:
          a `tf.Tensor` in the shape of [num_samples, num_bits] of `tf.int8`
          with
          bitstrings sampled from the classical distribution.
        """
    return tfp.distributions.Bernoulli(
        logits=2 * thetas, dtype=tf.int8).sample(num_samples)

  return energy, sampler


def get_random_pauli_sum(qubits):
  """Test fixture.

    Args:
      qubits: A list of `cirq.GridQubit`s on which to build the pauli sum.

    Returns:
      pauli_sum: A `cirq.PauliSum` which is a linear combination of random pauli
      strings on `qubits`.
    """
  paulis = [cirq.X, cirq.Y, cirq.Z]

  coeff_max = 1.5
  coeff_min = -1.0 * coeff_max

  num_qubits = len(qubits)
  num_pauli_terms = num_qubits - 1
  num_pauli_factors = num_qubits - 1

  pauli_sum = cirq.PauliSum()
  for _ in range(num_pauli_terms):
    pauli_term = random.uniform(coeff_min, coeff_max) * cirq.I(qubits[0])
    sub_qubits = random.sample(qubits, num_pauli_factors)
    for q in sub_qubits:
      pauli_factor = random.choice(paulis)(q)
      pauli_term *= pauli_factor
    pauli_sum += pauli_term
  return pauli_sum


def generate_pure_random_density_operator(num_qubits):
  dim = 2**num_qubits
  unitary = scipy.stats.unitary_group.rvs(dim)
  u_vec = unitary[:, 0:1]
  return tf.matmul(u_vec, u_vec, adjoint_b=True)


def generate_mixed_random_density_operator(num_qubits, num_mixtures=5):
  """Generates a random mixed density matrix.

    Generates `num_mixtures` random quantum states, takes their outer products,
    and generates a random convex combination of them.

    NOTE: the states in the mixture are all orthogonal.

    Args:
      num_qubits: 2**num_qubits is the size of the density matrix.
      num_mixtures: the number of pure states in the mixture.  Must be greater
        than 2**num_qubits.

    Returns:
      final_state: The mixed density matrix.
      prob: The probability of each random state in the mixture.
    """
  prob = tf.random.uniform(shape=[num_mixtures])
  prob = prob / tf.reduce_sum(prob)
  final_state = tf.zeros((2**num_qubits, 2**num_qubits), dtype=tf.complex128)
  dim = 2**num_qubits
  unitary = scipy.stats.unitary_group.rvs(dim)
  for i in range(num_mixtures):
    u_vec = unitary[:, i:i + 1]
    final_state += tf.multiply(
        tf.cast(prob[i], dtype=tf.complex128),
        tf.matmul(u_vec, u_vec, adjoint_b=True),
    )
  return final_state, prob


def generate_mixed_random_density_operator_pair(num_qubits, perm_basis=False):
  num_mixtures = 5
  dim = 2**num_qubits
  # Use common basis
  unitary = scipy.stats.unitary_group.rvs(dim)

  prob = tf.random.uniform(shape=[num_mixtures])
  prob = prob / tf.reduce_sum(prob)
  final_state1 = tf.zeros((dim, dim), dtype=tf.complex128)
  basis_indices = np.random.permutation(dim)[:num_mixtures]
  for i, idx in enumerate(basis_indices):
    u_vec = unitary[:, idx:idx + 1]
    final_state1 += tf.multiply(
        tf.cast(prob[i], dtype=tf.complex128),
        tf.matmul(u_vec, u_vec, adjoint_b=True),
    )

  prob = tf.random.uniform(shape=[num_mixtures])
  prob = prob / tf.reduce_sum(prob)
  final_state2 = tf.zeros((dim, dim), dtype=tf.complex128)
  if perm_basis:
    basis_indices = np.random.permutation(dim)[:num_mixtures]
  for i, idx in enumerate(basis_indices):
    u_vec = unitary[:, idx:idx + 1]
    final_state2 += tf.multiply(
        tf.cast(prob[i], dtype=tf.complex128),
        tf.matmul(u_vec, u_vec, adjoint_b=True),
    )
  return final_state1, final_state2


def stable_classical_entropy(probs):
  """Entropy function for a list of probabilities, allowing zeros."""
  return -tf.reduce_sum(tf.math.multiply_no_nan(tf.math.log(probs), probs))


def check_bitstring_exists(bitstring, bitstring_list):
  """True if `bitstring` is an entry of `bitstring_list`."""
  return tf.math.reduce_any(
      tf.reduce_all(tf.math.equal(bitstring, bitstring_list), 1))
