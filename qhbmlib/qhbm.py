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
"""Implementation of general QHBMs in TFQ."""

import inspect
import itertools
import numbers
from typing import Any, Callable, Iterable, List, Union

import cirq
import sympy
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_quantum as tfq


class QHBM(tf.keras.Model):

  def __init__(self, ebm, qnn, name=None):
    super().__init__(name=name)
    self._ebm = ebm
    self._qnn = qnn
    if ebm.has_operator:
      self._operator_shards = tfq.convert_to_tensor(
          ebm.operator_shards(qnn.raw_qubits))

  @property
  def ebm(self):
    return self._ebm

  @property
  def qnn(self):
    return self._qnn

  @property
  def trainable_variables(self):
    return self.ebm.trainable_variables + self.qnn.trainable_variables

  @trainable_variables.setter
  def trainable_variables(self, value):
    self.ebm.trainable_variables = value[:len(self.ebm.trainable_variables)]
    self.qnn.trainable_variables = value[len(self.ebm.trainable_variables):]

  @property
  def operator_shards(self):
    if self.ebm.has_operator:
      return self._operator_shards
    raise NotImplementedError()

  @property
  def raw_qubits(self):
    return self.qnn.raw_qubits

  @property
  def qubits(self):
    return self.qnn.qubits

  @property
  def is_analytic(self):
    return self.ebm.is_analytic and self.qnn.is_analytic

  def copy(self):
    return QHBM(self.ebm.copy(), self.qnn.copy(), name=self.name)

  def circuits(self, num_samples, unique=True, resolve=True):
    if unique:
      bitstrings, counts = self.ebm.sample(num_samples, unique=unique)
      circuits = self.qnn.circuits(bitstrings, resolve=resolve)
      return circuits, counts
    bitstrings = self.ebm.sample(num_samples, unique=unique)
    circuits = self.qnn.circuits(bitstrings, resolve=resolve)
    return circuits

  def sample(self, num_samples, mask=True, reduce=True, unique=True):
    bitstrings, counts = self.ebm.sample(num_samples)
    return self.qnn.sample(
        bitstrings, counts, mask=mask, reduce=reduce, unique=unique)

  def expectation(self, operators, num_samples, mask=True, reduce=True):
    """TODO: add gradient function"""
    if isinstance(operators, QHBM):
      circuits, counts = self.circuits(num_samples, resolve=False)
      return operators.operator_expectation((circuits, counts),
                                            symbol_names=self.qnn.symbols,
                                            symbol_values=self.qnn.values,
                                            mask=mask,
                                            reduce=reduce)
    bitstrings, counts = self.ebm.sample(num_samples)
    return self.qnn.expectation(bitstrings, counts, operators, reduce=reduce)

  def operator_expectation(self,
                           density_operator,
                           num_samples=None,
                           symbol_names=None,
                           symbol_values=None,
                           reduce=True,
                           mask=True):
    """TODO: add gradient function"""
    if isinstance(density_operator, tuple):
      circuits, counts = density_operator
    elif isinstance(density_operator, QHBM):
      circuits, counts = density_operator.circuits(num_samples, resolve=False)
      symbol_names = density_operator.qnn.symbols
      symbol_values = density_operator.qnn.values
    else:
      raise TypeError()

    if self.ebm.has_operator:
      expectation_shards = self.qnn.pulled_back_expectation(
          circuits,
          counts,
          self.operator_shards,
          symbol_names=symbol_names,
          symbol_values=symbol_values,
          reduce=reduce)
      return self.ebm.operator_expectation(expectation_shards)
    bitstrings, counts = self.qnn.pulled_back_sample(
        circuits, counts, mask=mask)
    energies = self.ebm.energy(bitstrings)
    if reduce:
      probs = tf.cast(counts, tf.float32) / tf.cast(
          tf.reduce_sum(counts), tf.float32)
      return tf.reduce_sum(probs * energies)
    return energies

  def probabilities(self):
    return self.ebm.probabilities()

  def log_partition_function(self):
    return self.ebm.log_partition_function()

  def entropy(self):
    return self.ebm.entropy()

  def unitary_matrix(self):
    return self.qnn.pqc_unitary()

  def density_matrix(self):
    probabilities = tf.cast(self.probabilities(), tf.complex64)
    unitary_matrix = self.unitary_matrix()
    unitary_probs = tf.multiply(
        unitary_matrix,
        tf.tile(
            tf.expand_dims(probabilities, 0), [tf.shape(unitary_matrix)[0], 1]))
    return tf.matmul(unitary_probs, tf.linalg.adjoint(unitary_matrix))

  def fidelity(self, sigma: tf.Tensor):
    """TODO: convert to tf.keras.metric.Metric

    Calculate the fidelity between a QHBM and a density matrix.
        Args:
          sigma: 2-D `tf.Tensor` of dtype `complex64` representing the right
            density matrix in the fidelity calculation.

        Returns:
          A scalar `tf.Tensor` which is the fidelity between the density matrix
            represented by this QHBM and `sigma`.
        """
    e_rho = tf.cast(self.probabilities(), tf.complex128)
    v_rho = tf.cast(self.unitary_matrix(), tf.complex128)
    sqrt_e_rho = tf.sqrt(e_rho)
    v_rho_sqrt_e_rho = tf.multiply(
        v_rho, tf.tile(tf.expand_dims(sqrt_e_rho, 0), (tf.shape(v_rho)[0], 1)))
    rho_sqrt = tf.linalg.matmul(v_rho_sqrt_e_rho, tf.linalg.adjoint(v_rho))
    omega = tf.linalg.matmul(
        tf.linalg.matmul(rho_sqrt, tf.cast(sigma, tf.complex128)), rho_sqrt)
    # TODO(zaqqwerty): find convincing proof that omega is hermitian,
    # in order to go back to eigvalsh.
    e_omega = tf.linalg.eigvals(omega)
    return tf.cast(
        tf.math.abs(tf.math.reduce_sum(tf.math.sqrt(e_omega)))**2, tf.float32)
