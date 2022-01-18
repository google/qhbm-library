# pylint: skip-file
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
"""Module for defining and sampling from orthogonal sets of QNNs."""
import numbers
from typing import Any, Callable, Iterable, List, Union

import cirq
import sympy
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
from qhbmlib import util
from qhbmlib import utils


def bit_circuit(qubits, name="bit_circuit"):
  """Returns exponentiated X gate on each qubit and the exponent symbols."""
  circuit = cirq.Circuit()
  for n, q in enumerate(qubits):
    bit = sympy.Symbol("{0}_bit_{1}".format(name, n))
    circuit += cirq.X(q)**bit
  return circuit


class QNN(tf.keras.Model):
  """Operations on parameterized unitaries with bitstring inputs."""

  def __init__(
      self,
      pqc,
      *,
      symbols=None,
      values=None,
      initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi),
      backend="noiseless",
      differentiator=None,
      is_analytic=False,
      name=None,
  ):
    """Initializes a QNN.

    Args:
      pqc: Representation of a parameterized quantum circuit.
      symbols: Optional 1-D `tf.Tensor` of strings which are the parameters of
        the QNN.  When `None`, parameters are inferred from the given PQC.
      values: Optional 1-D `tf.Tensor` of floats which are the parameter values
        corresponding to the symbols.  When `None`, parameters are chosen via
        `initializer` instead.
      initializer: A "tf.keras.initializers.Initializer" which specifies how to
        initialize the values of the parameters in `circuit`.  This argument is
        ignored if `values` is not None.
      backend: Optional Python `object` that specifies what backend TFQ will use
        for operations involving this QNN. Options are {"noisy", "noiseless"},
        or however users may also specify a preconfigured cirq execution
        object to use instead, which must inherit `cirq.Sampler`.
      differentiator: Either None or a `tfq.differentiators.Differentiator`,
        which specifies how to take the derivative of a quantum circuit.
      is_analytic: bool flag that enables is_analytic methods. If True, then backend
        must also be "noiseless".
      name: Identifier for this QNN.
    """
    super().__init__(name=name)

    if not isinstance(pqc, cirq.Circuit):
      raise TypeError("pqc must be a cirq.Circuit object."
                      " Given: {}".format(pqc))

    if symbols is None:
      raw_symbols = list(sorted(tfq.util.get_circuit_symbols(pqc)))
      symbols = tf.constant([str(x) for x in raw_symbols], dtype=tf.string)
    self._symbols = symbols

    if values is None:
      values = initializer(shape=[tf.shape(self._symbols)[0]])
    self.values = tf.Variable(
        initial_value=values, name=f"{self.name}_pqc_values")

    self._pqc = tfq.convert_to_tensor([pqc])
    self._inverse_pqc = tfq.convert_to_tensor([pqc**-1])

    self._raw_qubits = sorted(pqc.all_qubits())
    self._qubits = util.qubits_to_indices(self._raw_qubits)

    _bit_circuit = bit_circuit(self._raw_qubits)
    bit_symbols = list(sorted(tfq.util.get_circuit_symbols(_bit_circuit)))
    self._bit_symbols = tf.constant([str(x) for x in bit_symbols])
    self._bit_circuit = tfq.convert_to_tensor([_bit_circuit])

    self._differentiator = differentiator
    self._sample_layer = tfq.layers.Sample(backend=backend)
    if backend == "noiseless" or backend is None:
      self._backend = "noiseless"
      self._is_analytic = is_analytic
      self._expectation_layer = tfq.layers.Expectation(
          backend=backend, differentiator=differentiator)
    else:
      self._backend = backend
      self._is_analytic = False
      self._expectation_layer = tfq.layers.SampledExpectation(
          backend=backend, differentiator=differentiator)

    if self.is_analytic:
      self._unitary_layer = tfq.layers.Unitary()

  @property
  def raw_qubits(self):
    return self._raw_qubits

  @property
  def qubits(self):
    return self._qubits

  @property
  def symbols(self):
    return self._symbols

  @property
  def trainable_variables(self):
    return [self.values]

  @trainable_variables.setter
  def trainable_variables(self, value):
    self.values = value[0]

  @property
  def backend(self):
    return self._backend

  @property
  def differentiator(self):
    return self._differentiator

  @property
  def is_analytic(self):
    return self._is_analytic

  def copy(self):
    qnn = QNN(
        tfq.from_tensor(self.pqc(resolve=False))[0],
        backend=self.backend,
        differentiator=self.differentiator,
        is_analytic=self.is_analytic,
        name=self.name)
    qnn.values.assign(self.values)
    return qnn

  def _sample_function(self,
                       circuits,
                       counts,
                       mask=True,
                       reduce=True,
                       unique=True):
    """General function for sampling from circuits."""
    samples = self._sample_layer(
        circuits, repetitions=tf.expand_dims(tf.math.reduce_max(counts), 0))
    if mask:
      num_samples_mask = tf.cast((tf.ragged.range(counts) + 1).to_tensor(),
                                 tf.bool)
      num_samples_mask = tf.map_fn(tf.random.shuffle, num_samples_mask)
      samples = tf.ragged.boolean_mask(samples, num_samples_mask)
    if unique:
      samples = samples.values.to_tensor()
      return utils.unique_bitstrings_with_counts(samples)
    elif reduce:
      samples = samples.values.to_tensor()
    return samples

  def _expectation_function(self, circuits, counts, operators, reduce=True):
    """General function for taking sampled expectations from circuits.

    `counts[i]` sets the weight of `circuits[i]` in the expectation.
    Additionally, if `self.is_analytic` is false, `counts[i]` samples are drawn
    from `circuits[i]` and used to compute each expectation in `operators`.
    """
    num_circuits = tf.shape(circuits)[0]
    num_operators = tf.shape(operators)[0]
    tiled_values = tf.tile(tf.expand_dims(self.values, 0), [num_circuits, 1])
    tiled_operators = tf.tile(tf.expand_dims(operators, 0), [num_circuits, 1])
    if self.backend == "noiseless":
      expectations = self._expectation_layer(
          circuits,
          symbol_names=self.symbols,
          symbol_values=tiled_values,
          operators=tiled_operators,
      )
    else:
      expectations = self._expectation_layer(
          circuits,
          symbol_names=self.symbols,
          symbol_values=tiled_values,
          operators=tiled_operators,
          repetitions=tf.tile(tf.expand_dims(counts, 1), [1, num_operators]),
      )
    if reduce:
      probs = tf.cast(counts, tf.float32) / tf.cast(
          tf.reduce_sum(counts), tf.float32)
      return tf.reduce_sum(tf.transpose(probs * tf.transpose(expectations)), 0)
    return expectations

  def pqc(self, resolve=True):
    if resolve:
      return tfq.resolve_parameters(self._pqc, self.symbols,
                                    tf.expand_dims(self.values, 0))
    return self._pqc

  def inverse_pqc(self, resolve=True):
    if resolve:
      return tfq.resolve_parameters(self._inverse_pqc, self.symbols,
                                    tf.expand_dims(self.values, 0))
    return self._inverse_pqc

  def circuits(self, bitstrings, resolve=True):
    """Returns the current circuits for this QNN given bitstrings.

      Args:
        bitstrings: 2D tensor of dtype `tf.int8` whose entries are bits. These
          specify the state inputs to use in the returned set of circuits.
        resolve: bool which says whether or not to resolve the QNN
          unitary before appending to the bit injection circuits.

      Returns:
        1D tensor of strings which represent the current QNN circuits.
      """
    num_bitstrings = tf.shape(bitstrings)[0]
    bit_circuits = tfq.resolve_parameters(
        tf.tile(self._bit_circuit, [num_bitstrings]), self._bit_symbols,
        tf.cast(bitstrings, tf.float32))
    pqcs = tf.tile(self.pqc(resolve=resolve), [num_bitstrings])
    return tfq.append_circuit(bit_circuits, pqcs)

  def sample(self, bitstrings, counts, mask=True, reduce=True, unique=True):
    """Returns bitstring samples from the QNN.

      Args:
        bitstrings: 2D tensor of dtype `tf.int8` whose entries are bits. These
          specify the state inputs to the unitary of this QNN.
        counts: 1D tensor of dtype `tf.int32` such that `counts[i]` is the
          number of samples to draw from `self.pqc|bitstrings[i]>`.

      Returns:
        ragged_samples: `tf.RaggedTensor` of DType `tf.int8` structured such
          that `ragged_samples[i]` contains `counts[i]` bitstrings drawn from
          `self.u|bitstrings[i]>`.
    """
    circuits = self.circuits(bitstrings)
    return self._sample_function(
        circuits, counts, mask=mask, reduce=reduce, unique=unique)

  def expectation(self, bitstrings, counts, operators, reduce=True):
    """Returns the expectation values of the operators against the QNN.

      Args:
        bitstrings: 2D tensor of dtype `tf.int8` whose entries are bits.
        counts: 1D tensor of dtype `tf.int32` such that `counts[i]` is the
          relative weight of `bitstrings[i]` when computing expectations.
        operators: 1D tensor of strings, the result of calling
          `tfq.convert_to_tensor` on a list of cirq.PauliSum, `[op1, op2, ...]`.
          Will be tiled to measure `<opj>_self.u_dagger|circuits[i]>`
          for each i and j.
        reduce: bool flag for whether or not to average over i.

      Returns:
        1-D tensor of floats which are the bitstring-averaged expectation values
        if `reduce` is True, else 2-D tensor of floats which are per-bitstring
        expectation values.
      """
    circuits = self.circuits(bitstrings, resolve=False)
    return self._expectation_function(
        circuits, counts, operators, reduce=reduce)

  def pulled_back_circuits(self, circuits, resolve=True):
    """Returns the pulled back circuits for this QNN given input quantum data.

      Args:
        circuits: 1-D `tf.Tensor` of type `tf.string` which contains
          circuits serialized by `tfq.convert_to_tensor`. These represent pure
          state samples from the data density matrix.
        resolve: bool tensor which says whether or not to resolve the QNN
          inverse unitary before appending to the data circuits.

      Returns:
        1D tensor of strings which represent the pulled back circuits.
      """
    inverse_pqcs = tf.tile(
        self.inverse_pqc(resolve=resolve), [tf.shape(circuits)[0]])
    return tfq.append_circuit(circuits, inverse_pqcs)

  def pulled_back_sample(self,
                         circuits,
                         counts,
                         mask=True,
                         reduce=True,
                         unique=True):
    """Returns samples from the pulled back data distribution.

      The inputs represent the data density matrix. The inverse of `self.u`
      is appended to create the set of circuits representing the pulled back
      data density matrix. Then, the requested number of bitstrings are sampled
      from each circuit.

      Args:
        circuits: 1-D `tf.Tensor` of type `tf.string` which contains
          circuits serialized by `tfq.convert_to_tensor`. These represent pure
          state samples from the data density matrix.
        counts: 1-D `tf.Tensor` of type `tf.int32`, must be the same size as
          `circuits`. Contains the number of samples to draw from each
          input circuit.

      Returns:
        ragged_samples: `tf.RaggedTensor` of DType `tf.int8` structured such
            that `ragged_samples[i]` contains `counts[i]` bitstrings.
      """
    pulled_back_circuits = self.pulled_back_circuits(circuits)
    return self._sample_function(
        pulled_back_circuits, counts, mask=mask, reduce=reduce, unique=unique)

  def pulled_back_expectation(self, circuits, counts, operators, reduce=True):
    """Returns the expectation values for a given pulled-back dataset.

      Args:
        circuits: 1-D `tf.Tensor` of type `tf.string` which contains
          circuits serialized by `tfq.convert_to_tensor`. These represent pure
          state samples from the data density matrix.
        counts: 1D tensor of dtype `tf.int32` such that `counts[i]` is the
          relative weight of `circuits[i]` when computing expectations.
        operators: 1D tensor of strings, the result of calling
          `tfq.convert_to_tensor` on a list of cirq.PauliSum, `[op1, op2, ...]`.
          Will be tiled to measure `<opj>_self.u_dagger|circuits[i]>`
          for each i and j, then averaged over i.

      Returns:
        1-D tensor of floats which are the bitstring-averaged expectation values
        if `reduce` is True, else 2-D tensor of floats which are per-bitstring
        expectation values.
    """
    pulled_back_circuits = self.pulled_back_circuits(circuits, resolve=False)
    return self._expectation_function(
        pulled_back_circuits, counts, operators, reduce=reduce)

  def pqc_unitary(self):
    if self.is_analytic:
      return self._unitary_layer(self.pqc()).to_tensor()[0]
    raise NotImplementedError()

  def __add__(self, other):
    """Returns QNN which is the pqc of `other` appended to the pqc of `self`.

    Note: the backend, differentiator, and is_analytic attributes may be
    different between self and other. The resulting self + other uses such
    attributes from the QNN with the larger number of qubits.
    """
    if len(self.raw_qubits) >= len(other.raw_qubits):
      copy_qnn = self
    else:
      copy_qnn = other
    new_pqc = tfq.append_circuit(self.pqc(False), other.pqc(False))
    new_symbols = tf.concat([self.symbols, other.symbols], 0)
    new_values = tf.concat([self.values, other.values], 0)
    new_qnn = QNN(
        tfq.from_tensor(new_pqc)[0],
        symbols=new_symbols,
        values=new_values,
        backend=copy_qnn.backend,
        differentiator=copy_qnn.differentiator,
        is_analytic=copy_qnn.is_analytic,
        name=f"{self.name}_plus_{other.name}")
    return new_qnn

  def __pow__(self, exponent):
    """QNN raised to a power, only valid for exponent -1, the inverse."""
    if exponent != -1:
      raise ValueError("Only the inverse (exponent == -1) is supported.")
    new_qnn = self.copy()
    old_pqc = new_qnn.pqc(False)
    old_inverse_pqc = new_qnn.inverse_pqc(False)
    new_qnn._pqc = old_inverse_pqc
    new_qnn._inverse_pqc = old_pqc
    return new_qnn
