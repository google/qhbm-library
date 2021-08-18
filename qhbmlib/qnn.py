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


def build_bit_circuit(qubits, name='bit_circuit'):
  """Returns exponentiated X gate on each qubit and the exponent symbols."""
  circuit = cirq.Circuit()
  symbols = []
  for n, q in enumerate(qubits):
    new_bit = sympy.Symbol("{0}_bit_{1}".format(name, n))
    circuit += cirq.X(q)**new_bit
    symbols.append(new_bit)
  return circuit, symbols


def upgrade_symbols(
    symbols: Union[Iterable[sympy.Symbol], tf.Tensor],
    values: Union[tf.Tensor, tf.Variable],
) -> tf.Tensor:
  """Upgrades symbols and checks for correct shape.

    For a circuit compatible with TFQ, there must be a value associated with
    each symbol. This function ensures `values` is the same shape as `symbols`.

    Args:
      symbols: Iterable of `sympy.Symbol`s to upgrade.
      values: Values corresponding to the symbols.

    Returns:
      `tf.Tensor` containing the string representations of the input `symbols`.
    """
  if isinstance(symbols, Iterable):
    if not all([isinstance(s, sympy.Symbol) for s in symbols]):
      raise TypeError("Each entry of `symbols` must be `sympy.Symbol`.")
    symbols_partial_upgrade = [str(s) for s in symbols]
    if len(set(symbols_partial_upgrade)) != len(symbols):
      raise ValueError("All entries of `symbols` must be unique.")
    symbols_upgrade = tf.constant(symbols_partial_upgrade, dtype=tf.string)
    if tf.shape(symbols_upgrade) != tf.shape(values):
      raise ValueError("There must be a symbol for every value.")
    return symbols_upgrade
  raise TypeError("`symbols` must be an iterable of `sympy.Symbol`s.")


def upgrade_circuit(circuit: cirq.Circuit, symbols: tf.Tensor) -> tf.Tensor:
  """Upgrades a circuit and confirms all symbols are present.

    Args:
      circuit: Circuit to convert to tensor.
      symbols: Tensor of strings which are the symbols in `circuit`.

    Returns:
      Single entry 1D tensor of strings representing the input `circuit`.
    """
  if not isinstance(circuit, cirq.Circuit):
    raise TypeError(f"`circuit` must be a `cirq.Circuit`, got {type(circuit)}")
  if not isinstance(symbols, tf.Tensor):
    raise TypeError("`symbols` must be a `tf.Tensor`")
  if symbols.dtype != tf.string:
    raise TypeError("`symbols` must have dtype `tf.string`")
  if set(tfq.util.get_circuit_symbols(circuit)) != {
      s.decode("utf-8") for s in symbols.numpy()
  }:
    raise ValueError(
        "`circuit` must contain all and only the parameters in `symbols`.")
  if not circuit:
    raise ValueError("Empty circuit not allowed. "
                     "Instead, use identities on all unused qubits.")
  return tfq.convert_to_tensor([circuit])


class QNN(tf.keras.Model):
  """Operations on parameterized unitaries with bitstring inputs."""

  def __init__(
      self,
      pqc,
      symbols,
      initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi),
      backend='noiseless',
      differentiator=None,
      analytic=False,
      name=None,
  ):
    """Initialize a QNN.

    Args:
      pqc: Representation of a parameterized quantum circuit.
      symbols: All parameters of `pqc`.
      initializer: A 'tf.keras.initializers.Initializer' which specifies how to
        initialize the values of the parameters in `circuit`.
      backend: Optional Python `object` that specifies what backend TFQ will use
        for operations involving this QNN. Options are {'noisy', 'noiseless'},
        or however users may also specify a preconfigured cirq execution
        object to use instead, which must inherit `cirq.Sampler`.
      differentiator: Either None or a `tfq.differentiators.Differentiator`,
        which specifies how to take the derivative of a quantum circuit.
      analytic: bool flag that enables analytic methods. If True, then backend
        must also be 'noiseless'.
      name: Identifier for this QNN.
    """
    super().__init__(name=name)

    self._values = self.add_weight(
        name=f'{self.name}_pqc_values',
        shape=[len(symbols)],
        initializer=initializer)
    self._symbols = upgrade_symbols(symbols, self._values)
    self._pqc = upgrade_circuit(pqc, self.symbols)
    self._inverse_pqc = upgrade_circuit(pqc**-1, self.symbols)

    self._raw_qubits = sorted(pqc.all_qubits())
    self._qubits = tf.constant([[q.row, q.col] for q in self._raw_qubits])
    bit_circuit, bit_symbols = build_bit_circuit(self._raw_qubits)
    self._bit_symbols = upgrade_symbols(bit_symbols,
                                        tf.ones([len(self._raw_qubits)]))
    self._bit_circuit = upgrade_circuit(bit_circuit, self._bit_symbols)

    self._differentiator = differentiator
    self._sample_layer = tfq.layers.Sample(backend=backend)
    if backend == 'noiseless' or backend is None:
      self._backend = 'noiseless'
      self._analytic = analytic
      self._expectation_layer = tfq.layers.Expectation(
          backend=backend, differentiator=differentiator)
    else:
      self._backend = backend
      self._analytic = False
      self._expectation_layer = tfq.layers.SampledExpectation(
          backend=backend, differentiator=differentiator)

    if self.analytic:
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
  def values(self):
    return self._values

  @property
  def backend(self):
    return self._backend

  @property
  def differentiator(self):
    return self._differentiator

  @property
  def analytic(self):
    return self._analytic

  def copy(self):
    qnn = QNN(
        tfq.from_tensor(self.pqc(resolve=False))[0],
        [sympy.Symbol(s.decode("utf-8")) for s in self.symbols.numpy()],
        backend=self.backend,
        differentiator=self.differentiator,
        analytic=self.analytic,
        name=self.name)
    qnn._values.assign(self._values)
    return qnn

  @tf.function
  def _sample_function(self, circuits, counts, mask=True):
    """General function for sampling from circuits."""
    samples = self._sample_layer(
        circuits, repetitions=tf.expand_dims(tf.math.reduce_max(counts), 0))
    if mask:
      num_samples_mask = tf.cast((tf.ragged.range(counts) + 1).to_tensor(),
                                 tf.bool)
      return tf.ragged.boolean_mask(samples, num_samples_mask)
    return samples

  def _expectation_function(self, circuits, counts, operators, reduce=True):
    """General function for taking sampled expectations from circuits.

    `counts[i]` sets the weight of `circuits[i]` in the expectation.
    Additionally, if `self.analytic` is false, `counts[i]` samples are drawn
    from `circuits[i]` and used to compute each expectation in `operators`.
    """
    num_circuits = tf.shape(circuits)[0]
    num_operators = tf.shape(operators)[0]
    tiled_values = tf.tile(tf.expand_dims(self.values, 0), [num_circuits, 1])
    tiled_operators = tf.tile(tf.expand_dims(operators, 0), [num_circuits, 1])
    if self.backend == 'noiseless':
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

  @tf.function
  def pqc(self, resolve=True):
    if resolve:
      return tfq.resolve_parameters(self._pqc, self.symbols,
                                    tf.expand_dims(self._values, 0))
    return self._pqc

  @tf.function
  def inverse_pqc(self, resolve=True):
    if resolve:
      return tfq.resolve_parameters(self._inverse_pqc, self.symbols,
                                    tf.expand_dims(self._values, 0))
    return self._inverse_pqc

  @tf.function
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

  @tf.function
  def sample(self, bitstrings, counts, mask=True):
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
    return self._sample_function(circuits, counts, mask=mask)

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

  @tf.function
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

  @tf.function
  def pulled_back_sample(self, circuits, counts, mask=True):
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
    return self._sample_function(pulled_back_circuits, counts, mask=mask)

  @tf.function
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

  @tf.function
  def pqc_unitary(self):
    if self.analytic:
      return self._unitary_layer(self.pqc()).to_tensor()[0]
    raise NotImplementedError()
