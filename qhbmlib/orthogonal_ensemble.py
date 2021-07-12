# Copyright 2021 The QHBM Library Authors.
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
"""Module for defining and sampling from orthogonal ensembles."""

import numbers
from typing import Any, Callable, Iterable, List, Union

import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq


def build_bit_circuit(qubits, ident):
  """Returns exponentiated X gate on each qubit and the exponent symbols."""
  circuit = cirq.Circuit()
  symbols = []
  for n, q in enumerate(qubits):
    new_bit = sympy.Symbol("_bit_{0}_{1}".format(ident, n))
    circuit += cirq.X(q)**new_bit
    symbols.append(new_bit)
  return circuit, symbols


def upgrade_initial_values(
    initial_values: Union[List[numbers.Real], tf.Tensor, tf.Variable]
) -> tf.Variable:
  """Upgrades the given values to a tf.Variable.

    Args:
      initial_values: Numeric values to upgrade.

    Returns:
      The input values upgraded to a fresh `tf.Variable` of dtype `tf.float32`.
    """
  if isinstance(initial_values, tf.Variable):
    initial_values = initial_values.read_value()
  if isinstance(initial_values, (List, tf.Tensor)):
    initial_values = tf.Variable(
        tf.cast(initial_values, tf.float32), dtype=tf.float32)
    if len(tf.shape(initial_values)) != 1:
      raise ValueError("Values for QHBMs must be 1D.")
    return initial_values
  raise TypeError(
      f"Input needs to be a numeric type, got {type(initial_values)}")


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


class OrthogonalEnsemble:
  """Operations on ensembles of orthogonal states indexed by bitstrings."""

  def __init__(self, circuit, symbols, symbols_initial_values, name):
    """Initialize an OrthogonalEnsemble."""
    self.name = name
    self.phis = upgrade_initial_values(symbols_initial_values)
    self.phis_symbols = upgrade_symbols(symbols, self.phis)
    self.u = upgrade_circuit(circuit, self.phis_symbols)
    self.u_dagger = upgrade_circuit(circuit**-1, self.phis_symbols)

    self.raw_qubits = sorted(circuit.all_qubits())
    self.qubits = tf.constant([[q.row, q.col] for q in self.raw_qubits])
    raw_bit_circuit, raw_bit_symbols = build_bit_circuit(self.raw_qubits, name)
    self.bit_symbols = upgrade_symbols(raw_bit_symbols,
                                       tf.ones([len(self.raw_qubits)]))
    self.bit_circuit = upgrade_circuit(raw_bit_circuit, self.bit_symbols)

  def copy(self):
    return OrthogonalEnsemble(
        tfq.from_tensor(self.u)[0],
        [sympy.Symbol(s.decode("utf-8")) for s in self.phis_symbols.numpy()],
        self.phis,
        self.name,
    )

  def ensemble(self, bitstrings):
    """Returns the current concrete circuits for this orthogonal ensemble.

      Args:
        bitstrings: 2D tensor of dtype `tf.int8` whose entries are bits. These
          specify the labels of the states to use in the ensemble.

      Returns:
        1D tensor of strings which represent the current ensemble circuits.
      """
    num_labels = tf.shape(bitstrings)[0]
    tiled_bit_injectors = tf.tile(self.bit_circuit, [num_labels])
    bit_circuits = tfq.resolve_parameters(tiled_bit_injectors, self.bit_symbols,
                                          tf.cast(bitstrings, tf.float32))
    u_concrete = tfq.resolve_parameters(self.u, self.phis_symbols,
                                        tf.expand_dims(self.phis, 0))
    tiled_u_concrete = tf.tile(u_concrete, [num_labels])
    return tfq.append_circuit(bit_circuits, tiled_u_concrete)

  def sample(self, bitstrings, counts):
    """Returns bitstring samples from the ensemble.

      Args:
        bitstrings: 2D tensor of dtype `tf.int8` whose entries are bits. These
          specify the labels of the states to use in the ensemble.
        counts: 1D tensor of dtype `tf.int32` such that `counts[i]` is the
          number of samples to draw from `self.u|bitstrings[i]>`.

      Returns:
        ragged_samples: `tf.RaggedTensor` of DType `tf.int8` structured such
          that `ragged_samples[i]` contains `counts[i]` bitstrings drawn from
          `self.u|bitstrings[i]>`.
    """
    raise NotImplementedError

  def measure(self, bitstrings, observables):
    """Returns the expectation values of the observables against the ensemble.

      Args:
        bitstrings: 2D tensor of dtype `tf.int8` whose entries are bits.
        observables: 2D tensor of strings, the result of calling
          `tfq.convert_to_tensor` on a list of lists of cirq.PauliSum which has
          effectively 1D structure, `[[op1, op2, ... ]]`.  Will be tiled along
          the 0th dimension to measure `<opj>_self.u|bitstrings[i]>` for each i.

      Returns:
        2-D tensor of floats which are the expectation values.
      """
    raise NotImplementedError

  def pulled_back_sample(self, circuits, counts):
    """Returns samples from the pulled back data distribution.

      The inputs represent the data density matrix. The inverse of `self.u`
      is appended to create the set of circuits representing the
      pulled back data density matrix. Then, the requested number of bitstrings
      are sampled from each circuit.

      Args:
        circuit_samples: 1-D `tf.Tensor` of type `tf.string` which contains
          circuits serialized by `tfq.convert_to_tensor`. These represent pure
          state samples from the data density matrix. Each entry should be
          unique.
        counts: 1-D `tf.Tensor` of type `tf.int32`, must be the same size as
          `circuit_samples`. Contains the number of samples to draw from each
          inputcircuit.

      Returns:
        ragged_samples: `tf.RaggedTensor` of DType `tf.int8` structured such
            that `ragged_samples[i]` contains `counts[i]` bitstrings.
      """
    raise NotImplementedError

  def pulled_back_measure(self, circuits, counts, observables):
    """Returns the expectation values for a given pulled-back dataset."""
    raise NotImplementedError
