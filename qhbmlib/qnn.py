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
"""Module for defining and sampling from orthogonal sets of QNNs."""

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


class QNN:
  """Operations on parameterized unitaries with bitstring inputs."""

  def __init__(self,
               circuit: cirq.Circuit,
               symbols: Union[Iterable[sympy.Symbol], tf.Tensor],
               symbols_initial_values: Union[List[numbers.Real], tf.Tensor,
                                             tf.Variable],
               name: str,
               backend='noiseless'):
    """Initialize a QNN.

    Args:
      circuit: Representation of a parameterized unitary.
      symbols: All parameters of `circuit`.
      symbols_initial_values: Real number for each entry of `symbols`, which are
        the initial values of the parameters in `circuit`.
      name: Identifier for this QNN.
      backend: Optional Python `object` that specifies what backend TFQ will use
        for operations involving this QNN. Options are {'noisy', 'noiseless'},
        or however users may also specify a preconfigured cirq execution
        object to use instead, which must inherit `cirq.Sampler`.
    """
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
    self._sample_layer = tfq.layers.Sample(backend=backend)
    if backend == 'noiseless' or backend is None:
      self._expectation_layer = tfq.layers.Expectation(backend=backend)
      self.analytic = tf.constant(True)
    else:
      self._expectation_layer = tfq.layers.SampledExpectation(backend=backend)
      self.analytic = tf.constant(False)

  def copy(self):
    return QNN(
        tfq.from_tensor(self.u)[0],
        [sympy.Symbol(s.decode("utf-8")) for s in self.phis_symbols.numpy()],
        self.phis,
        self.name,
    )

  def _sample_function(self, circuits, counts):
    """General function for sampling from circuits."""
    raw_samples = self._sample_layer(
        circuits,
        symbol_names=tf.constant([], dtype=tf.string),
        symbol_values=tf.tile(
            tf.constant([[]], dtype=tf.float32), [tf.shape(counts)[0], 1]),
        repetitions=tf.expand_dims(tf.math.reduce_max(counts), 0),
    )
    num_samples_mask = tf.cast((tf.ragged.range(counts) + 1).to_tensor(),
                               tf.bool)
    return tf.ragged.boolean_mask(raw_samples, num_samples_mask)

  def _sample_expectation_function(self, circuits, counts, observables):
    """General function for taking sampled expectations from circuits.

    `counts[i]` samples are drawn from `circuits[i]` and used to compute
    each expectation in `observables`.  Additionally, `counts[i]` sets the
    weight of `circuits[i]` in the expectation.
    """
    prob_terms = tf.cast(counts, tf.float32) / tf.cast(
        tf.reduce_sum(counts), tf.float32)
    num_circuits = tf.shape(counts)[0]
    tiled_observables = tf.tile(observables, [num_circuits, 1])
    bare_expectations = self._expectation_layer(
        circuits,
        symbol_names=tf.constant([], dtype=tf.string),
        symbol_values=tf.tile(
            tf.constant([[]], dtype=tf.float32), [num_circuits, 1]),
        operators=tiled_observables,
        repetitions=tf.expand_dims(counts, 1),
    )
    return tf.expand_dims(prob_terms, 1) * bare_expectations

  def _exact_expectation_function(self, circuits, counts, observables):
    """General function for taking sampled expectations from circuits.

    `counts[i]` sets the weight of `circuits[i]` in the expectation.
    """
    prob_terms = tf.cast(counts, tf.float32) / tf.cast(
        tf.reduce_sum(counts), tf.float32)
    num_circuits = tf.shape(counts)[0]
    tiled_observables = tf.tile(observables, [num_circuits, 1])
    bare_expectations = self._expectation_layer(
        circuits,
        symbol_names=tf.constant([], dtype=tf.string),
        symbol_values=tf.tile(
            tf.constant([[]], dtype=tf.float32), [num_circuits, 1]),
        operators=tiled_observables,
    )
    return tf.expand_dims(prob_terms, 1) * bare_expectations

  @property
  def resolved_u(self):
    """Returns the diagonalizing unitary with current phis resolved."""
    return tfq.resolve_parameters(self.u, self.phis_symbols,
                                  tf.expand_dims(self.phis, 0))

  @property
  def resolved_u_dagger(self):
    """Returns the diagonalizing adjoint unitary with current phis resolved."""
    return tfq.resolve_parameters(self.u_dagger, self.phis_symbols,
                                  tf.expand_dims(self.phis, 0))

  def circuits(self, bitstrings):
    """Returns the current concrete circuits for this QNN given bitstrings.

      Args:
        bitstrings: 2D tensor of dtype `tf.int8` whose entries are bits. These
          specify the state inputs to use in the returned set of circuits.

      Returns:
        1D tensor of strings which represent the current QNN circuits.
      """
    num_labels = tf.shape(bitstrings)[0]
    tiled_bit_injectors = tf.tile(self.bit_circuit, [num_labels])
    bit_circuits = tfq.resolve_parameters(tiled_bit_injectors, self.bit_symbols,
                                          tf.cast(bitstrings, tf.float32))
    tiled_u_concrete = tf.tile(self.resolved_u, [num_labels])
    return tfq.append_circuit(bit_circuits, tiled_u_concrete)

  def sample(self, bitstrings, counts):
    """Returns bitstring samples from the QNN.

      Args:
        bitstrings: 2D tensor of dtype `tf.int8` whose entries are bits. These
          specify the state inputs to the unitary of this QNN.
        counts: 1D tensor of dtype `tf.int32` such that `counts[i]` is the
          number of samples to draw from `self.u|bitstrings[i]>`.

      Returns:
        ragged_samples: `tf.RaggedTensor` of DType `tf.int8` structured such
          that `ragged_samples[i]` contains `counts[i]` bitstrings drawn from
          `self.u|bitstrings[i]>`.
    """
    current_circuits = self.circuits(bitstrings)
    return self._sample_function(current_circuits, counts)

  def measure(self, bitstrings, counts, observables):
    """Returns the expectation values of the observables against the QNN.

      Args:
        bitstrings: 2D tensor of dtype `tf.int8` whose entries are bits.
        counts: 1D tensor of dtype `tf.int32` such that `counts[i]` is the
          relative weight of `bitstrings[i]` when computing expectations.
        observables: 2D tensor of strings, the result of calling
          `tfq.convert_to_tensor` on a list of lists of cirq.PauliSum which has
          effectively 1D structure, `[[op1, op2, ... ]]`.  Will be tiled along
          the 0th dimension to measure `<opj>_self.u|bitstrings[i]>` for each i.

      Returns:
        2-D tensor of floats which are the expectation values.
      """
    current_circuits = self.circuits(bitstrings)
    return tf.cond(
        self.analytic,
        lambda: self._exact_expectation_function(circuits, counts, observables),
        lambda: self._sample_expectation_function(circuits, counts, observables
                                                 ))

  def pulled_back_circuits(self, circuit_samples):
    """Returns the pulled back circuits for this QNN given input quantum data.

      Args:
        circuit_samples: 1-D `tf.Tensor` of type `tf.string` which contains
          circuits serialized by `tfq.convert_to_tensor`. These represent pure
          state samples from the data density matrix.

      Returns:
        1D tensor of strings which represent the pulled back circuits.
      """
    num_samples = tf.shape(circuit_samples)[0]
    tiled_u_dagger_concrete = tf.tile(self.resolved_u_dagger, [num_samples])
    return tfq.append_circuit(circuit_samples, tiled_u_dagger_concrete)

  def pulled_back_sample(self, circuit_samples, counts):
    """Returns samples from the pulled back data distribution.

      The inputs represent the data density matrix. The inverse of `self.u`
      is appended to create the set of circuits representing the pulled back
      data density matrix. Then, the requested number of bitstrings are sampled
      from each circuit.

      Args:
        circuit_samples: 1-D `tf.Tensor` of type `tf.string` which contains
          circuits serialized by `tfq.convert_to_tensor`. These represent pure
          state samples from the data density matrix.
        counts: 1-D `tf.Tensor` of type `tf.int32`, must be the same size as
          `circuit_samples`. Contains the number of samples to draw from each
          input circuit.

      Returns:
        ragged_samples: `tf.RaggedTensor` of DType `tf.int8` structured such
            that `ragged_samples[i]` contains `counts[i]` bitstrings.
      """
    current_circuits = self.pulled_back_circuits(circuit_samples)
    return self._sample_function(current_circuits, counts)

  def pulled_back_measure(self, circuit_samples, counts, observables):
    """Returns the expectation values for a given pulled-back dataset.

      Args:
        circuit_samples: 1-D `tf.Tensor` of type `tf.string` which contains
          circuits serialized by `tfq.convert_to_tensor`. These represent pure
          state samples from the data density matrix.
        counts: 1D tensor of dtype `tf.int32` such that `counts[i]` is the
          relative weight of `circuit_samples[i]` when computing expectations.
        observables: 2D tensor of strings, the result of calling
          `tfq.convert_to_tensor` on a list of lists of cirq.PauliSum which has
          effectively 1D structure, `[[op1, op2, ... ]]`.  Will be tiled along
          the 0th dimension to measure `<opj>_self.u_dagger|circuit_samples[i]>`
          for each i.

      Returns:
        2-D tensor of floats which are the expectation values.
    """
    current_circuits = self.pulled_back_circuits(circuit_samples)
    return tf.cond(
        self.analytic, lambda: self._exact_expectation_function(
            current_circuits, counts, observables), lambda: self.
        _sample_expectation_function(current_circuits, counts, observables))
