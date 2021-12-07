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
"""Tools for defining quantum circuit models."""

import abc

import cirq
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq


class QuantumCircuit(tf.keras.Model, abc.ABC):
  """Class for representing quantum circuits."""

  def __init__(self, name=None):
    super().__init__(name=name)

  @property
  @abc.abstractmethod
  def qubits(self):
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def symbols(self):
    """1D `tf.Tensor` of strings which are the parameters of circuit."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def values(self):
    """1D `tf.Tensor` of real numbers specifying the current values of the
       circuit parameters, such that `self.values[i]` is the current value of
       `self.symbols[i]` in `self.pqc` and `self.inverse_pqc`."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def pqc(self):
    """The symbolically parameterized circuit."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def inverse_pqc(self):
    """The symbolically parameterized inverse circuit."""
    raise NotImplementedError()

  @property
  def trainable_variables(self):
    return super().trainable_variables

  @trainable_variables.setter
  @abc.abstractmethod
  def trainable_variables(self, value):
    raise NotImplementedError()


class DirectQuantumCircuit(QuantumCircuit):
  """QuantumCircuit with direct map from model variables to circuit params."""

  def __init__(
      self,
      pqc,
      *,
      symbols=None,
      values=None,
      initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi),
      name=None,
  ):
    """Initializes a DirectQuantumCircuit.

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
      name: Identifier for this DirectQuantumCircuit.
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
    self._values = tf.Variable(
        initial_value=values, name=f"{self.name}_pqc_values")

    self._pqc = tfq.convert_to_tensor([pqc])
    self._inverse_pqc = tfq.convert_to_tensor([pqc**-1])

    self._qubits = sorted(pqc.all_qubits())

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
  def pqc(self):
    return self._pqc

  @property
  def inverse_pqc(self):
    return self._inverse_pqc

  @property
  def trainable_variables(self):
    return [self.values]

  @trainable_variables.setter
  def trainable_variables(self, value):
    self._values = value[0]

  def copy(self):
    new_qnn = DirectQuantumCircuit(tfq.from_tensor(self.pqc)[0], name=self.name)
    new_qnn._values.assign(self.values)
    return new_qnn

  def __add__(self, other):
    """DirectQuantumCircuit which is pqc of `other` appended to pqc of `self`"""
    new_pqc = tfq.append_circuit(self.pqc, other.pqc)
    new_symbols = tf.concat([self.symbols, other.symbols], 0)
    new_values = tf.concat([self.values, other.values], 0)
    new_qnn = DirectQuantumCircuit(
        tfq.from_tensor(new_pqc)[0],
        symbols=new_symbols,
        values=new_values,
        name=f"{self.name}_plus_{other.name}")
    return new_qnn

  def __pow__(self, exponent):
    """QNN raised to a power, only valid for exponent -1, the inverse."""
    if exponent != -1:
      raise ValueError("Only the inverse (exponent == -1) is supported.")
    new_qnn = self.copy()
    old_pqc = new_qnn.pqc
    old_inverse_pqc = new_qnn.inverse_pqc
    new_qnn._pqc = old_inverse_pqc
    new_qnn._inverse_pqc = old_pqc
    return new_qnn
