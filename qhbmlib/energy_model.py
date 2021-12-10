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
"""Tools for modelling energy functions."""

import abc
import itertools
from typing import List, Union

import cirq
import tensorflow as tf


def check_bits(bits):
  """Confirms the input is a valid bit index list."""
  if not isinstance(bits, list) or not all(isinstance(i, int) for i in bits):
    raise TypeError("`bits` must be a list of integers.")
  if len(set(bits)) != len(bits):
    raise ValueError("All entries of `bits` must be unique.")
  return bits


def check_order(order):
  """Confirms the input is a valid parity order."""
  if not isinstance(order, int):
    raise TypeError("`order` must be an integer.")
  if order <= 0:
    raise ValueError("`order` must be greater than zero.")
  return order


def check_layers(layer_list):
  """Confirms the input is a valid list of keras Layers."""
  if not isinstance(layer_list, list) or not all(
      [isinstance(e, tf.keras.layers.Layer) for e in layer_list]):
    raise TypeError("must be a list of keras layers.")
  return layer_list


class Squeeze(tf.keras.layers.Layer):
  """Wraps tf.squeeze in a Keras Layer."""

  def __init__(self, axis=None):
    """Initializes a Squeeze layer.

    Args:
      axis: An optional list of ints. Defaults to []. If specified, only
        squeezes the dimensions listed. The dimension index starts at 0. It is
        an error to squeeze a dimension that is not 1. Must be in the range
        [-rank(input), rank(input)). Must be specified if input is
        a RaggedTensor.
    """
    super().__init__()
    if axis is None:
      axis = []
    self._axis = axis

  def call(self, inputs):
    """Applies tf.squeeze to the inputs."""
    return tf.squeeze(inputs, axis=self._axis)


class SpinsFromBitstrings(tf.keras.layers.Layer):
  """Simple layer taking bits to spins."""

  def __init__(self):
    """Initializes a SpinsFromBitstrings."""
    super().__init__(trainable=False)

  def call(self, inputs):
    """Returns the spins corresponding to the input bitstrings.

    Note that this maps |0> -> +1 and |1> -> -1.  This is in accordance with
    the usual interpretation of the Bloch sphere.
    """
    return tf.cast(1 - 2 * inputs, tf.float32)


class VariableDot(tf.keras.layers.Layer):
  """Utility layer for dotting input with a same-sized variable."""

  def __init__(self,
               input_width,
               initializer=tf.keras.initializers.RandomUniform()):
    """Initializes a VariableDot layer.

    Args:
      input_width: Size of the last dimension of input on which this layer acts.
      initializer: A `tf.keras.initializers.Initializer` which specifies how to
        initialize the values of the parameters.
    """
    super().__init__()
    self.kernel = self.add_weight(
        name=f'kernel',
        shape=[input_width],
        initializer=initializer,
        trainable=True)

  def call(self, inputs):
    """Returns the dot product between the inputs and this layer's variables."""
    return tf.reduce_sum(inputs * self.kernel, -1)


class Parity(tf.keras.layers.Layer):
  """Computes the parities of input spins."""

  def __init__(self, bits, order):
    """Initializes a Parity layer."""
    super().__init__(trainable=False)
    bits = check_bits(bits)
    order = check_order(order)
    indices_list = []
    for i in range(1, order + 1):
      combos = itertools.combinations(range(len(bits)), i)
      indices_list.extend(list(combos))
    self.indices = tf.ragged.stack(indices_list)
    self.num_terms = len(indices_list)

  def call(self, inputs):
    """Returns a batch of parities corresponding to the input bitstrings."""
    parities_t = tf.zeros([self.num_terms, tf.shape(inputs)[0]])
    for i in tf.range(self.num_terms):
      parity = tf.reduce_prod(tf.gather(inputs, self.indices[i], axis=-1), -1)
      parities_t = tf.tensor_scatter_nd_update(parities_t, [[i]], [parity])
    return tf.transpose(parities_t)


class BitstringEnergy(tf.keras.layers.Layer):
  """Class for representing an energy function over bitstrings.

  Keras Layer which can be interpreted as outputting an unnormalized
  log-probability for each given bit-string x, written E(x).  Hence, this class
  implicitly defines a probability distribution over all bit-strings given by
  $$p(x) = \frac{\exp(-E(x))}{\sum_x \exp(-E(x))}.$$

  Moving to its use in QHBMs: each bit-string can also be interpreted as
  an index for an entry of the diagonal eigenvalue matrix in the spectral
  representation of a density operator.  Hence, for a QHBM, inference
  corresponds to sampling computational basis states $|x><x|$ for $x ~ p$,
  where $p$ is the probability distribution written above.
  """

  def __init__(self,
               bits: List[int],
               energy_layers: List[tf.keras.layers.Layer],
               name: Union[None, str] = None):
    """Initializes a BitstringEnergy.

    Args:
      bits: Labels for the bits on which this distribution is supported.
      energy_layers: Concatenation of these layers yields trainable map from
        bitstrings to scalars.
      name: Optional name for the model.
    """
    super().__init__(name=name)
    self._bits = check_bits(bits)
    self._energy_layers = check_layers(energy_layers)

  @property
  def num_bits(self):
    """Number of bits on which this layer acts."""
    return len(self.bits)

  @property
  def bits(self):
    """Labels for the bits on which this distribution is supported."""
    return self._bits

  @property
  def energy_layers(self):
    """List of keras layers which, when stacked, map bitstrings to energies."""
    return self._energy_layers

  def call(self, inputs):
    """Returns the energies corresponding to the input bitstrings."""
    x = inputs
    for layer in self._energy_layers:
      x = layer(x)
    return x


class PauliBitstringEnergy(BitstringEnergy, abc.ABC):
  """Augments BitstringEnergy with a Pauli Z representation."""

  def __init__(self,
               bits,
               pre_process,
               post_process,
               name=None):
    """Initializes a PauliBitstringEnergy.

    Args:
      bits: Labels for the bits on which this distribution is supported.
      pre_process: List of keras layers. Concatenation of these layers yields
        the trainable map from bitstrings to the intermediate representation.
      post_process: List of keras layers.  Concatenation of these layers yields
        the trainable map from the intermediate representation to scalars.
      name: Optional name for the model.
    """
    self._pre_process = check_layers(pre_process)
    self._post_process = check_layers(post_process)
    super().__init__(bits, self._pre_process + self._post_process, name)

  @abc.abstractmethod
  def operator_shards(self, qubits):
    """Parameter independent Pauli Z strings to measure.

    Args:
      qubits: List of cirq.GridQubits. objects to measure.
    
    Returns:
      List of PauliSum objects whose expectation values are fed to
        `operator_expectation` to compute average energy.
    """
    raise NotImplementedError()

  def operator_expectation(self, expectation_shards):
    """Computes the average energy given operator shard expectation values."""
    x = expectation_shards
    for layer in self._post_process:
      x = layer(x)
    return x


class BernoulliEnergy(PauliBitstringEnergy):
  """Tensor product of coin flip distributions."""

  def __init__(self,
               bits: List[int],
               initializer=tf.keras.initializers.RandomUniform(),
               name=None):
    """Initializes a BernoulliEnergy.
    
    Args:
      bits: Each entry is an index on which the distribution is supported.
      initializer: A `tf.keras.initializers.Initializer` which specifies how to
        initialize the values of the parameters.
      name: Optional name for the model.
    """
    pre_process = [SpinsFromBitstrings()]
    post_process = [VariableDot(len(bits), initializer=initializer)]
    super().__init__(bits, pre_process, post_process, name)

  def operator_shards(self, qubits):
    """See base class description."""
    return [cirq.PauliSum.from_pauli_strings(cirq.Z(q)) for q in qubits]


class KOBE(PauliBitstringEnergy):
  """Kth Order Binary Energy function."""

  def __init__(self,
               bits: List[int],
               order: int,
               initializer=tf.keras.initializers.RandomUniform(),
               name=None):
    """Initializes a KOBE.

    Args:
      bits: Each entry is an index on which the distribution is supported.
      order: The order of the KOBE.
      initializer: A `tf.keras.initializers.Initializer` which specifies how to
        initialize the values of the parameters.
      name: Optional name for the model.
    """
    parity_layer = Parity(bits, order)
    self._num_terms = parity_layer.num_terms
    self._indices = parity_layer.indices
    pre_process = [SpinsFromBitstrings(), parity_layer]
    post_process = [
        VariableDot(parity_layer.num_terms, initializer=initializer)
    ]
    super().__init__(bits, pre_process, post_process, name)

  def operator_shards(self, qubits):
    """See base class description."""
    ops = []
    for i in range(self._num_terms):
      string_factors = []
      for loc in self._indices[i]:
        string_factors.append(cirq.Z(qubits[loc]))
        string = cirq.PauliString(string_factors)
      ops.append(cirq.PauliSum.from_pauli_strings(string))
    return ops

