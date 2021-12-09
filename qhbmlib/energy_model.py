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


def check_layer_list(layer_list):
  """Confirms the input is a valid list of keras Layers."""
  if not isinstance(layer_list, list) or not all([isinstance(e, tf.keras.layers.Layer) for e in layer_list]):
    raise TypeError("must be a list of keras layers.")
  return layer_list


class BitstringEnergy(tf.keras.layers.Layer):
  """Class for representing an energy function over bitstrings."""

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
    self.energy_layers = check_layer_list(energy_layers)

  def get_config(self):
    config = super().get_config()
    config.update({
        "bits": copy.deepcopy(self._bits),
        "energy_layers": copy.deepcopy(self.energy_layers),
    })
    return config

  @classmethod
  def from_config(cls, config):
    return cls(config["bits"], config["energy_layers"], config["name"])

  def copy(self):
    return BitstringEnergy.from_config(self.get_config())

  @property
  def num_bits(self):
    return len(self.bits)

  @property
  def bits(self):
    return self._bits

  def call(self, inputs):
    x = inputs
    for layer in self.energy_layers:
      x = layer(x)
    return tf.squeeze(x, -1)


class PauliBitstringEnergy(BitstringEnergy):
  """Augments BitstringEnergy with a Pauli Z representation."""

  def __init__(self, bits, pre_process, post_process, operator_shards_func, name=None):
    """Initializes a PauliBitstringEnergy.

    Args:
      bits: Labels for the bits on which this distribution is supported.
      pre_process: List of keras layers. Concatenation of these layers yields
        the trainable map from bitstrings to the intermediate representation.
      post_process: List of keras layers.  Concatenation of these layers yields
        the trainable map from the intermediate representation to scalars.
      operator_shards_func: Callable which, given a list of qubits, returns
        PauliSum objects whose expectation values are the intermediate
        representation to feed to post_process..
      name: Optional name for the model.
    """
    self._pre_process = check_layer_list(pre_process)
    self._post_process = check_layer_list(post_process)
    super().__init__(bits, self._pre_process + self._post_process, name)
    self._operator_shards_func = operator_shards_func

  def operator_shards(self, qubits):
    """Parameter independent Pauli Z strings to measure."""
    raise self._operator_shards_func(qubits)

  def operator_expectation(self, expectation_shards):
    """Computes the average energy given operator shard expectation values."""
    x = expectation_shards
    for layer in self._post_process:
      x = layer(x)
    return x


class SpinsFromBitstrings(tf.keras.layers.Layer):
  """Simple layer taking bits to spins."""

  def __init__(self):
    """Initializes a SpinsFromBitstrings."""
    super().__init__(trainable=False)
  
  def call(inputs):
    tf.cast(1 - 2 * inputs, tf.float32)


class VariableDot(tf.keras.layers.Layer):
  """Utility layer for dotting input with a same-sized variable."""

  def __init__(self,
               input_width,
               initializer=initializer=tf.keras.initializers.RandomUniform()):
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
    self._dot = tf.keras.layers.Dot()

  def call(self, inputs):
    input_shape = tf.shape(inputs)
    tiled = tf.tile(self.kernel, [input_shape[0], 1])
    return self._dot([inputs, tiled])
  

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
    def operator_shards_func(qubits):
      return [
        cirq.PauliSum.from_pauli_strings(cirq.Z(q))
        for q in qubits
      ]
    super().__init__(bits, pre_process, post_process, operator_shards_func, name)


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
    self.indices = tf.ragged_stack(indices_list)
    self.num_terms = len(indices_list)

  def call(self, inputs):
    parities_t = tf.zeros(
        [self.num_terms, tf.shape(inputs)[0]], dtype=tf.float32)
    for i in tf.range(self.num_terms):
      parity = tf.reduce_prod(tf.gather(inputs, self.indices[i], axis=-1), -1)
      parities_t = tf.tensor_scatter_nd_update(parities_t, [[i]], [parity])
    return parities_t


class KOBE(PauliBitstringFunction):
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
    pre_process = [SpinsFromBitstrings(), parity_layer]
    post_process = [VariableDot(tf.shape(parity_layer.indices)[0], initializer=initializer)]

    def operator_shards_func(qubits):
      ops = []
      for i in range(parity_layer.num_terms):
        string_factors = []
        for loc in parity_layer.indices[i]:
          string_factors.append(cirq.Z(qubits[loc]))
        string = cirq.PauliString(string_factors)
        ops.append(cirq.PauliSum.from_pauli_strings(string))
      return ops

    super().__init__(bits, pre_process, post_process, operator_shards_func, name)



