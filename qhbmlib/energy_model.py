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
    if not isinstance(bits, list) or not all(isinstance(i, int) for i in bits):
      raise TypeError("`bits` must be a list of integers.")
    if len(set(bits)) != len(bits):
      raise ValueError("All entries of `bits` must be unique.")
    self._bits = bits
    if not isinstance(energy_layers, list) or not all([isinstance(e, tf.keras.layers.Layer) for e in energy_layers]):
      raise TypeError("`energy_layers` must be a list of keras layers.")
    self.energy_layers = energy_layers

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


# def get_mlp_bitstring_energy(
#     bits: List[int],
#     units: List[int],
#     activations: List[Union[None, str]],
#     kernel_initializer=tf.keras.initializers.RandomUniform(),
#     bias_initializer=tf.keras.initializers.Zeros(),
#     name=None):
#   self.hidden_layers = [
#     tf.keras.layers.Dense(
#       u,
#       activation=a,
#       kernel_initializer=kernel_initializer,
#       bias_initializer=bias_initializer)
#     for u, a in zip(units, activations)
#   ]
#     self._energy_layer = tf.keras.layers.Dense(
#         1,
#         kernel_initializer=kernel_initializer,
#         bias_initializer=bias_initializer)


  
#   return BitstringEnergy()
# class MLP(BitstringEnergy):
#   """Basic dense neural network energy based model."""

#   def __init__(self,
#                bits: List[int],
#                units: List[int],
#                activations: List[Union[None, str]],
#                kernel_initializer=tf.keras.initializers.RandomUniform(),
#                bias_initializer=tf.keras.initializers.Zeros(),
#                name=None):
#     """Initialize an MLP.

#     The arguments paramterize a stack of Dense layers.

#     Args:
#       bits: Labels for the bits on which this distribution is supported.
#       units: Positive integers which are the dimensions each layer's output.
#       activations: Activation functions to use at each layer. Entry of None
#         makes the corresponding layer have linear activation.
#       kernel_initializer: A `tf.keras.initializers.Initializer` which specifies how to
#         initialize the kernels of all layers.
#       bias_initializer: A `tf.keras.initializers.Initializer` which specifies how to
#         initialize the biases of all layers.
#     """
#     super().__init__(bits, name=name)
#     self._hidden_layers = [
#         tf.keras.layers.Dense(
#             u,
#             activation=a,
#             kernel_initializer=kernel_initializer,
#             bias_initializer=bias_initializer)
#         for u, a in zip(units, activations)
#     ]
#     self._energy_layer = tf.keras.layers.Dense(
#         1,
#         kernel_initializer=kernel_initializer,
#         bias_initializer=bias_initializer)
#     self.build([1, self.num_bits])

#   @property
#   def trainable_variables(self):
#     trainable_variables = []
#     for layer in self.layers:
#       trainable_variables.extend([layer.kernel, layer.bias])
#     return trainable_variables

#   @trainable_variables.setter
#   def trainable_variables(self, value):
#     i = 0
#     for layer in self.layers:
#       layer.kernel = value[i]
#       layer.bias = value[i + 1]
#       i += 2

#   def copy(self):
#     units = [layer.units for layer in self.layers[:-1]]
#     activations = [layer.activation for layer in self.layers[:-1]]
#     new_mlp = MLP(self.bits, units, activations, name=self.name)
#     for i in tf.range(len(new_mlp.trainable_variables)):
#       new_mlp.trainable_variables[i].assign(self.trainable_variables[i])
#     return new_mlp

#   def energy(self, bitstrings):
#     x = bitstrings
#     for hidden_layer in self._hidden_layers:
#       x = hidden_layer(x)
#     x = self._energy_layer(x)
#     return tf.squeeze(x, -1)


# class PauliBitstringEnergy(BitstringEnergy):
#   """Augments BitstringEnergy with a Pauli Z representation."""

#   @abc.abstractmethod
#   def operator_shards(self, qubits):
#     """Parameter independent Pauli Z strings to measure."""
#     raise NotImplementedError()

#   @abc.abstractmethod
#   def operator_expectation(self, expectation_shards):
#     """Computes the average energy given operator shard expectation values."""
#     raise NotImplementedError()


# class Bernoulli(PauliBitstringDistribution):
#   """Tensor product of coin flip distributions."""

#   def __init__(self,
#                bits: List[int],
#                initializer=tf.keras.initializers.RandomUniform(),
#                name=None):
#     """Initializes a Bernoulli distribution.

#     Args:
#       bits: Each entry is an index on which the distribution is supported.
#       initializer: A `tf.keras.initializers.Initializer` which specifies how to
#         initialize the values of the parameters.
#     """
#     super().__init__(bits, name=name)
#     self.kernel = self.add_weight(
#         name=f'kernel',
#         shape=[self.num_bits],
#         initializer=initializer,
#         trainable=True)

#   @property
#   def trainable_variables(self):
#     return [self.kernel]

#   @trainable_variables.setter
#   def trainable_variables(self, value):
#     self.kernel = value[0]

#   def copy(self):
#     bernoulli = Bernoulli(self.bits, name=self.name)
#     bernoulli.kernel.assign(self.kernel)
#     return bernoulli

#   def energy(self, bitstrings):
#     return tf.reduce_sum(
#         tf.cast(1 - 2 * bitstrings, tf.float32) * self.kernel, -1)

#   def operator_shards(self, qubits):
#     return [
#         cirq.PauliSum.from_pauli_strings(cirq.Z(qubits[i]))
#         for i in range(self.num_bits)
#     ]

#   def operator_expectation(self, expectation_shards):
#     return tf.reduce_sum(expectation_shards * self.kernel, -1)
