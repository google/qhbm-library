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
"""Tools for modeling energy functions."""

import abc
from typing import List, Union

import cirq
import tensorflow as tf

from qhbmlib import energy_model_utils


class BitstringEnergy(tf.keras.layers.Layer):
  r"""Class for representing an energy function over bitstrings.

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
      bits: Unique labels for the bits on which this distribution is supported.
      energy_layers: Concatenation of these layers yields trainable map from
        bitstrings to scalars.
      name: Optional name for the model.
    """
    super().__init__(name=name)
    self._bits = energy_model_utils.check_bits(bits)
    self._energy_layers = energy_layers

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
    """List of keras layers which, when stacked, map bitstrings to energies.

    This list of layers is where the caller would access model weights to be
    updated from a secondary model or hypernetwork.
    """
    return self._energy_layers

  def build(self, input_shape):
    """Builds all the internal layers."""
    x = input_shape
    for layer in self._energy_layers:
      x = layer.compute_output_shape(x)

  def call(self, inputs):
    """Returns the energies corresponding to the input bitstrings."""
    x = inputs
    for layer in self._energy_layers:
      x = layer(x)
    return x


class PauliMixin(abc.ABC):
  """Mixin class to add a Pauli Z representation to  BitstringEnergy."""

  @property
  @abc.abstractmethod
  def post_process(self):
    """List of keras layers.  Concatenation of these layers yields
    the trainable map from the operator shard expectations to a single scalar
    which is the average energy.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def operator_shards(self, qubits: List[cirq.GridQubit]):
    """Parameter independent Pauli Z strings to measure.

    Args:
      qubits: List of cirq.GridQubits. objects to measure.

    Returns:
      List of PauliSum objects whose expectation values are fed to
        `operator_expectation` to compute average energy.
    """
    raise NotImplementedError()

  def operator_expectation(self, expectation_shards: tf.Tensor):
    """Computes the average energy given operator shard expectation values."""
    x = expectation_shards
    for layer in self.post_process:
      x = layer(x)
    return x


class BernoulliEnergy(BitstringEnergy, PauliMixin):
  """Tensor product of coin flip distributions.

  Note that we parameterize using the energy of a spin in a magnetic field,
  which is offset from the log probability of a typical Bernoulli.
  """

  def __init__(self,
               bits: List[int],
               initializer: tf.keras.initializers.Initializer = tf.keras
               .initializers.RandomUniform(),
               name: Union[None, str] = None):
    """Initializes a BernoulliEnergy.

    Args:
      bits: Unique labels for the bits on which this distribution is supported.
      initializer: A `tf.keras.initializers.Initializer` which specifies how to
        initialize the values of the parameters.
      name: Optional name for the model.
    """
    pre_process = [energy_model_utils.SpinsFromBitstrings()]
    post_process = [energy_model_utils.VariableDot(initializer=initializer)]
    super().__init__(bits, pre_process + post_process, name)
    self._post_process = post_process

  @property
  def logits(self):
    r"""Returns the current logits of the distribution.

    For our Bernoulli distribution, let $p$ be the probability of bit being `1`.
    In this case, we have $p = \frac{e^{theta}}{{e^{theta}+e^{-theta}}}$.
    Therefore, each independent logit is:
    $$logit = \log\frac{p}{1-p} = \log\frac{e^{theta}}{e^{-theta}}
            = \log{e^{2*theta}} = 2*theta$$
    """
    return 2 * self.post_process[0].kernel

  @property
  def post_process(self):
    """See base class description."""
    return self._post_process

  def operator_shards(self, qubits):
    """See base class description."""
    return [cirq.PauliSum.from_pauli_strings(cirq.Z(q)) for q in qubits]


class KOBE(BitstringEnergy, PauliMixin):
  """Kth Order Binary Energy function."""

  def __init__(self,
               bits: List[int],
               order: int,
               initializer: tf.keras.initializers.Initializer = tf.keras
               .initializers.RandomUniform(),
               name: Union[None, str] = None):
    """Initializes a KOBE.

    Args:
      bits: Each entry is an index on which the distribution is supported.
      order: The order of the KOBE.
      initializer: Specifies how to initialize the values of the parameters.
      name: Optional name for the model.
    """
    parity_layer = energy_model_utils.Parity(bits, order)
    self._num_terms = parity_layer.num_terms
    self._indices = parity_layer.indices
    pre_process = [energy_model_utils.SpinsFromBitstrings(), parity_layer]
    post_process = [energy_model_utils.VariableDot(initializer=initializer)]
    super().__init__(bits, pre_process + post_process, name)
    self._post_process = post_process

  @property
  def post_process(self):
    """See base class description."""
    return self._post_process

  def operator_shards(self, qubits: List[cirq.GridQubit]):
    """See base class description."""
    ops = []
    for i in range(self._num_terms):
      string_factors = []
      for loc in self._indices[i]:
        string_factors.append(cirq.Z(qubits[loc]))
        string = cirq.PauliString(string_factors)
      ops.append(cirq.PauliSum.from_pauli_strings(string))
    return ops
