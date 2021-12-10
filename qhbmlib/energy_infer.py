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

from qhbmlib import energy_model


class AnalyticInferenceLayer(tf.keras.layers.Layer):
  """Sampler which calculates all probabilities and samples as categorical."""

  def __init__(self, energy: energy_model.BitstringEnergy, name=None):
    """Instantiates an AnalyticInferenceLayer object.

    Internally, this class saves all possible bitstrings as a tensor,
    whose energies are calculated relative to input distributions for sampling.
    """
    super().__init__(name=name)
    self._energy = energy
    self._sampler = tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: tfd.Categorical(logits=-1 * t))

  def call(self, inputs):
    x = tf.squeeze(self._bit_string_energy(inputs))
    return self._partition_estimator(x)


# Other inference objects will perform selective sampling of bit-strings, still
# working via forward passes.
def analytic_infer(analytic_inference_layer):
  """ Intuitively, compares to tfp.mcmc.sample_chain.
  However, in this case, we are doing exact inference and so can return a
  tensor-coercible distribution object directly which is stronger than sampling
  access. So, here, inference corresponds to a single forward pass.
  """

  all_bitstrings = tf.constant(
      list(itertools.product([0, 1], repeat=num_bits)), dtype=tf.int8)
  return analytic_inference_layer(all_bitstrings)
