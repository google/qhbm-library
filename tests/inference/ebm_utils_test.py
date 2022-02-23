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
"""Tests for infer.ebm_utils"""

import random

import tensorflow as tf

from qhbmlib.inference import ebm
from qhbmlib.inference import ebm_utils
from qhbmlib.models import energy
from qhbmlib import utils


class ProbabilitiesTest(tf.test.TestCase):
  """Tests the probabilities function."""

  def test_probabilities(self):
    """Confirms probabilities are correct for an MLP."""
    num_bits = 5
    num_layers = 3
    units = random.sample(range(1, 100), num_layers)
    activations = random.sample([
        "elu", "exponential", "gelu", "hard_sigmoid", "linear", "relu", "selu",
        "sigmoid", "softmax", "softplus", "softsign", "swish", "tanh"
    ], num_layers)
    expected_layer_list = []
    for i in range(num_layers):
      expected_layer_list.append(
          tf.keras.layers.Dense(units[i], activation=activations[i]))
    expected_layer_list.append(tf.keras.layers.Dense(1))
    expected_layer_list.append(utils.Squeeze(-1))
    actual_energy = energy.BitstringEnergy(
        list(range(num_bits)), expected_layer_list)

    num_expectation_samples = 1  # Required but unused
    infer = ebm.AnalyticEnergyInference(actual_energy, num_expectation_samples)
    expected_probabilities = infer.distribution.probs_parameter()

    probabilities_wrapped = tf.function(ebm_utils.probabilities)
    actual_probabilities = probabilities_wrapped(actual_energy)
    self.assertAllClose(actual_probabilities, expected_probabilities)


if __name__ == "__main__":
  print("Running ebm_utils_test.py ...")
  tf.test.main()