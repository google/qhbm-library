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
"""Utilities for metrics on BitstringEnergy."""

import itertools

from qhbmlib import energy_model


def probabilities(energy: energy_model.BitstringEnergy):
  """Returns the probabilities of the EBM.

  Args:
    energy: The energy function defining the EBM.
  """
  all_bitstrings = tf.constant(
      list(itertools.product([0, 1], repeat=energy.num_bits)), dtype=tf.int8)
  all_energies = energy(all_bitstrings)
  energy_exp = tf.math.exp(-all_energies)
  partition = tf.math.reduce_sum(energy_exp)
  return energy_exp / partition
