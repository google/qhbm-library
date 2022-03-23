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

import tensorflow as tf
import tensorflow_probability as tfp

from qhbmlib.models import energy
from qhbmlib import utils


def probabilities(input_energy: energy.BitstringEnergy):
  """Returns the probabilities of the EBM.

  Args:
    input_energy: The energy function defining the EBM.
  """
  all_bitstrings = tf.constant(
      list(itertools.product([0, 1], repeat=input_energy.num_bits)),
      dtype=tf.int8)
  all_energies = input_energy(all_bitstrings)
  energy_exp = tf.math.exp(-all_energies)
  partition = tf.math.reduce_sum(energy_exp)
  return energy_exp / partition


def relaxed_categorical_probabilities(category_samples, input_samples):
  """Returns the probabilities of samples relative to a relaxed categorical.

  Args:
    category_samples: 2D `tf.int8` tensor whose rows are bitstrings sampled from
      the target distribution approximated by the relaxed categorical.
    num_resamples: number of times to sample from the 

  Returns:
    The probability of each entry of `input_samples`.
  """
  unique_samples, _, counts = utils.unique_bitstrings_with_counts(samples)
  normalizing_constant = tf.math.reduce_sum(counts)
  categorical_probabilities = tf.cast(counts, tf.float32) / tf.cast(normalizing_constant, tf.float32)

  num_unique_samples = tf.shape(unique_samples)[0]
  # set so that if `num_unique_samples == 0`, then `categorical_weight == 0`
  numerator = tf.math.log(tf.cast(num_unique_samples + 1, tf.float32))
  num_bits = tf.shape(category_samples)[1]
  denominator = tf.math.reduce_logsumexp([num_bits * tf.math.log(2.0), 0.0])
  categorical_weight = numerator / denominator

  uniform_probability = tf.math.pow(2.0, -num_bits)

  def bitstring_prob(bitstring):
    """Returns the probability of a single bitstring."""
    tiled_bitstring = tf.tile(tf.expand_dims(bitstring, 0), [num_unique_samples, 1])
    bitstring_matches = tf.math.reduce_all(tf.math.equal(tiled_bitstring, unique_samples), 1)
    if tf.math.reduce_any(bitstring_matches):
      index = tf.where(bitstring_matches)[0][0]
      prob = categorical_weight * categorical_probabilities[index]
    else:
      prob = (1.0 - categorical_weight) * uniform_probability
    return prob

  return tf.map_fn(bitstring_prob, input_samples, fn_output_signature=tf.float32)


def relaxed_categorical_sampler(category_samples, num_resamples):
  """Returns samples from a relaxed categorical distribution.

  Args:
    category_samples: 2D `tf.int8` tensor whose rows are bitstrings sampled from
      the target distribution approximated by the relaxed categorical.
    num_resamples: Number of times to sample from the relaxed categorical.

  Returns:
    Samples from the relaxed categorical.
  """
  # Set up categorical over given samples
  unique_samples, _, counts = utils.unique_bitstrings_with_counts(samples)
  normalizing_constant = tf.math.reduce_sum(counts)
  categorical_probabilities = tf.cast(counts, tf.float32) / tf.cast(normalizing_constant, tf.float32)
  categorical = tfp.distributions.Categorical(probs=categorical_probabilities)

  # Choose how many times we sample from categorical versus uniform distribution
  num_unique_samples = tf.shape(unique_samples)[0]
  # set so that if `num_unique_samples == 0`, then `categorical_weight == 0`
  numerator = tf.math.log(tf.cast(num_unique_samples + 1, tf.float32))
  num_bits = tf.shape(category_samples)[1]
  denominator = tf.math.reduce_logsumexp([num_bits * tf.math.log(2.0), 0.0])
  categorical_weight = numerator / denominator
  binary_samples = tfp.distributions.Bernoulli(probs=[categorical_weight]).sample(num_resamples)
  _, _, resample_choices = utils.unique_bitstrings_with_counts(binary_samples)

  # Get samples
  raw_categorical_samples = categorical.sample(resample_choices[0])
  categorical_samples = tf.gather(unique_samples, raw_categorical_samples, axis=0)
  uniform_samples = tfp.distributions.Bernoulli(logits=[0.0] * num_bits, dtype=tf.int8).sample(resample_choices[1])

  return tf.concat([categorical_samples, uniform_samples], 0)
