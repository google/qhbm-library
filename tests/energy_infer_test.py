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
"""Tests for the energy_infer module."""

import itertools

import cirq
import tensorflow as tf

from qhbmlib import energy_infer
from qhbmlib import energy_model


class BernoulliSamplerTest(tf.test.TestCase):
  """Tests the BernoulliSampler class."""

  num_bits = 5

  def test_sampler_bernoulli(self):
    """Confirm that bitstrings are sampled as expected."""
    test_b_sampler = energy_infer.BernoulliSampler()

    @tf.function
    def test_sampler_traced(dist, num_samples):
      return test_b_sampler(dist, num_samples)

    for sampler in [test_b_sampler, test_sampler_traced]:
      # For single factor Bernoulli, theta = 0 is 50% chance of 1.
      test_b = energy_model.Bernoulli([0])
      test_b.kernel.assign(tf.constant([0.0]))
      num_samples = tf.constant(int(1e7), dtype=tf.int32)
      bitstrings, counts = sampler(test_b, num_samples)
      # Confirm shapes are correct
      self.assertAllEqual(tf.shape(bitstrings), [2, 1])
      self.assertAllEqual(tf.shape(counts), [2])
      self.assertEqual(tf.math.reduce_sum(counts), num_samples)

      # check that we got both bitstrings
      self.assertTrue(
          tf.reduce_any(tf.equal(tf.constant([0], dtype=tf.int8), bitstrings)))
      self.assertTrue(
          tf.reduce_any(tf.equal(tf.constant([1], dtype=tf.int8), bitstrings)))
      # Check that the fraction is approximately 0.5 (equal counts)
      self.assertAllClose(1.0, counts[0] / counts[1], atol=1e-3)

      # Large value of theta pins the bit.
      test_b.kernel.assign(tf.constant([1000.0]))
      num_samples = tf.constant(int(1e7), dtype=tf.int32)
      bitstrings, counts = sampler(test_b, num_samples)
      # check that we got only one bitstring
      self.assertAllEqual(bitstrings, [[1]])

      # Two bit tests.
      test_b = energy_model.Bernoulli([1, 2])
      test_b.kernel.assign(tf.constant([0.0, 0.0]))
      num_samples = tf.constant(int(1e7), dtype=tf.int32)
      bitstrings, counts = sampler(test_b, num_samples)

      def check_bitstring_exists(bitstring, bitstring_list):
        return tf.math.reduce_any(
            tf.reduce_all(tf.math.equal(bitstring, bitstring_list), 1))

      self.assertTrue(
        check_bitstring_exists(tf.constant([0, 0], dtype=tf.int8), bitstrings))
      self.assertTrue(
        check_bitstring_exists(tf.constant([0, 1], dtype=tf.int8), bitstrings))
      self.assertTrue(
        check_bitstring_exists(tf.constant([1, 0], dtype=tf.int8), bitstrings))
      self.assertTrue(
        check_bitstring_exists(tf.constant([1, 1], dtype=tf.int8), bitstrings))

      # Check that the fraction is approximately 0.25 (equal counts)
      self.assertAllClose(
        [0.25] * 4,
        [counts[i].numpy() / num_samples for i in range(4)],
        atol=1e-3,
      )
      test_b.kernel.assign(tf.constant([-1000.0, 1000.0]))
      num_samples = tf.constant(int(1e7), dtype=tf.int32)
      bitstrings, counts = sampler(test_b, num_samples)
      # check that we only get 01.
      self.assertFalse(
        check_bitstring_exists(tf.constant([0, 0], dtype=tf.int8), bitstrings))
      self.assertTrue(
        check_bitstring_exists(tf.constant([0, 1], dtype=tf.int8), bitstrings))
      self.assertFalse(
        check_bitstring_exists(tf.constant([1, 0], dtype=tf.int8), bitstrings))
      self.assertFalse(
        check_bitstring_exists(tf.constant([1, 1], dtype=tf.int8), bitstrings))
      self.assertAllEqual(counts, [num_samples])


# class MLPTest(tf.test.TestCase):
#   num_bits = 5

#   def test_trainable_variables_mlp(self):
#     test_mlp = ebm.MLP(
#         self.num_bits, [4, 3, 2], activations=['relu', 'tanh', 'sigmoid'])

#     i = 0
#     for layer in test_mlp.layers:
#       self.assertAllEqual(layer.kernel, test_mlp.trainable_variables[i])
#       self.assertAllEqual(layer.bias, test_mlp.trainable_variables[i + 1])
#       i += 2

#     variables = [
#         tf.random.uniform(tf.shape(v)) for v in test_mlp.trainable_variables
#     ]
#     test_mlp.trainable_variables = variables
#     for i in range(len(test_mlp.trainable_variables)):
#       self.assertAllEqual(variables[i], test_mlp.trainable_variables[i])

#     variables = [tf.Variable(v) for v in variables]
#     test_mlp.trainable_variables = variables
#     for i in range(len(test_mlp.trainable_variables)):
#       self.assertAllEqual(variables[i], test_mlp.trainable_variables[i])


if __name__ == "__main__":
  print("Running ebm_test.py ...")
  tf.test.main()
