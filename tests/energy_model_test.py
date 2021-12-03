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
"""Tests for the energy_model module."""

import itertools
import random

import cirq
import tensorflow as tf

from qhbmlib import energy_model


class BernoulliTest(tf.test.TestCase):
  """Test the Bernoulli class."""

  def test_init(self):
    """Test that components are initialized correctly."""
    num_bits = 5
    bits = [i for i in range(num_bits)]
    init_const = 1.5
    test_name = "init_test"
    test_b = energy_model.Bernoulli(
      bits,
      tf.keras.initializers.Constant(init_const),
      name=test_name)
    self.assertEqual(test_b.num_bits, num_bits)
    self.assertAllEqual(test_b.bits, bits)
    self.assertAllEqual(test_b.kernel, [init_const] * num_bits)
    self.assertEqual(test_b.name, test_name)

  def test_init_error(self):
    """Confirms bad inputs are caught."""
    with self.assertRaisesRegex(TypeError, expected_regex="a list of integers"):
      _ = energy_model.Bernoulli(90)
    with self.assertRaisesRegex(TypeError, expected_regex="a list of integers"):
      _ = energy_model.Bernoulli(["junk"])
    with self.assertRaisesRegex(ValueError, expected_regex="must be unique"):
      _ = energy_model.Bernoulli([1, 1])

  def test_copy(self):
    """Test that the copy has the same values, but new variables."""
    num_bits = 8
    bits = [i for i in range(num_bits)]
    test_b = energy_model.Bernoulli(bits, name="test_copy")
    test_b_copy = test_b.copy()
    self.assertEqual(test_b_copy.num_bits, test_b.num_bits)
    self.assertAllEqual(test_b_copy.bits, test_b.bits)
    self.assertAllEqual(test_b_copy.kernel, test_b.kernel)
    self.assertNotEqual(id(test_b_copy.kernel), id(test_b.kernel))
    self.assertEqual(test_b_copy.name, test_b.name)

  def test_trainable_variables_bernoulli(self):
    bits = [1, 3, 4, 8, 9]
    test_b = energy_model.Bernoulli(bits, name="test")
    self.assertAllEqual(test_b.kernel, test_b.trainable_variables[0])

    kernel = tf.random.uniform([len(bits)])
    test_b.trainable_variables = [kernel]
    self.assertAllEqual(kernel, test_b.trainable_variables[0])

    kernel = tf.Variable(tf.random.uniform([len(bits)]))
    test_b.trainable_variables = [kernel]
    self.assertAllEqual(kernel, test_b.trainable_variables[0])

  def test_energy_bernoulli_simple(self):
    """Test Bernoulli.energy and its derivative in a simple case.

    For a given bitstring b, the energy is
      $$E_\theta(b) = \sum_i (1-2b_i)\theta_i$$
    Then the derivative of the energy with respect to the thetas vector is
      $$\partial / \partial \theta E_\theta(b) = [(1-2b_i) for b_i in b]$$
    """
    test_b = energy_model.Bernoulli([1, 2, 3])
    test_vars = tf.constant([1.0, 1.7, -2.8], dtype=tf.float32)
    test_b.kernel.assign(test_vars)
    test_bitstrings = tf.constant([[0, 0, 0], [1, 0, 0], [0, 0, 1]])
    test_spins = 1 - 2 * test_bitstrings
    with tf.GradientTape() as tape:
      test_energy = test_b.energy(test_bitstrings)
    test_energy_grad = tape.jacobian(test_energy, test_b.kernel)
    ref_energy = [
      test_vars[0] + test_vars[1] + test_vars[2],
      -test_vars[0] + test_vars[1] + test_vars[2],
      test_vars[0] + test_vars[1] - test_vars[2]]
    self.assertAllClose(test_energy, ref_energy)
    self.assertAllClose(test_energy_grad, [test_spins])

  def test_energy_bernoulli_simple(self):
    """Test Bernoulli.energy and its derivative in all cases.

    For a given bitstring b, the energy is
      $$E_\theta(b) = \sum_i (1-2b_i)\theta_i$$
    Then the derivative of the energy with respect to the thetas vector is
      $$\partial / \partial \theta E_\theta(b) = [(1-2b_i) for b_i in b]$$
    """
    num_bits = 9
    test_bitstrings = tf.constant(
        list(itertools.product([0, 1], repeat=num_bits)))
    test_spins = 1 - 2 * test_bitstrings
    for _ in range(10):
      bits = random.sample(range(1000), num_bits)
      thetas = tf.random.uniform([num_bits], minval=-100, maxval=100)
      test_b = energy_model.Bernoulli(bits)
      test_b.kernel.assign(thetas)      
      with tf.GradientTape() as tape:
        test_energy = test_b.energy(test_bitstrings)

      ref_energy = tf.reduce_sum(tf.cast(test_spins, tf.float32) * thetas, -1)
      self.assertAllClose(test_energy, ref_energy)

      test_energy_grad = tape.jacobian(test_energy, test_b.trainable_variables)
      self.assertAllClose(test_energy_grad, [test_spins])
