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
"""Tests for qhbmlib.inference.ebm"""

import functools
import itertools
import random

import tensorflow as tf
import tensorflow_probability as tfp

from qhbmlib import inference
from qhbmlib import models
from qhbmlib import utils

from tests import test_util


class EnergyInferenceTest(tf.test.TestCase):
  """Tests a simple instantiation of EnergyInference."""

  class EnergyInferenceBernoulliSampler(inference.EnergyInference):
    """EnergyInference whose sampler is just a Bernoulli."""

    def __init__(self, energy, num_expectation_samples, initial_seed):
      """See base class docstring."""
      super().__init__(energy, num_expectation_samples, initial_seed)
      self._logits_variable = tf.Variable(energy.logits, trainable=False)
      self._distribution = tfp.distributions.Bernoulli(
          logits=self._logits_variable, dtype=tf.int8)

    def _ready_inference(self):
      """See base class docstring."""
      self._logits_variable.assign(self.energy.logits)

    def _call(self, inputs):
      """Pass through to sample."""
      return self.sample(inputs)

    def _sample(self, num_samples: int):
      """See base class docstring"""
      return self._distribution.sample(num_samples, seed=self.seed)

  def setUp(self):
    """Initializes test objects."""
    super().setUp()

    self.close_rtol = 2e-2
    self.close_atol = 2e-3
    self.not_zero_atol = 4e-3
    self.num_samples = int(1e6)

    self.num_bits = 5
    self.all_bitstrings = tf.constant(
        list(itertools.product([0, 1], repeat=self.num_bits)), dtype=tf.int8)
    self.tfp_seed = tf.constant([3, 4], tf.int32)
    self.tf_random_seed = 4
    energy_init = tf.keras.initializers.RandomUniform(seed=self.tf_random_seed)
    self.energy = models.BernoulliEnergy(
        list(range(self.num_bits)), initializer=energy_init)
    # TODO(#209)
    _ = self.energy(tf.constant([[0] * self.num_bits], dtype=tf.int8))
    self.ebm = self.EnergyInferenceBernoulliSampler(self.energy,
                                                    self.num_samples,
                                                    self.tfp_seed)

    spins_from_bitstrings = models.SpinsFromBitstrings()
    parity = models.Parity(list(range(self.num_bits)), 2)

    def test_function(bitstrings):
      """Simple test function to send to expectation."""
      return parity(spins_from_bitstrings(bitstrings))

    self.test_function = test_function

  @test_util.eager_mode_toggle
  def test_entropy(self):
    """Compares estimated entropy to exact value."""

    def manual_entropy():
      """Returns the exact entropy of the distribution."""
      return tf.reduce_sum(
          tfp.distributions.Bernoulli(logits=self.energy.logits).entropy())

    expected_entropy = manual_entropy()
    entropy_wrapper = tf.function(self.ebm.entropy)
    actual_entropy = entropy_wrapper()
    self.assertAllClose(actual_entropy, expected_entropy, rtol=self.close_rtol)

    expected_gradient = test_util.approximate_gradient(
        manual_entropy, self.energy.trainable_variables)
    with tf.GradientTape() as tape:
      value = entropy_wrapper()
    actual_gradient = tape.gradient(value, self.energy.trainable_variables)
    self.assertAllClose(
        actual_gradient, expected_gradient, rtol=self.close_rtol)

  @test_util.eager_mode_toggle
  def test_expectation(self):
    """Confirms correct averaging over input function."""

    def manual_expectation(f):
      """A manual function for taking expectation values."""
      samples = tfp.distributions.Bernoulli(logits=self.energy.logits).sample(
          self.num_samples, seed=self.tfp_seed)
      unique_samples, _, counts = utils.unique_bitstrings_with_counts(samples)
      values = f(unique_samples)
      return utils.weighted_average(counts, values)

    expected_expectation = manual_expectation(self.test_function)
    expectation_wrapper = tf.function(self.ebm.expectation)
    actual_expectation = expectation_wrapper(self.test_function)
    self.assertAllClose(
        actual_expectation, expected_expectation, rtol=self.close_rtol)

    expected_gradient = test_util.approximate_gradient(
        functools.partial(manual_expectation, self.test_function),
        self.energy.trainable_variables)
    with tf.GradientTape() as tape:
      value = expectation_wrapper(self.test_function)
    actual_gradient = tape.gradient(value, self.energy.trainable_variables)
    self.assertAllClose(
        actual_gradient, expected_gradient, rtol=self.close_rtol)

  @test_util.eager_mode_toggle
  def test_log_partition(self):
    """Confirms log partition function and derivative match analytic."""

    def manual_log_partition():
      """Returns the exact log partition function."""
      return tf.reduce_logsumexp(-1.0 * self.energy(self.all_bitstrings))

    expected_value = manual_log_partition()

    log_partition_wrapper = tf.function(self.ebm.log_partition)
    actual_value = log_partition_wrapper()
    self.assertAllClose(actual_value, expected_value, rtol=self.close_rtol)

    expected_gradient = test_util.approximate_gradient(
        manual_log_partition, self.energy.trainable_variables)
    self.assertNotAllClose(
        tf.math.abs(expected_gradient),
        tf.zeros_like(expected_gradient),
        atol=self.not_zero_atol)
    with tf.GradientTape() as tape:
      v = log_partition_wrapper()
    actual_gradient = tape.gradient(v, self.energy.trainable_variables)
    self.assertAllClose(
        actual_gradient, expected_gradient, atol=self.close_atol)


class AnalyticEnergyInferenceTest(tf.test.TestCase):
  """Tests the AnalyticEnergyInference class."""

  def setUp(self):
    """Initializes test objects."""
    super().setUp()
    self.num_samples = int(5e6)
    self.tf_random_seed = 4
    self.tfp_seed = tf.constant([3, 4], tf.int32)
    self.close_rtol = 1e-2
    self.zero_atol = 1e-5
    self.not_zero_atol = 1e-1

  def test_init(self):
    """Confirms internal values are set correctly."""
    bits = [0, 1, 3]
    order = 2
    expected_name = "test_analytic_dist_name"
    actual_energy = models.KOBE(bits, order)
    expected_bitstrings = tf.constant(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1],
         [1, 1, 0], [1, 1, 1]],
        dtype=tf.int8)
    expected_seed = tf.constant([44, 22], tf.int32)
    expected_energies = actual_energy(expected_bitstrings)
    actual_layer = inference.AnalyticEnergyInference(actual_energy,
                                                     self.num_samples,
                                                     expected_seed,
                                                     expected_name)
    self.assertEqual(actual_layer.name, expected_name)
    self.assertAllEqual(actual_layer.seed, expected_seed)
    self.assertAllEqual(actual_layer.all_bitstrings, expected_bitstrings)
    self.assertAllClose(actual_layer.all_energies, expected_energies)
    self.assertIsInstance(actual_layer.distribution,
                          tfp.distributions.Categorical)

  @test_util.eager_mode_toggle
  def test_sample(self):
    """Confirms bitstrings are sampled as expected."""

    # Single bit test.
    one_bit_energy = models.KOBE([0], 1)
    actual_layer = inference.AnalyticEnergyInference(
        one_bit_energy, self.num_samples, initial_seed=self.tfp_seed)

    # For single factor Bernoulli, theta=0 is 50% chance of 1.
    one_bit_energy.set_weights([tf.constant([0.0])])
    sample_wrapper = tf.function(actual_layer.sample)
    samples = sample_wrapper(self.num_samples)
    # check that we got both bitstrings
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0], dtype=tf.int8), samples))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1], dtype=tf.int8), samples))
    # Check that the fraction is approximately 0.5 (equal counts)
    _, _, counts = utils.unique_bitstrings_with_counts(samples)
    self.assertAllClose(1.0, counts[0] / counts[1], rtol=self.close_rtol)

    # Large energy penalty pins the bit.
    one_bit_energy.set_weights([tf.constant([100.0])])
    samples = sample_wrapper(self.num_samples)
    # check that we got only one bitstring
    self.assertFalse(
        test_util.check_bitstring_exists(
            tf.constant([0], dtype=tf.int8), samples))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1], dtype=tf.int8), samples))

    # Three bit tests.
    # First a uniform sampling test.
    three_bit_energy = models.KOBE([0, 1, 2], 3,
                                   tf.keras.initializers.Constant(0.0))
    actual_layer = inference.AnalyticEnergyInference(
        three_bit_energy, self.num_samples, initial_seed=self.tfp_seed)

    # Redefine sample wrapper because we made a new AnalyticEnergyInference.
    sample_wrapper = tf.function(actual_layer.sample)
    samples = sample_wrapper(self.num_samples)
    for b in [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1],
              [1, 1, 0], [1, 1, 1]]:
      b_tf = tf.constant([b], dtype=tf.int8)
      self.assertTrue(test_util.check_bitstring_exists(b_tf, samples))

    _, _, counts = utils.unique_bitstrings_with_counts(samples)
    # Check that the fraction is approximately 0.125 (equal counts)
    self.assertAllClose(
        [0.125] * 8,
        tf.cast(counts, tf.float32) / tf.cast(self.num_samples, tf.float32),
        rtol=self.close_rtol,
    )

    # Confirm correlated spins.
    three_bit_energy.set_weights(
        [tf.constant([100.0, 0.0, 0.0, -100.0, 0.0, 100.0, 0.0])])
    samples = sample_wrapper(self.num_samples)

    # Confirm we only get the 110 bitstring.
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1, 1, 0], dtype=tf.int8), samples))
    for b in [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        # [1, 1, 0],
        [1, 1, 1]
    ]:
      b_tf = tf.constant([b], dtype=tf.int8)
      self.assertFalse(test_util.check_bitstring_exists(b_tf, samples))

  @test_util.eager_mode_toggle
  def test_samples_seeded(self):
    """Confirm seeding fixes samples for given energy."""
    num_bits = 5
    actual_energy = models.KOBE(list(range(num_bits)), 2)
    actual_layer = inference.AnalyticEnergyInference(
        actual_energy, self.num_samples, initial_seed=self.tfp_seed)

    sample_wrapper = tf.function(actual_layer.sample)
    samples_1 = sample_wrapper(self.num_samples)
    samples_2 = sample_wrapper(self.num_samples)
    self.assertAllEqual(samples_1, samples_2)

    # check unseeding lets samples be different again
    actual_layer.seed = None
    samples_1 = sample_wrapper(self.num_samples)
    samples_2 = sample_wrapper(self.num_samples)
    self.assertNotAllEqual(samples_1, samples_2)

  @test_util.eager_mode_toggle
  def test_expectation_explicit(self):
    r"""Test expectation value and derivative with simple energy.

    Let $\bm{1}$ be the all ones bitstring.  Then let the energy function be
    $$ E_\theta(x) = \begin{cases}
                         \theta, & \text{if}\ x = \bm{1} \\
                         0, & \text{otherwise}
                     \end{cases} $$
    Given this energy function, the partition function is
    $$ Z_\theta = \sum_x e^{-E_\theta (x)} = 2^N - 1 + e^{-\theta}$$
    and the corresponding probability distribution is
    $$ p_\theta(x) = \begin{cases}
                         Z_\theta^{-1} e^{-\theta}, & \text{if}\ x = \bm{1}\\
                         Z_\theta^{-1}, & \text{otherwise}
                     \end{cases} $$

    Suppose the function to average is
    $$ f(x) = \begin{cases}
                  \mu, & \text{if}\ x = \bm{1} \\
                  0, & \text{otherwise}
              \end{cases} $$
    and let $X$ be a random variable distributed according to $p_\theta$.  Then,
    $$ \mathbb{E}_{x \sim X} [f(x)] = \mu p_\theta(\bm{1})$$

    # TODO(#119)
    From equation A3 in the appendix, we have
    $$ \nabla_\theta p_\theta(x) = p_\theta(x) \left(
         \mathbb{E}_{x\sim X}\left[\nabla_\theta E_\theta(x)\right]
         - \nabla_\theta E_\theta(x)
       \right) $$
    Filling in $\nabla_\theta E_\theta(\bm{1}) = 1$ and
    $\mathbb{E}_{x\sim X}[\nabla_\theta E_\theta(x)] = p_\theta(\bm{1})$
    we have
    $$\nabla_\theta p_\theta(\bm{1}) = p_\theta(\bm{1})(p_\theta(\bm{1}) - 1)$$
    Thus
    $$ \nabla_\theta \mathbb{E}_{x \sim X} [f(x)] =
           \mu p_\theta(\bm{1})(p_\theta(\bm{1}) - 1) $$

    Suppose now the function to average contains the same variable as energy,
    $$ g(x) = \begin{cases}
                  \theta, &\text{if}\ x = \bm{1} \\
                  0, & \text{otherwise}
              \end{cases} $$
    Then,
    $$ \mathbb{E}_{x \sim X} [g(x)] = \theta p_\theta(\bm{1})$$
    and the derivative becomes
    $$ \nabla_\theta \mathbb{E}_{x \sim X} [f(x)] =
           \theta p_\theta(\bm{1})(p_\theta(\bm{1}) - 1) + p_\theta(\bm{1})$$
    """

    class AllOnes(tf.keras.layers.Layer):
      """Detects all ones."""

      def __init__(self, ones_prefactor):
        """ Initializes an AllOnes layer.

        Args:
          ones_prefactor: the scalar to emit when all ones is detected.
        """
        super().__init__()
        self.ones_prefactor = ones_prefactor

      def call(self, inputs):
        """Return prefactor for scalar"""
        return self.ones_prefactor * tf.math.reduce_prod(
            tf.cast(inputs, tf.float32), 1)

    num_bits = 3
    theta = tf.Variable(tf.random.uniform([], -3, -2), name="theta")
    energy_layers = [AllOnes(theta)]
    actual_energy = models.BitstringEnergy(list(range(num_bits)), energy_layers)
    theta_exp = tf.math.exp(-1.0 * theta)
    partition = tf.math.pow(2.0, num_bits) - 1 + theta_exp
    partition_inverse = tf.math.pow(partition, -1)
    prob_x_star = partition_inverse * theta_exp

    e_infer = inference.AnalyticEnergyInference(
        actual_energy, self.num_samples, initial_seed=self.tfp_seed)

    mu = tf.Variable(tf.random.uniform([], 1, 2), name="mu")
    f = AllOnes(mu)
    expected_average = mu * prob_x_star
    expected_gradient_theta = mu * prob_x_star * (prob_x_star - 1)
    expected_gradient_mu = prob_x_star

    expectation_wrapper = tf.function(e_infer.expectation)
    with tf.GradientTape() as tape:
      actual_average = expectation_wrapper(f)
    actual_gradient_theta, actual_gradient_mu = tape.gradient(
        actual_average, (theta, mu))

    # Confirm expectations and gradients are not negligible
    self.assertAllGreater(tf.math.abs(actual_average), self.not_zero_atol)
    self.assertAllGreater(
        tf.math.abs(actual_gradient_theta), self.not_zero_atol)
    self.assertAllGreater(tf.math.abs(actual_gradient_mu), self.not_zero_atol)

    self.assertAllClose(actual_average, expected_average, rtol=self.close_rtol)
    self.assertAllClose(
        actual_gradient_theta, expected_gradient_theta, rtol=self.close_rtol)
    self.assertAllClose(
        actual_gradient_mu, expected_gradient_mu, rtol=self.close_rtol)

    # Confirm gradients are connected upstream
    mul_const = tf.random.uniform([], -2, -1)

    def wrap_f(bitstrings):
      return mul_const * f(bitstrings)

    mul_expected_average = mul_const * expected_average
    mul_expected_gradient_theta = mul_const * expected_gradient_theta
    mul_expected_gradient_mu = mul_const * expected_gradient_mu

    with tf.GradientTape() as tape:
      actual_average = expectation_wrapper(wrap_f)
    actual_gradient_theta, actual_gradient_mu = tape.gradient(
        actual_average, (theta, mu))

    self.assertAllClose(
        actual_average, mul_expected_average, rtol=self.close_rtol)
    self.assertAllClose(
        actual_gradient_theta,
        mul_expected_gradient_theta,
        rtol=self.close_rtol)
    self.assertAllClose(
        actual_gradient_mu, mul_expected_gradient_mu, rtol=self.close_rtol)

    # Test a function sharing variables with the energy.
    g = AllOnes(theta)
    expected_average = theta * prob_x_star
    expected_gradient_theta = theta * prob_x_star * (prob_x_star -
                                                     1) + prob_x_star

    with tf.GradientTape() as tape:
      actual_average = expectation_wrapper(g)
    actual_gradient_theta, actual_gradient_mu = tape.gradient(
        actual_average, (theta, mu))

    # Confirm expectation and gradients are not negligible
    self.assertAllGreater(
        tf.math.abs(actual_gradient_theta), self.not_zero_atol)

    self.assertAllClose(actual_average, expected_average, rtol=self.close_rtol)
    self.assertAllClose(
        actual_gradient_theta, expected_gradient_theta, rtol=self.close_rtol)
    # Check unconnected gradient
    self.assertIsNone(actual_gradient_mu)

    # Check unconnected gradient with zeros tape setting
    with tf.GradientTape() as tape:
      actual_average = expectation_wrapper(g)
    actual_gradient_mu = tape.gradient(
        actual_average, mu, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    self.assertAllLess(tf.math.abs(actual_gradient_mu), self.zero_atol)

  @test_util.eager_mode_toggle
  def test_expectation_finite_difference(self):
    """Tests a function with nested structural output."""

    num_bits = 3
    order = 2
    nonzero_init = tf.keras.initializers.RandomUniform(
        1, 2, seed=self.tf_random_seed)
    actual_energy = models.KOBE(list(range(num_bits)), order, nonzero_init)
    e_infer = inference.AnalyticEnergyInference(
        actual_energy, self.num_samples, initial_seed=self.tfp_seed)
    # Only one trainable variable in KOBE.
    energy_var = actual_energy.trainable_variables[0]

    scalar_var = tf.Variable(
        tf.random.uniform([], 1, 2, tf.float32, self.tf_random_seed))

    num_units = 5
    dense = tf.keras.layers.Dense(
        num_units,
        kernel_initializer=nonzero_init,
        bias_initializer=nonzero_init)
    dense.build([None, num_bits])

    def f(bitstrings):
      """Returns nested batched structure which is a function of the inputs."""
      reduced = tf.cast(tf.reduce_sum(bitstrings, 1), tf.float32)
      ret_scalar = scalar_var * reduced
      ret_vector = dense(bitstrings)
      ret_thetas = [tf.einsum("i,j->ij", reduced, energy_var)]
      return [ret_scalar, ret_vector, ret_thetas]

    expectation_wrapper = tf.function(e_infer.expectation)
    with tf.GradientTape() as tape:
      actual_expectation = expectation_wrapper(f)
    actual_gradient = tape.gradient(actual_expectation, energy_var)

    def expectation_func():
      """Evaluate the current expectation value."""
      samples = e_infer.sample(self.num_samples)
      bitstrings, _, counts = utils.unique_bitstrings_with_counts(samples)
      values = f(bitstrings)
      return tf.nest.map_structure(lambda x: utils.weighted_average(counts, x),
                                   values)

    expected_expectation = expectation_func()
    tf.nest.map_structure(
        lambda x: self.assertAllGreater(tf.abs(x), self.not_zero_atol),
        expected_expectation)
    self.assertAllClose(actual_expectation, expected_expectation)

    expected_gradient = test_util.approximate_gradient(expectation_func,
                                                       energy_var)
    tf.nest.map_structure(
        lambda x: self.assertAllGreater(tf.abs(x), self.not_zero_atol),
        expected_gradient)
    self.assertAllClose(
        actual_gradient, expected_gradient, rtol=self.close_rtol)

  @test_util.eager_mode_toggle
  def test_log_partition(self):
    """Confirms correct value of the log partition function."""
    test_thetas = tf.constant([1.5, 2.7, -4.0])
    expected_log_partition = tf.math.log(tf.constant(3641.8353))

    actual_energy = models.KOBE([0, 1], 2)
    actual_layer = inference.AnalyticEnergyInference(actual_energy,
                                                     self.num_samples)
    actual_energy.set_weights([test_thetas])

    log_partition_wrapper = tf.function(actual_layer.log_partition)
    with tf.GradientTape() as tape:
      actual_log_partition = log_partition_wrapper()
    self.assertAllClose(actual_log_partition, expected_log_partition)

    old_kernel = actual_energy.post_process[0].kernel.read_value()
    kernel_len = tf.shape(old_kernel)[0].numpy().tolist()
    all_bitstrings = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]],
                                 dtype=tf.int8)

    def exact_log_partition():
      """Calculates the current log partition."""
      return tf.reduce_logsumexp(-1.0 * actual_energy(all_bitstrings))

    actual_log_partition_grad = tape.gradient(actual_log_partition,
                                              actual_energy.trainable_variables)
    expected_log_partition_grad = test_util.approximate_gradient(
        exact_log_partition, actual_energy.trainable_variables)
    self.assertAllClose(actual_log_partition_grad, expected_log_partition_grad,
                        self.close_rtol)

  @test_util.eager_mode_toggle
  def test_entropy(self):
    """Confirms correct value of the entropy function."""
    test_thetas = tf.constant([1.5, 2.7, -4.0])
    expected_entropy = tf.constant(0.00233551808)

    actual_energy = models.KOBE([0, 1], 2)
    actual_layer = inference.AnalyticEnergyInference(actual_energy,
                                                     self.num_samples)
    actual_energy.set_weights([test_thetas])

    entropy_wrapper = tf.function(actual_layer.entropy)
    actual_entropy = entropy_wrapper()
    self.assertAllClose(actual_entropy, expected_entropy)

  @test_util.eager_mode_toggle
  def test_call(self):
    """Confirms that call behaves correctly."""
    one_bit_energy = models.KOBE([0], 1, tf.keras.initializers.Constant(0.0))
    actual_layer = inference.AnalyticEnergyInference(
        one_bit_energy, self.num_samples, initial_seed=self.tfp_seed)
    actual_dist = actual_layer(None)
    self.assertIsInstance(actual_dist, tfp.distributions.Categorical)

    actual_layer_wrapper = tf.function(actual_layer)
    samples = actual_layer_wrapper(self.num_samples)
    # check that we got both bitstrings
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0], dtype=tf.int8), samples))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1], dtype=tf.int8), samples))
    # Check that the fraction is approximately 0.5 (equal counts)
    _, _, counts = utils.unique_bitstrings_with_counts(samples)
    self.assertAllClose(1.0, counts[0] / counts[1], rtol=self.close_rtol)


class BernoulliEnergyInferenceTest(tf.test.TestCase):
  """Tests the BernoulliEnergyInference class."""

  def setUp(self):
    """Initializes test objects."""
    super().setUp()
    self.num_samples = int(5e6)
    self.tf_random_seed = 4
    self.tfp_seed = tf.constant([3, 4], tf.int32)
    self.close_rtol = 1e-2
    self.not_zero_atol = 1e-1

  def test_init(self):
    """Tests that components are initialized correctly."""
    bits = [-3, 4, 6]
    expected_name = "test_bernoulli_dist_name"
    actual_energy = models.BernoulliEnergy(bits)
    expected_seed = tf.constant([4, 12], dtype=tf.int32)
    actual_layer = inference.BernoulliEnergyInference(actual_energy,
                                                      self.num_samples,
                                                      expected_seed,
                                                      expected_name)
    self.assertEqual(actual_layer.name, expected_name)
    self.assertAllEqual(actual_layer.seed, expected_seed)
    self.assertIsInstance(actual_layer.distribution,
                          tfp.distributions.Bernoulli)

  @test_util.eager_mode_toggle
  def test_sample(self):
    """Confirms that bitstrings are sampled as expected."""
    actual_energy = models.BernoulliEnergy([1])
    actual_layer = inference.BernoulliEnergyInference(
        actual_energy, self.num_samples, initial_seed=self.tfp_seed)

    # For single factor Bernoulli, theta = 0 is 50% chance of 1.
    actual_energy.set_weights([tf.constant([0.0])])

    sample_wrapper = tf.function(actual_layer.sample)
    samples = sample_wrapper(self.num_samples)
    # check that we got both bitstrings
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0], dtype=tf.int8), samples))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1], dtype=tf.int8), samples))
    # Check that the fraction is approximately 0.5 (equal counts)
    _, _, counts = utils.unique_bitstrings_with_counts(samples)
    self.assertAllClose(1.0, counts[0] / counts[1], rtol=self.close_rtol)

    # Large value of theta pins the bit.
    actual_energy.set_weights([tf.constant([1000.0])])

    samples = sample_wrapper(self.num_samples)
    # check that we got only one bitstring
    bitstrings, _, _ = utils.unique_bitstrings_with_counts(samples)
    self.assertAllEqual(bitstrings, [[1]])

    # Two bit tests.
    actual_energy = models.BernoulliEnergy([0, 1],
                                           tf.keras.initializers.Constant(0.0))
    actual_layer = inference.BernoulliEnergyInference(
        actual_energy, self.num_samples, initial_seed=self.tfp_seed)

    # New sample wrapper because we have new layer.
    sample_wrapper = tf.function(actual_layer.sample)
    samples = sample_wrapper(self.num_samples)
    for b in [[0, 0], [0, 1], [1, 0], [1, 1]]:
      b_tf = tf.constant([b], dtype=tf.int8)
      self.assertTrue(test_util.check_bitstring_exists(b_tf, samples))
    # Check that the fraction is approximately 0.25 (equal counts)
    _, _, counts = utils.unique_bitstrings_with_counts(samples)
    self.assertAllClose(
        [0.25] * 4,
        tf.cast(counts, tf.float32) / tf.cast(self.num_samples, tf.float32),
        rtol=self.close_rtol,
    )

    # Test one pinned, one free bit
    actual_energy.set_weights([tf.constant([-1000.0, 0.0])])
    samples = sample_wrapper(self.num_samples)
    # check that we get 00 and 01.
    for b in [[0, 0], [0, 1]]:
      b_tf = tf.constant([b], dtype=tf.int8)
      self.assertTrue(test_util.check_bitstring_exists(b_tf, samples))
    for b in [[1, 0], [1, 1]]:
      b_tf = tf.constant([b], dtype=tf.int8)
      self.assertFalse(test_util.check_bitstring_exists(b_tf, samples))
    _, _, counts = utils.unique_bitstrings_with_counts(samples)
    self.assertAllClose(
        counts, [self.num_samples / 2] * 2, atol=self.num_samples / 1000)

  @test_util.eager_mode_toggle
  def test_samples_seeded(self):
    """Confirm seeding fixes samples for given energy."""
    num_bits = 5
    actual_energy = models.BernoulliEnergy(list(range(num_bits)))
    actual_layer = inference.BernoulliEnergyInference(
        actual_energy, self.num_samples, initial_seed=self.tfp_seed)

    sample_wrapper = tf.function(actual_layer.sample)
    samples_1 = sample_wrapper(self.num_samples)
    samples_2 = sample_wrapper(self.num_samples)
    self.assertAllEqual(samples_1, samples_2)

    # check unseeding lets samples be different again
    actual_layer.seed = None
    samples_1 = sample_wrapper(self.num_samples)
    samples_2 = sample_wrapper(self.num_samples)
    self.assertNotAllEqual(samples_1, samples_2)

  @test_util.eager_mode_toggle
  def test_log_partition(self):
    """Confirms correct value of the log partition function and derivative."""
    all_bitstrings = tf.constant([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                                  [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                                 dtype=tf.int8)
    ebm_init = tf.keras.initializers.RandomUniform(
        -2, -1, seed=self.tf_random_seed)
    actual_energy = models.BernoulliEnergy([5, 6, 7], ebm_init)
    actual_layer = inference.BernoulliEnergyInference(actual_energy,
                                                      self.num_samples,
                                                      self.tfp_seed)
    expected_log_partition = tf.reduce_logsumexp(-1.0 *
                                                 actual_energy(all_bitstrings))

    log_partition_wrapper = tf.function(actual_layer.log_partition)
    with tf.GradientTape() as tape:
      actual_log_partition = log_partition_wrapper()
    self.assertAllClose(actual_log_partition, expected_log_partition)

    old_kernel = actual_energy.post_process[0].kernel.read_value()
    kernel_len = tf.shape(old_kernel)[0].numpy().tolist()

    def exact_log_partition():
      """Returns the current value of log partition."""
      return tf.reduce_logsumexp(-1.0 * actual_energy(all_bitstrings))

    actual_log_partition_grad = tape.gradient(actual_log_partition,
                                              actual_energy.trainable_variables)
    expected_log_partition_grad = test_util.approximate_gradient(
        exact_log_partition, actual_energy.trainable_variables)
    self.assertAllClose(actual_log_partition_grad, expected_log_partition_grad,
                        self.close_rtol)

  @test_util.eager_mode_toggle
  def test_entropy(self):
    r"""Confirms that the entropy is S(p) = -\sum_x p(x)\ln(p(x)).

    For logit $\eta$ and probability of 1 $p$, we have
    $\eta = \log(p / 1-p)$, so $p = \frac{e^{\eta}}{1 + e^{\eta}}$.
    """
    test_thetas = tf.constant([-1.5, 0.6, 2.1])
    logits = 2 * test_thetas
    num = tf.math.exp(logits)
    denom = 1 + num
    test_probs = (num / denom).numpy()
    all_probs = tf.constant([
        (1 - test_probs[0]) * (1 - test_probs[1]) * (1 - test_probs[2]),
        (1 - test_probs[0]) * (1 - test_probs[1]) * (test_probs[2]),
        (1 - test_probs[0]) * (test_probs[1]) * (1 - test_probs[2]),
        (1 - test_probs[0]) * (test_probs[1]) * (test_probs[2]),
        (test_probs[0]) * (1 - test_probs[1]) * (1 - test_probs[2]),
        (test_probs[0]) * (1 - test_probs[1]) * (test_probs[2]),
        (test_probs[0]) * (test_probs[1]) * (1 - test_probs[2]),
        (test_probs[0]) * (test_probs[1]) * (test_probs[2]),
    ])
    # probabilities sum to 1
    self.assertAllClose(1.0, tf.reduce_sum(all_probs))
    expected_entropy = -1.0 * tf.reduce_sum(all_probs * tf.math.log(all_probs))

    actual_energy = models.BernoulliEnergy([0, 1, 2])
    actual_layer = inference.BernoulliEnergyInference(actual_energy,
                                                      self.num_samples)
    actual_energy.set_weights([test_thetas])

    entropy_wrapper = tf.function(actual_layer.entropy)
    actual_entropy = entropy_wrapper()
    self.assertAllClose(actual_entropy, expected_entropy)

  @test_util.eager_mode_toggle
  def test_call(self):
    """Confirms that calling the layer works correctly."""
    actual_energy = models.BernoulliEnergy([1],
                                           tf.keras.initializers.Constant(0.0))
    actual_layer = inference.BernoulliEnergyInference(
        actual_energy, self.num_samples, initial_seed=self.tfp_seed)
    actual_dist = actual_layer(None)
    self.assertIsInstance(actual_dist, tfp.distributions.Bernoulli)

    # For single factor Bernoulli, theta = 0 is 50% chance of 1.
    actual_layer_wrapper = tf.function(actual_layer)
    samples = actual_layer_wrapper(self.num_samples)
    # check that we got both bitstrings
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0], dtype=tf.int8), samples))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1], dtype=tf.int8), samples))
    # Check that the fraction is approximately 0.5 (equal counts)
    _, _, counts = utils.unique_bitstrings_with_counts(samples)
    self.assertAllClose(1.0, counts[0] / counts[1], rtol=self.close_rtol)


class GibbsWithGradientsKernelTest(tf.test.TestCase):
  """Tests the GibbsWithGradientsKernel class."""

  def test_init(self):
    """Confirms initialization works, and tests basic functions."""
    bits = [0, 1, 3]
    order = 2
    expected_energy = models.KOBE(bits, order)
    actual_kernel = inference.ebm.GibbsWithGradientsKernel(expected_energy)
    self.assertEqual(actual_kernel._energy, expected_energy)

    self.assertTrue(actual_kernel.is_calibrated)

    test_initial_state = tf.constant([0] * len(bits), tf.int8)
    self.assertEqual(actual_kernel.bootstrap_results(test_initial_state), [])

  @test_util.eager_mode_toggle
  def test_get_index_proposal_probs(self):
    """Confirms the distribution returned is correct."""
    num_bits = 3
    bits = random.sample(range(1000), num_bits)
    bernoulli_energy = models.BernoulliEnergy(bits)
    bernoulli_energy.build([None, num_bits])
    # make sure the probabilities are unique
    bernoulli_energy.set_weights([tf.constant([-2.0, 1.0, 3.0])])
    actual_kernel = inference.ebm.GibbsWithGradientsKernel(bernoulli_energy)
    test_x = tf.constant([0, 1, 1], dtype=tf.int8)
    get_index_proposal_probs_wrapper = tf.function(
        actual_kernel._get_index_proposal_probs)
    actual_probs = get_index_proposal_probs_wrapper(test_x)

    hamming_ball_energies = bernoulli_energy(
        tf.constant([[1, 1, 1], [0, 0, 1], [0, 1, 0]], dtype=tf.int8))
    test_x_energies = bernoulli_energy(
        tf.tile(tf.expand_dims(test_x, 0), [3, 1]))
    # f(x') - f(x) = -E(x') + E(x)
    direct_energy_diff = -hamming_ball_energies + test_x_energies
    expected_probs = tf.nn.softmax(direct_energy_diff / 2)
    self.assertAllClose(actual_probs, expected_probs)

  @test_util.eager_mode_toggle
  def test_one_step(self):
    """Confirms transitions occur from high to low energy states."""
    num_bits = 5
    test_energy = models.BernoulliEnergy(list(range(num_bits)))
    test_energy.build([None, num_bits])
    # Set the energy high for the all zeros state
    test_energy.set_weights([tf.constant([10] * num_bits, tf.float32)])
    actual_kernel = inference.ebm.GibbsWithGradientsKernel(test_energy)

    initial_state = tf.constant([0] * num_bits, tf.int8)
    initial_energy = test_energy(tf.expand_dims(initial_state, 0))
    one_step_wrapper = tf.function(actual_kernel.one_step)
    next_state, _ = one_step_wrapper(initial_state, [])
    next_energy = test_energy(tf.expand_dims(next_state, 0))

    self.assertNotAllEqual(initial_state, next_state)
    self.assertGreater(initial_energy, next_energy)


class GibbsWithGradientsInferenceTest(tf.test.TestCase):
  """Tests the GibbsWithGradientsInference class."""

  def setUp(self):
    """Initializes test objects."""
    super().setUp()
    self.tf_random_seed = 11
    self.tfp_seed = tf.constant([3, 4], tf.int32)
    self.close_rtol = 1e-2

  def test_init(self):
    """Confirms internal values are set correctly."""
    bits = [0, 1, 3]
    order = 2
    expected_energy = models.KOBE(bits, order)
    expected_num_expectation_samples = 14899
    expected_num_burnin_samples = 32641
    expected_name = "test_analytic_dist_name"
    actual_layer = inference.GibbsWithGradientsInference(
        expected_energy, expected_num_expectation_samples,
        expected_num_burnin_samples, expected_name)

    self.assertEqual(actual_layer.energy, expected_energy)
    self.assertAllEqual(actual_layer.num_expectation_samples,
                        expected_num_expectation_samples)
    self.assertAllEqual(actual_layer.num_burnin_samples,
                        expected_num_burnin_samples)
    self.assertEqual(actual_layer.name, expected_name)

  @test_util.eager_mode_toggle
  def test_sample(self):
    """Confirms that bitstrings are sampled as expected."""
    # Set up energy function
    num_bits = 4
    # one layer so that probabilities are less uniform
    num_layers = 2
    bits = random.sample(range(1000), num_bits)
    units = num_bits
    expected_layer_list = []
    for i in range(num_layers):
      initializer = tf.keras.initializers.Orthogonal(seed=self.tf_random_seed +
                                                     i)
      expected_layer_list.append(
          tf.keras.layers.Dense(units, kernel_initializer=initializer))
    expected_layer_list.append(
        tf.keras.layers.Dense(1, kernel_initializer=initializer))
    expected_layer_list.append(utils.Squeeze(-1))
    actual_energy = models.BitstringEnergy(bits, expected_layer_list)
    # TODO(#209)
    _ = actual_energy(tf.constant([[0] * num_bits], dtype=tf.int8))

    # Sampler
    num_expectation_samples = int(2e4)
    num_burnin_samples = int(2e3)
    actual_layer = inference.GibbsWithGradientsInference(
        actual_energy, num_expectation_samples, num_burnin_samples)

    sample_wrapper = tf.function(actual_layer.sample)
    samples = sample_wrapper(num_expectation_samples)

    all_bitstrings = tf.constant(
        list(itertools.product([0, 1], repeat=num_bits)), dtype=tf.int8)
    expected_probs = tf.math.softmax(-actual_energy(all_bitstrings))
    expected_entropy = test_util.entropy(expected_probs)
    unique_samples, _, unique_counts = utils.unique_bitstrings_with_counts(
        samples)
    actual_probs_unsorted = tf.cast(unique_counts, tf.float32) / tf.cast(
        tf.math.reduce_sum(unique_counts), tf.float32)
    actual_entropy = test_util.entropy(actual_probs_unsorted)
    self.assertAllClose(actual_entropy, expected_entropy, rtol=self.close_rtol)

    # make sure the distribution is not uniform
    num_bitstrings = 2**num_bits
    uniform_probs = tf.constant([1 / num_bitstrings] * num_bitstrings)
    uniform_entropy = test_util.entropy(uniform_probs)
    self.assertNotAllClose(
        actual_entropy, uniform_entropy, rtol=2 * self.close_rtol)

    # ensure all bitstrings have been sampled
    self.assertAllEqual(tf.shape(all_bitstrings), tf.shape(unique_samples))

    def get_corresponding_sample_prob(b):
      """Get the actual probability corresponding to the given bitstring."""
      index_truth = tf.reduce_all(tf.math.equal(b, unique_samples), 1)
      index = tf.where(index_truth)[0][0]
      return actual_probs_unsorted[index]

    actual_probs = tf.map_fn(
        get_corresponding_sample_prob,
        all_bitstrings,
        fn_output_signature=tf.float32)
    histogram_rtol = 0.1
    self.assertAllClose(actual_probs, expected_probs, rtol=histogram_rtol)


if __name__ == "__main__":
  print("Running ebm_test.py ...")
  tf.test.main()
