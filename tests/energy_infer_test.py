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

import functools

import tensorflow as tf
import tensorflow_probability as tfp

from qhbmlib import energy_infer
from qhbmlib import energy_model
from qhbmlib import energy_model_utils
from qhbmlib import utils

from tests import test_util


class EnergyInferenceTest(tf.test.TestCase):
  """Tests a simple instantiation of EnergyInference."""

  class TwoOutcomes(energy_infer.EnergyInference):
    """EnergyInference which is independent of the input energy."""

    def __init__(self, bitstring_1, bitstring_2, p_1):
      """Initializes a simple inference class.

      Args:
        bitstring_1: First bitstring to sample.
        bitstring_2: Second bitstring to sample.
        p_1: probability of sampling the first bitstring.
      """
      super().__init__()
      self.bitstring_1 = bitstring_1
      self.bitstring_2 = bitstring_2
      self.p_1 = p_1

    def _ready_inference(self):
      """Does nothing."""
      pass

    def _sample(self, n):
      """Deterministically samples bitstrings."""
      n_1 = round(self.p_1 * n)
      n_2 = n - n_1
      bitstring_1_tile = tf.tile(tf.expand_dims(self.bitstring_1, 0), [n_1, 1])
      bitstring_2_tile = tf.tile(tf.expand_dims(self.bitstring_2, 0), [n_2, 1])
      return tf.concat([bitstring_1_tile, bitstring_2_tile], 0)

    def _entropy(self):
      """Not implemented in this test class."""
      raise NotImplementedError()

    def _log_partition(self):
      """Not implemented in this test class."""
      raise NotImplementedError()
    
  class NullEnergy(energy_model.BitstringEnergy):
    """Simple empty energy."""

    def __init__(self, bits):
      """Initializes a NullEnergy."""
      energy_layers = []
      super().__init__(bits, energy_layers)

  class InferenceCheck(energy_infer.EnergyInference):
    """Class for checking variable updating."""

    def _sample(self, n):
      """Default implementation wrapped by `self.sample`."""
      raise NotImplementedError()

    def _entropy(self):
      """Default implementation wrapped by `self.entropy`."""
      raise NotImplementedError()

    def _log_partition(self):
      """Default implementation wrapped by `self.log_partition`."""
      raise NotImplementedError()


  def setUp(self):
    """Initializes test objects."""
    super().setUp()
    self.bitstring_1 = tf.constant([1, 1, 0, 1, 0], dtype=tf.int8)
    self.bitstring_2 = tf.constant([0, 0, 0, 1, 1], dtype=tf.int8)
    self.p_1 = 0.1
    self.e_infer = self.TwoOutcomes(self.bitstring_1, self.bitstring_2,
                                    self.p_1)
    self.energy = self.NullEnergy(list(range(5)))
    self.e_infer.update_energy(self.energy)
    spins_from_bitstrings = energy_model_utils.SpinsFromBitstrings()
    parity = energy_model_utils.Parity(list(range(5)), 2)

    def test_function(bitstrings):
      """Simple test function to send to expectation."""
      return parity(spins_from_bitstrings(bitstrings))

    self.test_function = test_function

  @test_util.eager_mode_toggle
  def test_expectation(self):
    """Confirms correct averaging over input function."""
    values = []
    for b in [
        tf.expand_dims(self.bitstring_1, 0),
        tf.expand_dims(self.bitstring_2, 0)
    ]:
      values.append(self.test_function(tf.constant(b))[0])
    expected_expectation = self.p_1 * values[0] + (1 - self.p_1) * values[1]

    num_samples = int(1e6)

    @tf.function
    def expectation_wrapper(function, num_samples):
      return self.e_infer.expectation(function, num_samples)

    actual_expectation = expectation_wrapper(self.test_function, num_samples)

    self.assertAllClose(actual_expectation, expected_expectation)


class AnalyticEnergyInferenceTest(tf.test.TestCase):
  """Tests the AnalyticEnergyInference class."""

  def setUp(self):
    """Initializes test objects."""
    super().setUp()
    self.tf_random_seed = 4
    self.tfp_seed = tf.constant([3, 4], tf.int32)
    self.grad_close_rtol = 1e-2
    self.not_zero_atol = 1e-1

  def test_init(self):
    """Confirms internal values are set correctly."""
    bits = [0, 1, 3]
    order = 2
    expected_name = "test_analytic_dist_name"
    actual_layer = energy_infer.AnalyticEnergyInference(
        len(bits), expected_name)
    self.assertEqual(actual_layer.name, expected_name)

    energy = energy_model.KOBE(bits, order)
    expected_bitstrings = tf.constant(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1],
         [1, 1, 0], [1, 1, 1]],
        dtype=tf.int8)
    expected_energies = energy(expected_bitstrings)
    actual_layer.update_energy(energy)
    self.assertAllEqual(actual_layer.all_bitstrings, expected_bitstrings)
    self.assertAllClose(actual_layer.all_energies, expected_energies)

  @test_util.eager_mode_toggle
  def test_sample(self):
    """Confirms bitstrings are sampled as expected."""
    n_samples = 1e7
    seed = tf.constant([1, 2], tf.int32)

    # Single bit test.
    one_bit_energy = energy_model.KOBE([0], 1)
    one_bit_energy.build([None, one_bit_energy.num_bits])
    actual_layer = energy_infer.AnalyticEnergyInference(1, seed=seed)

    @tf.function
    def sample_wrapper(num_samples):
      return actual_layer.sample(num_samples)

    # For single factor Bernoulli, theta=0 is 50% chance of 1.
    one_bit_energy.set_weights([tf.constant([0.0])])

    # TODO(#115)
    actual_layer.update_energy(one_bit_energy)
    samples = sample_wrapper(n_samples)
    # check that we got both bitstrings
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0], dtype=tf.int8), samples))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1], dtype=tf.int8), samples))
    # Check that the fraction is approximately 0.5 (equal counts)
    _, counts = utils.unique_bitstrings_with_counts(samples)
    self.assertAllClose(1.0, counts[0] / counts[1], atol=1e-3)

    # Large energy penalty pins the bit.
    one_bit_energy.set_weights([tf.constant([100.0])])
    actual_layer.update_energy(one_bit_energy)

    # TODO(#115): Currently need to redefine wrapper,
    #             investigate resolving this with auto inference.
    @tf.function
    def sample_wrapper_2(num_samples):
      return actual_layer.sample(num_samples)

    samples = sample_wrapper_2(n_samples)
    # check that we got only one bitstring
    self.assertFalse(
        test_util.check_bitstring_exists(
            tf.constant([0], dtype=tf.int8), samples))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1], dtype=tf.int8), samples))

    # Three bit tests.
    # First a uniform sampling test.
    three_bit_energy = energy_model.KOBE([0, 1, 2], 3,
                                         tf.keras.initializers.Constant(0.0))
    actual_layer = energy_infer.AnalyticEnergyInference(3, seed=seed)
    actual_layer.update_energy(three_bit_energy)

    @tf.function
    def sample_wrapper_3(num_samples):
      return actual_layer.sample(num_samples)

    samples = sample_wrapper_3(n_samples)

    for b in [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1],
              [1, 1, 0], [1, 1, 1]]:
      b_tf = tf.constant([b], dtype=tf.int8)
      self.assertTrue(test_util.check_bitstring_exists(b_tf, samples))

    _, counts = utils.unique_bitstrings_with_counts(samples)
    # Check that the fraction is approximately 0.125 (equal counts)
    self.assertAllClose(
        [0.125] * 8,
        tf.cast(counts, tf.float32) / tf.cast(n_samples, tf.float32),
        atol=1e-3,
    )

    # Confirm correlated spins.
    three_bit_energy.set_weights(
        [tf.constant([100.0, 0.0, 0.0, -100.0, 0.0, 100.0, 0.0])])
    actual_layer.update_energy(three_bit_energy)

    # TODO(#115): Currently need to redefine wrapper,
    #             investigate resolving this with auto inference.
    @tf.function
    def sample_wrapper_4(num_samples):
      return actual_layer.sample(num_samples)

    samples = sample_wrapper_4(n_samples)
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
    seed = tf.constant([1, 2], tf.int32)  # seed in TFP style
    num_samples = 1e6
    energy = energy_model.KOBE(list(range(num_bits)), 2)
    energy.build([None, num_bits])
    actual_layer = energy_infer.AnalyticEnergyInference(num_bits, seed=seed)
    actual_layer.update_energy(energy)

    @tf.function
    def sample_wrapper(n_samples):
      return actual_layer.sample(n_samples)

    samples_1 = sample_wrapper(num_samples)
    samples_2 = sample_wrapper(num_samples)
    self.assertAllEqual(samples_1, samples_2)

    # check unseeding lets samples be different again
    actual_layer.seed = None
    samples_1 = actual_layer.sample(num_samples)
    samples_2 = actual_layer.sample(num_samples)
    self.assertNotAllEqual(samples_1, samples_2)

    # TODO(#115): Currently need to redefine wrapper,
    #             investigate resolving this with auto inference.
    @tf.function
    def sample_wrapper_2(num_samples):
      return actual_layer.sample(num_samples)

    samples_1 = sample_wrapper_2(num_samples)
    samples_2 = sample_wrapper_2(num_samples)
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
    energy = energy_model.BitstringEnergy(list(range(num_bits)), energy_layers)
    theta_exp = tf.math.exp(-1.0 * theta)
    partition = tf.math.pow(2.0, num_bits) - 1 + theta_exp
    partition_inverse = tf.math.pow(partition, -1)
    prob_x_star = partition_inverse * theta_exp

    seed = tf.constant([3, 4], tf.int32)
    e_infer = energy_infer.AnalyticEnergyInference(num_bits, seed=seed)
    e_infer.update_energy(energy)

    @tf.function
    def expectation_wrapper(function, n_samples):
      return e_infer.expectation(function, n_samples)

    mu = tf.Variable(tf.random.uniform([], 1, 2), name="mu")
    f = AllOnes(mu)
    expected_average = mu * prob_x_star
    expected_gradient_theta = mu * prob_x_star * (prob_x_star - 1)
    expected_gradient_mu = prob_x_star

    num_samples = int(5e6)
    with tf.GradientTape() as tape:
      actual_average = expectation_wrapper(f, num_samples)
    actual_gradient_theta, actual_gradient_mu = tape.gradient(actual_average, (theta, mu))

    # Confirm gradients are not negligible
    not_negligible_atol = 1e-1
    self.assertAllGreater(
        tf.math.abs(actual_gradient_theta), not_negligible_atol)
    self.assertAllGreater(tf.math.abs(actual_gradient_mu), not_negligible_atol)

    closeness_atol = 1e-3
    self.assertAllClose(actual_average, expected_average, atol=closeness_atol)
    self.assertAllClose(
        actual_gradient_theta, expected_gradient_theta, atol=closeness_atol)
    self.assertAllClose(
        actual_gradient_mu, expected_gradient_mu, atol=closeness_atol)

    # Confirm gradients are connected upstream
    mul_const = tf.random.uniform([], -2, -1)

    def wrap_f(bitstrings):
      return mul_const * f(bitstrings)

    mul_expected_average = mul_const * expected_average
    mul_expected_gradient_theta = mul_const * expected_gradient_theta
    mul_expected_gradient_mu = mul_const * expected_gradient_mu

    with tf.GradientTape() as tape:
      actual_average = expectation_wrapper(wrap_f, num_samples)
    actual_gradient_theta, actual_gradient_mu = tape.gradient(actual_average, (theta, mu))

    self.assertAllClose(
        actual_average, mul_expected_average, atol=closeness_atol)
    self.assertAllClose(
        actual_gradient_theta, mul_expected_gradient_theta, atol=closeness_atol)
    self.assertAllClose(
        actual_gradient_mu, mul_expected_gradient_mu, atol=closeness_atol)

    # Test a function sharing variables with the energy.
    g = AllOnes(theta)
    expected_average = theta * prob_x_star
    expected_gradient_theta = theta * prob_x_star * (prob_x_star -
                                                     1) + prob_x_star

    with tf.GradientTape() as tape:
      actual_average = expectation_wrapper(g, num_samples)
    actual_gradient_theta, actual_gradient_mu = tape.gradient(actual_average, (theta, mu))

    # Confirm gradients are not negligible
    self.assertAllGreater(
        tf.math.abs(actual_gradient_theta), not_negligible_atol)

    self.assertAllClose(actual_average, expected_average, atol=closeness_atol)
    self.assertAllClose(
        actual_gradient_theta, expected_gradient_theta, atol=closeness_atol)
    # Check unconnected gradient
    self.assertIsNone(actual_gradient_mu)

    # Check unconnected gradient with zeros tape setting
    with tf.GradientTape() as tape:
      actual_average = expectation_wrapper(g, num_samples)
    actual_gradient_mu = tape.gradient(
        actual_average, mu, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    zeros_atol = 1e-6
    self.assertAllClose(
        actual_gradient_mu, tf.zeros_like(actual_gradient_mu), atol=zeros_atol)

  def test_expectation_finite_difference(self):
    """Tests a function with nested structural output."""

    num_bits = 3
    order = 2
    nonzero_init = tf.keras.initializers.RandomUniform(
        1, 2, seed=self.tf_random_seed)
    energy = energy_model.KOBE(list(range(num_bits)), order, nonzero_init)
    energy.build([None, num_bits])
    # Only one trainable variable in KOBE.
    energy_var = energy.trainable_variables[0]

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

    e_infer = energy_infer.AnalyticEnergyInference(num_bits, seed=self.tfp_seed)
    e_infer.update_energy(energy)

    num_samples = int(2e6)
    with tf.GradientTape() as tape:
      actual_expectation = e_infer.expectation(f, num_samples)
    actual_derivative = tape.gradient(actual_expectation, energy_var)

    # Trainable variable of KOBE is 1D.
    num_elts = tf.size(energy_var)

    def delta_expectation(k, delta):
      """Calculate the expectation with kth variable perturbed."""
      old_value = energy_var.read_value()
      energy_var.assign(old_value + delta * tf.one_hot(k, num_elts, 1.0, 0.0))
      e_infer.update_energy(energy)
      samples = e_infer.sample(num_samples)
      bitstrings, counts = utils.unique_bitstrings_with_counts(samples)
      values = f(bitstrings)
      delta_expectation = tf.nest.map_structure(
          lambda x: utils.weighted_average(counts, x), values)
      energy_var.assign(old_value)
      return delta_expectation

    expected_expectation = delta_expectation(0, 0)
    tf.nest.map_structure(
        lambda x: self.assertAllGreater(tf.abs(x), self.not_zero_atol),
        expected_expectation)
    self.assertAllClose(actual_expectation, expected_expectation)

    derivative_list = []
    for n in range(num_elts):
      this_derivative = test_util.approximate_derivative(
          functools.partial(delta_expectation, n))
      derivative_list.append(this_derivative.numpy())
    expected_derivative = tf.constant(derivative_list)
    tf.nest.map_structure(
        lambda x: self.assertAllGreater(tf.abs(x), self.not_zero_atol),
        expected_derivative)
    self.assertAllClose(
        actual_derivative, expected_derivative, rtol=self.grad_close_rtol)

  @test_util.eager_mode_toggle
  def test_log_partition(self):
    """Confirms correct value of the log partition function."""
    test_thetas = tf.constant([1.5, 2.7, -4.0])
    expected_log_partition = tf.math.log(tf.constant(3641.8353))

    energy = energy_model.KOBE([0, 1], 2)
    energy.build([None, energy.num_bits])
    actual_layer = energy_infer.AnalyticEnergyInference(2)
    energy.set_weights([test_thetas])
    actual_layer.update_energy(energy)

    @tf.function
    def log_partition_wrapper(layer):
      return layer.log_partition()

    actual_log_partition = log_partition_wrapper(actual_layer)
    self.assertAllClose(actual_log_partition, expected_log_partition)

  @test_util.eager_mode_toggle
  def test_entropy(self):
    """Confirms correct value of the entropy function."""
    test_thetas = tf.constant([1.5, 2.7, -4.0])
    expected_entropy = tf.constant(0.00233551808)

    energy = energy_model.KOBE([0, 1], 2)
    energy.build([None, energy.num_bits])
    actual_layer = energy_infer.AnalyticEnergyInference(2)
    energy.set_weights([test_thetas])
    actual_layer.update_energy(energy)

    @tf.function
    def entropy_wrapper(layer):
      return layer.entropy()

    actual_entropy = entropy_wrapper(actual_layer)
    self.assertAllClose(actual_entropy, expected_entropy)

  @test_util.eager_mode_toggle
  def test_call(self):
    """Confirms that call behaves correctly."""
    seed = tf.constant([1, 2], tf.int32)
    one_bit_energy = energy_model.KOBE([0], 1,
                                       tf.keras.initializers.Constant(0.0))
    actual_layer = energy_infer.AnalyticEnergyInference(1, seed=seed)
    with self.assertRaisesRegex(
        AttributeError, expected_regex="_distribution"):
      _ = actual_layer(None)
    actual_layer.update_energy(one_bit_energy)
    actual_dist = actual_layer(None)
    self.assertIsInstance(actual_dist, tfp.distributions.Categorical)

    n_samples = 1e7

    @tf.function
    def wrapper(n_samples):
      return actual_layer(n_samples)

    samples = wrapper(n_samples)
    # check that we got both bitstrings
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0], dtype=tf.int8), samples))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1], dtype=tf.int8), samples))
    # Check that the fraction is approximately 0.5 (equal counts)
    _, counts = utils.unique_bitstrings_with_counts(samples)
    self.assertAllClose(1.0, counts[0] / counts[1], atol=1e-3)


class BernoulliEnergyInferenceTest(tf.test.TestCase):
  """Tests the BernoulliEnergyInference class."""

  def test_init(self):
    """Tests that components are initialized correctly."""
    expected_name = "test_analytic_dist_name"
    actual_layer = energy_infer.BernoulliEnergyInference(expected_name)
    self.assertEqual(actual_layer.name, expected_name)

  @test_util.eager_mode_toggle
  def test_sample(self):
    """Confirms that bitstrings are sampled as expected."""
    n_samples = 1e7
    seed = tf.constant([1, 2], tf.int32)
    energy = energy_model.BernoulliEnergy([1])
    energy.build([None, energy.num_bits])
    actual_layer = energy_infer.BernoulliEnergyInference(seed=seed)

    # For single factor Bernoulli, theta = 0 is 50% chance of 1.
    energy.set_weights([tf.constant([0.0])])
    actual_layer.update_energy(energy)

    @tf.function
    def sample_wrapper(num_samples):
      return actual_layer.sample(num_samples)

    samples = sample_wrapper(n_samples)
    # check that we got both bitstrings
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0], dtype=tf.int8), samples))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1], dtype=tf.int8), samples))
    # Check that the fraction is approximately 0.5 (equal counts)
    _, counts = utils.unique_bitstrings_with_counts(samples)
    self.assertAllClose(1.0, counts[0] / counts[1], atol=1e-3)

    # Large value of theta pins the bit.
    energy.set_weights([tf.constant([1000.0])])
    actual_layer.update_energy(energy)

    # TODO(#115): Currently need to redefine wrapper,
    #             investigate resolving this with auto inference.
    @tf.function
    def sample_wrapper_2(num_samples):
      return actual_layer.sample(num_samples)

    samples = sample_wrapper_2(n_samples)
    # check that we got only one bitstring
    bitstrings, _ = utils.unique_bitstrings_with_counts(samples)
    self.assertAllEqual(bitstrings, [[1]])

    # Two bit tests.
    energy = energy_model.BernoulliEnergy([0, 1],
                                          tf.keras.initializers.Constant(0.0))
    energy.build([None, energy.num_bits])
    actual_layer = energy_infer.BernoulliEnergyInference(seed=seed)
    actual_layer.update_energy(energy)

    @tf.function
    def sample_wrapper_3(num_samples):
      return actual_layer.sample(num_samples)

    samples = sample_wrapper_3(n_samples)
    for b in [[0, 0], [0, 1], [1, 0], [1, 1]]:
      b_tf = tf.constant([b], dtype=tf.int8)
      self.assertTrue(test_util.check_bitstring_exists(b_tf, samples))
    # Check that the fraction is approximately 0.25 (equal counts)
    _, counts = utils.unique_bitstrings_with_counts(samples)
    self.assertAllClose(
        [0.25] * 4,
        tf.cast(counts, tf.float32) / tf.cast(n_samples, tf.float32),
        atol=1e-3,
    )

    # Test one pinned, one free bit
    energy.set_weights([tf.constant([-1000.0, 0.0])])
    actual_layer.update_energy(energy)

    # TODO(#115): Currently need to redefine wrapper,
    #             investigate resolving this with auto inference.
    @tf.function
    def sample_wrapper_4(num_samples):
      return actual_layer.sample(num_samples)

    samples = sample_wrapper_4(n_samples)
    # check that we get 00 and 01.
    for b in [[0, 0], [0, 1]]:
      b_tf = tf.constant([b], dtype=tf.int8)
      self.assertTrue(test_util.check_bitstring_exists(b_tf, samples))
    for b in [[1, 0], [1, 1]]:
      b_tf = tf.constant([b], dtype=tf.int8)
      self.assertFalse(test_util.check_bitstring_exists(b_tf, samples))
    _, counts = utils.unique_bitstrings_with_counts(samples)
    self.assertAllClose(counts, [n_samples / 2] * 2, atol=n_samples / 1000)

  @test_util.eager_mode_toggle
  def test_samples_seeded(self):
    """Confirm seeding fixes samples for given energy."""
    num_bits = 5
    seed = tf.constant([1, 2], tf.int32)  # seed in TFP style
    num_samples = 1e6
    energy = energy_model.BernoulliEnergy(list(range(num_bits)))
    energy.build([None, num_bits])
    actual_layer = energy_infer.BernoulliEnergyInference(seed=seed)
    actual_layer.update_energy(energy)

    @tf.function
    def sample_wrapper(num_samples):
      return actual_layer.sample(num_samples)

    samples_1 = sample_wrapper(num_samples)
    samples_2 = sample_wrapper(num_samples)
    self.assertAllEqual(samples_1, samples_2)

    # check unseeding lets samples be different again
    actual_layer.seed = None

    # TODO(#115): Currently need to redefine wrapper,
    #             investigate resolving this with auto inference.
    @tf.function
    def sample_wrapper_2(num_samples):
      return actual_layer.sample(num_samples)

    samples_1 = sample_wrapper_2(num_samples)
    samples_2 = sample_wrapper_2(num_samples)
    self.assertNotAllEqual(samples_1, samples_2)

  @test_util.eager_mode_toggle
  def test_log_partition(self):
    """Confirms correct value of the log partition function."""
    all_bitstrings = tf.constant([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                                  [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                                 dtype=tf.int8)
    energy = energy_model.BernoulliEnergy([5, 6, 7])
    energy.build([None, energy.num_bits])
    actual_layer = energy_infer.BernoulliEnergyInference()
    actual_layer.update_energy(energy)
    expected_log_partition = tf.reduce_logsumexp(-1.0 * energy(all_bitstrings))

    @tf.function
    def log_partition_wrapper(layer):
      return layer.log_partition()

    actual_log_partition = log_partition_wrapper(actual_layer)
    self.assertAllClose(actual_log_partition, expected_log_partition)

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

    energy = energy_model.BernoulliEnergy([0, 1, 2])
    energy.build([None, energy.num_bits])
    actual_layer = energy_infer.BernoulliEnergyInference()
    energy.set_weights([test_thetas])
    actual_layer.update_energy(energy)

    @tf.function
    def entropy_wrapper(layer):
      return layer.entropy()

    actual_entropy = entropy_wrapper(actual_layer)
    self.assertAllClose(actual_entropy, expected_entropy)

  @test_util.eager_mode_toggle
  def test_call(self):
    """Confirms that calling the layer works correctly."""
    seed = tf.constant([1, 2], tf.int32)
    energy = energy_model.BernoulliEnergy([1],
                                          tf.keras.initializers.Constant(0.0))
    energy.build([None, energy.num_bits])
    actual_layer = energy_infer.BernoulliEnergyInference(seed=seed)
    with self.assertRaisesRegex(
        AttributeError, expected_regex="_distribution"):
      _ = actual_layer(None)
    actual_layer.update_energy(energy)
    actual_dist = actual_layer(None)
    self.assertIsInstance(actual_dist, tfp.distributions.Bernoulli)

    # For single factor Bernoulli, theta = 0 is 50% chance of 1.
    n_samples = 1e7

    @tf.function
    def wrapper(n_samples):
      return actual_layer(n_samples)

    samples = actual_layer(n_samples)
    # check that we got both bitstrings
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([0], dtype=tf.int8), samples))
    self.assertTrue(
        test_util.check_bitstring_exists(
            tf.constant([1], dtype=tf.int8), samples))
    # Check that the fraction is approximately 0.5 (equal counts)
    _, counts = utils.unique_bitstrings_with_counts(samples)
    self.assertAllClose(1.0, counts[0] / counts[1], atol=1e-3)


if __name__ == "__main__":
  print("Running energy_infer_test.py ...")
  tf.test.main()
