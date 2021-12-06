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
"""Tools for inference on quantum circuits."""

import tensorflow as tf
import tensorflow_quantum as tfq

class Expectation(tf.keras.Model):
  """Class for taking expectation values of quantum circuits."""
  
  def __init__(self,
               backend="noiseless",
               differentiator=None,
               name=None):
    """Initialize an instance of Expectation.

    Args:
      backend: Optional Python `object` that specifies what backend TFQ will use
        to compute expectation values. Options are {"noisy", "noiseless"}.
        Users may also specify a preconfigured cirq execution
        object to use instead, which must inherit `cirq.Sampler`.
      differentiator: Either None or a `tfq.differentiators.Differentiator`,
        which specifies how to take the derivative of a quantum circuit.
      name: Identifier for this inference engine.
    """
    super().__init__(name=name)

    self._differentiator = differentiator
    if backend == "noiseless" or backend is None:
      self._backend = "noiseless"
      self._expectation_layer = tfq.layers.Expectation(
          backend=backend, differentiator=differentiator)
      def _expectation_function(circuits, symbol_names, symbol_values, operators, _):
        return self._expectation_layer(
          circuits,
          symbol_names=symbol_names,
          symbol_values=symbol_values,
          operators=operators
        )
    else:
      self._backend = backend
      self._expectation_layer = tfq.layers.SampledExpectation(
          backend=backend, differentiator=differentiator)
      def _expectation_function(circuits, symbol_names, symbol_values, operators, repetitions):
        return self._expectation_layer(
          circuits,
          symbol_names=symbol_names,
          symbol_values=symbol_values,
          operators=operators
          repetitions=repetitions,
        )
    self._expectation_function = _expectation_function

  @property
  def backend(self):
    return self._backend

  @property
  def differentiator(self):
    return self._differentiator

  def expectation(self,
                  counts,
                  circuits,
                  symbol_names,
                  symbol_values,
                  operators,
                  reduce=True):
    """Returns the expectation values of the operators against the QNN.

      Args:
        counts: 1D tensor of dtype `tf.int32` such that `counts[i]` is the
          relative weight of `circuits[i]` when computing expectations.
          Additionally, if `self.backend != "noiseless", `counts[i]` samples
          are drawn from `circuits[i]` and used to compute each expectation.
        circuits: `tf.Tensor` of strings with shape [batch_size] containing
            the string representations of the circuits to be executed.
        symbol_names: `tf.Tensor` of strings with shape [n_params], which
            is used to specify the order in which the values in
            `symbol_values` should be placed inside of the circuits in
            `programs`.
        symbol_values: `tf.Tensor` of real numbers with shape
            [batch_size, n_params] specifying parameter values to resolve
            into the circuits specified by programs, following the ordering
            dictated by `symbol_names`.
        operators: `tf.Tensor` of strings with shape [n_ops], result of calling
          `tfq.convert_to_tensor` on a list of cirq.PauliSum, `[op1, op2, ...]`.
          Will be tiled to measure `<op_j>_self.u_dagger|circuits[i]>`
          for each i and j.
        reduce: bool flag for whether or not to average over i.

      Returns:
        If `reduce` is true, a `tf.Tensor` with shape [n_ops] whose entries are
        are the batch-averaged expectation values of `operators`.
        Else, a `tf.Tensor` with shape [batch_size, n_ops] whose entries are the
        unaveraged expectation values of each `operator` against each `circuit`.
      """
    num_circuits = tf.shape(circuits)[0]
    num_operators = tf.shape(operators)[0]
    tiled_values = tf.tile(tf.expand_dims(symbol_values, 0), [num_circuits, 1])
    tiled_operators = tf.tile(tf.expand_dims(operators, 0), [num_circuits, 1])
    expectations = self._expectation_function(
          circuits,
          symbol_names=symbol_names,
          symbol_values=tiled_values,
          operators=tiled_operators,
          repetitions=tf.tile(tf.expand_dims(counts, 1), [1, num_operators]),
    )
    if reduce:
      probs = tf.cast(counts, tf.float32) / tf.cast(
          tf.reduce_sum(counts), tf.float32)
      return tf.reduce_sum(tf.transpose(probs * tf.transpose(expectations)), 0)
    return expectations


  def call(self,
           counts,
           circuits,
           symbol_names,
           symbol_values,
           operators,
           reduce=True):
    return expectation(counts,
                       circuits,
                       symbol_names,
                       symbol_values,
                       operators,
                       reduce=reduce)
