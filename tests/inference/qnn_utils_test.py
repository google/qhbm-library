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
"""Tests for qhbmlib.inference.qnn_utils"""

from absl import logging
import random
import string

import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
from tensorflow_quantum.python import util as tfq_util

from qhbmlib import inference
from qhbmlib import models


class UnitaryTest(tf.test.TestCase):
  """Tests the unitary function."""

  def setUp(self):
    """Initializes test objects."""
    super().setUp()

    self.num_bits = 4
    self.tf_random_seed = 7
    self.close_rtol = 1e-4

  def test_unitary(self):
    """Confirms unitary is correct for a random circuit."""
    # TODO(#171) this random circuit construction happens in a few places.
    qubits = cirq.GridQubit.rect(1, self.num_bits)
    batch_size = 1
    n_moments = 10
    act_fraction = 0.9
    num_symbols = 2
    symbols = set()
    for _ in range(num_symbols):
      symbols.add("".join(random.sample(string.ascii_letters, 10)))
    symbols = sorted(list(symbols))
    raw_circuits, _ = tfq_util.random_symbol_circuit_resolver_batch(
        qubits, symbols, batch_size, n_moments=n_moments, p=act_fraction)
    raw_circuit = raw_circuits[0]
    random_values = tf.random.uniform([len(symbols)], -1, 1, tf.float32,
                                      self.tf_random_seed).numpy().tolist()
    resolver = dict(zip(symbols, random_values))

    actual_circuit = models.QuantumCircuit(
        tfq.convert_to_tensor([raw_circuit]), qubits, tf.constant(symbols),
        [tf.Variable([resolver[s] for s in symbols])], [[]])

    resolved_circuit = cirq.protocols.resolve_parameters(raw_circuit, resolver)
    expected_unitary = resolved_circuit.unitary()

    unitary_wrapper = tf.function(inference.unitary)
    actual_unitary = unitary_wrapper(actual_circuit)
    self.assertAllClose(actual_unitary, expected_unitary, rtol=self.close_rtol)


if __name__ == "__main__":
  logging.info("Running qnn_utils_test.py ...")
  tf.test.main()
