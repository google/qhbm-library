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
"""Tests for the QMHL loss and gradients."""

import cirq
from qhbmlib import qmhl
import tensorflow as tf
from tests import test_util

class QMHLExactLossTest(tf.test.TestCase):
  """Tests for the QMHL loss and gradients."""

  def test_zero_grad(self):
    """Confirm correct gradients and loss at the optimal settings."""
    for num_qubits in [1, 2, 3, 4, 5]:
      qubits = cirq.GridQubit.rect(1, num_qubits)
      target = test_util.get_random_qhbm(
          qubits, 1, "QMHLLossTest{}".format(num_qubits))
      model = target.copy()

      # Get the QMHL loss gradients
      model_samples = tf.constant(1e6)
      target_samples = tf.constant(1e6)
      target_circuits, target_counts = target.circuits(target_samples)
      with tf.GradientTape() as tape:
        loss = qmhl.qmhl_loss(model, target_circuits, target_counts)
      thetas_grads, phis_grads = tape.gradient(loss, (model.thetas, model.phis))
      self.assertAllClose(loss, target.ebm.entropy(), atol=2e-3)
      self.assertAllClose(thetas_grads, tf.zeros(tf.shape(thetas_grads)), atol=1e-3)
      self.assertAllClose(phis_grads, tf.zeros(tf.shape(phis_grads)), atol=1e-3)

if __name__ == "__main__":
  print("Running qmhl_test.py ...")
  tf.test.main()
