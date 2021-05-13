# Copyright 2021 The QHBM Library Authors.
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
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

from qhbm_library import ebm
from qhbm_library import qhbm_base
from qhbm_library import qmhl
from tests import test_util


class QMHLExactLossTest(tf.test.TestCase):
    """Tests for the QMHL loss and gradients."""

    def test_zero_grad(self):
        """Confirm correct gradients and loss at the optimal settings."""
        for num_qubits in [1, 2, 3, 4, 5]:
            qubits = cirq.GridQubit.rect(1, num_qubits)
            target_qhbm = test_util.get_random_qhbm(
                qubits, 1, "QMHLLossTest{}".format(num_qubits)
            )
            model_qhbm = target_qhbm.copy()
            shape_thetas = target_qhbm.thetas.shape
            shape_phis = target_qhbm.phis.shape

            # Get the QMHL loss gradients
            model_samples = 1e6
            target_samples = 1e6
            target_circuits, _, target_counts = target_qhbm.sample_state_circuits(
                target_samples
            )
            loss = qmhl.exact_qmhl_loss(model_qhbm, target_circuits, target_counts)
            thetas_grad = qmhl.exact_qmhl_loss_thetas_grad(
                model_qhbm, model_samples, target_circuits, target_counts
            )
            print("Current num qubits: {}".format(num_qubits))
            self.assertAllClose(loss, target_qhbm.entropy_function(), atol=1e-3)
            self.assertAllClose(thetas_grad, tf.zeros(shape_thetas), atol=1e-3)


if __name__ == "__main__":
    print("Running qmhl_test.py ...")
    tf.test.main()
