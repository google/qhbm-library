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

from qhbm_library import ansatze
from qhbm_library import density_operator
from qhbm_library import ebm
from qhbm_library import qhbm
from qhbm_library import qmhl


class QMHLLossTest(tf.test.TestCase):
    """Tests for the QMHL loss and gradients."""

    def test_zero_grad(self):
        """Confirm correct gradients and loss at the optimal settings."""
        for num_qubits in [1, 2, 3, 4, 5, 6]:
            qubits = cirq.GridQubit.rect(1, num_qubits)
            target_qhbm = qhbm.get_random_qhbm(
                qubits, ebm.build_boltzmann, 1, "QMHLLossTest{}".format(num_qubits)
            )
            model_qhbm = target_qhbm.copy()
            self.assertAllClose(model_qhbm.thetas, target_qhbm.thetas, atol=1e-6)
            self.assertAllClose(model_qhbm.phis, target_qhbm.phis, atol=1e-6)
            shape_thetas = target_qhbm.thetas.shape
            shape_phis = target_qhbm.phis.shape

            # Get the QMHL loss gradients
            model_samples = 2e5
            target_samples = 2e5
            target_circuits, _, target_counts = qhbm.sample_state_circuits(
                target_qhbm.sampler_function,
                target_qhbm.thetas,
                target_qhbm.bit_and_u,
                target_qhbm.bit_symbols,
                target_qhbm.phis_symbols,
                target_qhbm.phis,
                target_samples,
            )
            target_density = density_operator.DensityOperator(
                target_circuits, target_counts
            )
            loss = qmhl.qmhl_loss(model_qhbm, target_density)
            thetas_grad = qmhl.qmhl_loss_thetas_grad(
                model_qhbm, model_samples, target_density
            )
            phis_grad = qmhl.qmhl_loss_phis_grad(model_qhbm, target_density, eps=0.02)
            print("Current num qubits: {}".format(num_qubits))
            self.assertAllClose(
                loss, target_qhbm.entropy_function(target_qhbm.thetas), atol=1e-2
            )
            self.assertAllClose(thetas_grad, tf.zeros(shape_thetas), atol=1e-2)
            self.assertAllClose(phis_grad, tf.zeros(shape_phis), atol=5e-2)


if __name__ == "__main__":
    print("Running qmhl_test.py ...")
    tf.test.main()
