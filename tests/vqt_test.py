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
"""Tests for the VQT loss and gradients."""

import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib import qhbm_base
from qhbmlib import vqt
from qhbmlib.tests import test_util


class VQTTest(tf.test.TestCase):
  """Tests for the sample-based VQT."""

  num_bits = 5
  initial_thetas = tf.random.uniform([num_bits], minval=-1.0)
  raw_phis_symbols = [sympy.Symbol("s0"), sympy.Symbol("s1")]
  phis_symbols = tf.constant([str(s) for s in raw_phis_symbols])
  initial_phis = tf.random.uniform([len(phis_symbols)], minval=-1.0)
  raw_qubits = cirq.GridQubit.rect(1, num_bits)
  u = cirq.Circuit()
  for s in raw_phis_symbols:
    for q in raw_qubits:
      u += cirq.X(q)**s
  name = "TestQHBM"

  def test_loss_consistency(self):
    """Confirms that the sample-based and exact losses are close."""
    energy, sampler = test_util.get_ebm_functions(self.num_bits)
    test_qhbm = qhbm_base.ExactQHBM(
        self.initial_thetas,
        energy,
        sampler,
        self.initial_phis,
        self.raw_phis_symbols,
        self.u,
        self.name,
    )
    num_samples = tf.constant(int(5e6))
    num_random_hamiltonians = 2
    for beta in tf.constant([0.1, 0.4, 1.6, 6.4]):
      for _ in range(num_random_hamiltonians):
        cirq_ham = test_util.get_random_pauli_sum(self.raw_qubits)
        tf_ham = tfq.convert_to_tensor([[cirq_ham]])
        loss_estimate = vqt.vqt_loss(test_qhbm, num_samples, beta, tf_ham)
        loss_exact = vqt.exact_vqt_loss(test_qhbm, num_samples, beta, tf_ham)
        self.assertAllClose(loss_estimate, loss_exact, rtol=1e-2)


# TODO(#14): wait for rewrite
# class VQTLossTest(tf.test.TestCase):
#     """Tests for the VQT loss and gradients."""

#     def test_zero_grad(self):
#         """Confirm correct gradients and loss at the optimal settings."""
#         for num_qubits in [1, 2]:
#             qubits = cirq.GridQubit.rect(1, num_qubits)
#             target_qhbm = test_util.get_random_qhbm(
#                 qubits, 1, "VQTLossTest{}".format(num_qubits)
#             )
#             model_qhbm = target_qhbm.copy()
#             shape_thetas = target_qhbm.thetas.shape
#             shape_phis = target_qhbm.phis.shape

#             # Get the QMHL loss gradients
#             model_samples = 2e5
#             target_samples = 2e5
#             sub_term_energy_func = vqt.build_sub_term_energy_func(
#                 model_qhbm, [target_qhbm]
#             )
#             loss = vqt.vqt_qhbm_loss(model_qhbm, model_samples, sub_term_energy_func)
#             thetas_grad = vqt.vqt_qhbm_loss_thetas_grad(
#                 model_qhbm, model_samples, sub_term_energy_func
#             )
#             phis_grad = vqt.vqt_qhbm_loss_phis_grad(
#                 model_qhbm, model_samples, sub_term_energy_func, eps=0.02
#             )
#             print("Current num qubits: {}".format(num_qubits))
#             self.assertAllClose(
#               loss, -target_qhbm.log_partition_function(target_qhbm.thetas), atol=1e-2
#             )
#             self.assertAllClose(thetas_grad, tf.zeros(shape_thetas), atol=1e-2)
#             self.assertAllClose(phis_grad, tf.zeros(shape_phis), atol=1e-1)

#     def train_independent_multi_input(self):
#         """Confirm VQT successfully thermalizes independent multi-term inputs."""
#         # Build the target QHBM terms
#         qubits = cirq.GridQubit.rect(1, 3)
#         qubits_1 = qubits[:1]
#         qubits_2 = qubits[1:]
#         qhbm_1 = test_util.get_random_qhbm(qubits_1, 1, "qhbm_1")
#         qhbm_2 = test_util.get_random_qhbm(qubits_2, 1, "qhbm_2")
#         target_samples = 3000
#         raw_dm_1 = qhbm_1.density_matrix()
#         raw_dm_2 = qhbm_2.density_matrix()
#         target_dm = tf.linalg.LinearOperatorKronecker([raw_dm_1, raw_dm_2]).to_dense()
#         self.assertAllClose(tf.linalg.trace(target_dm), 1.0, atol=1e-5)
#         # Since the input QHBMs are independent, the optimal loss is the sum of
#         # individual minus log partitions functions.
#         target_loss = -1.0 * tf.reduce_sum(
#             [
#                 qhbm_1.log_partition_function(),
#                 qhbm_2.log_partition_function(),
#             ]
#         )

#         # Set up training loop.
#         model_qhbm = test_util.get_random_qhbm(
#             qubits, 1, "model_qhbm"
#         )
#         schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#             0.25, 10, 0.8, staircase=False, name=None
#         )
#         optimizer = tf.keras.optimizers.SGD(learning_rate=schedule)
#         model_samples = 3000
#         sub_term_energy_func = vqt.build_sub_term_energy_func(
#             model_qhbm, [qhbm_1, qhbm_2]
#         )

#         # Train the model.
#         train_steps = 100
#         for _ in range(train_steps):
#             thetas_gradients = vqt.vqt_qhbm_loss_thetas_grad(
#                 model_qhbm, model_samples, sub_term_energy_func
#             )
#             phis_gradients = vqt.vqt_qhbm_loss_phis_grad(
#                 model_qhbm, model_samples, sub_term_energy_func, eps=0.02
#             )
#             optimizer.apply_gradients(
#                 [
#                     (thetas_gradients, model_qhbm.thetas),
#                     (phis_gradients, model_qhbm.phis),
#                 ]
#             )

#         final_loss = vqt.vqt_qhbm_loss(model_qhbm,model_samples,sub_term_energy_func)
#         final_dm = model_qhbm.density_matrix()
#         final_fidelity = util.fidelity(final_dm, target_dm)
#         # TODO(zaqqwerty): tighten these once better sample gradients are ready
#         self.assertAllClose(final_loss, target_loss, atol=1e-1)
#         self.assertAllClose(final_fidelity, 1.0, atol=5e-2)

#     def test_train_overlapping_multi_input(self):
#         """Confirm VQT successfully thermalizes overlapping multi-term inputs."""
#         # Build the target QHBM terms
#         qubits = cirq.GridQubit.rect(1, 3)
#         qubits_2_0 = qubits[:2]
#         qubits_2_1 = qubits[1:]
#         qhbm_2_0 = test_util.get_random_qhbm(
#             qubits_2_0, 1, "qhbm_2_0"
#         )
#         qhbm_2_1 = test_util.get_random_qhbm(
#             qubits_2_1, 1, "qhbm_2_1"
#         )
#         target_samples = 3000
#         raw_dm_2_0 = qhbm_2_0.density_matrix()
#         raw_dm_2_1 = qhbm_2_1.density_matrix()
#         ident_1 = tf.linalg.LinearOperatorFullMatrix(
#             tf.divide(
#                 tf.cast(tf.linalg.diag(tf.ones([2])), tf.complex64),
#                 tf.cast(2, tf.complex64),
#             )
#         )
#         dm_2_0 = tf.linalg.LinearOperatorKronecker([raw_dm_2_0, ident_1]).to_dense()
#         dm_2_1 = tf.linalg.LinearOperatorKronecker([ident_1, raw_dm_2_1]).to_dense()
#         raw_exp_dm = tf.linalg.expm(tf.linalg.logm(dm_2_0) + tf.linalg.logm(dm_2_1))
#         target_dm = raw_exp_dm / tf.linalg.trace(raw_exp_dm)
#         self.assertAllClose(tf.linalg.trace(raw_dm_2_0), 1.0, atol=1e-5)
#         self.assertAllClose(tf.linalg.trace(raw_dm_2_1), 1.0, atol=1e-5)
#         self.assertAllClose(tf.linalg.trace(dm_2_0), 1.0, atol=1e-5)
#         self.assertAllClose(tf.linalg.trace(dm_2_1), 1.0, atol=1e-5)
#         self.assertAllClose(tf.linalg.trace(target_dm), 1.0, atol=1e-5)

#         # Set up training loop.
#         model_qhbm = test_util.get_random_qhbm(
#             qubits, 2, "model_qhbm"
#         )
#         schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#             0.25, 10, 0.8, staircase=False, name=None
#         )
#         optimizer = tf.keras.optimizers.SGD(learning_rate=schedule)
#         model_samples = 3000
#         sub_term_energy_func = vqt.build_sub_term_energy_func(
#             model_qhbm, [qhbm_2_0, qhbm_2_1]
#         )

#         # Train the model.
#         train_steps = 100
#         for _ in range(train_steps):
#             thetas_gradients = vqt.vqt_qhbm_loss_thetas_grad(
#                 model_qhbm, model_samples, sub_term_energy_func
#             )
#             phis_gradients = vqt.vqt_qhbm_loss_phis_grad(
#                 model_qhbm, model_samples, sub_term_energy_func, eps=0.02
#             )
#             optimizer.apply_gradients(
#                 [
#                     (thetas_gradients, model_qhbm.thetas),
#                     (phis_gradients, model_qhbm.phis),
#                 ]
#             )

#         final_dm = model_qhbm.density_matrix()
#         final_fidelity = util.fidelity(final_dm, target_dm)
#         # TODO(zaqqwerty): tighten these once better sample gradients are ready
#         self.assertAllClose(final_fidelity, 1.0, atol=1e-1)

if __name__ == "__main__":
  print("Running vqt_test.py ...")
  tf.test.main()
