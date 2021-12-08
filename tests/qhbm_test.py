# # Copyright 2021 The QHBM Library Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     https://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================
# """Tests for the qhbm module."""

# import cirq
# import sympy
# import tensorflow as tf
# import tensorflow_probability as tfp
# import tensorflow_quantum as tfq

# from qhbmlib import ebm
# from qhbmlib import qhbm
# from qhbmlib import qnn
# from qhbmlib import util
# from tests import test_util

# # Global tolerance, set for float32.
# ATOL = 1e-5

# class QHBMTest(tf.test.TestCase):
#   """Tests the base QHBM class."""

#   num_bits = 5
#   raw_phis_symbols = [sympy.Symbol("s0"), sympy.Symbol("s1")]
#   phis_symbols = tf.constant([str(s) for s in raw_phis_symbols])
#   raw_qubits = cirq.GridQubit.rect(1, num_bits)
#   operators = tfq.convert_to_tensor([
#       cirq.PauliSum.from_pauli_strings(s)
#       for s in [cirq.Z(q) for q in raw_qubits]
#   ])
#   u = cirq.Circuit()
#   for s in raw_phis_symbols:
#     for q in raw_qubits:
#       u += cirq.X(q)**s
#   name = "TestQHBM"
#   raw_bit_circuit = qnn.bit_circuit(raw_qubits)
#   raw_bit_symbols = list(sorted(tfq.util.get_circuit_symbols(raw_bit_circuit)))
#   bit_symbols = tf.constant([str(s) for s in raw_bit_symbols])

#   def test_init(self):
#     """Confirms QHBM is initialized correctly."""
#     test_ebm = ebm.Bernoulli(self.num_bits)
#     test_qnn = qnn.QNN(self.u)
#     test_qhbm = qhbm.QHBM(test_ebm, test_qnn, self.name)
#     self.assertEqual(self.name, test_qhbm.name)
#     self.assertEqual(test_ebm, test_qhbm.ebm)
#     self.assertAllEqual(test_qhbm.operator_shards, self.operators)
#     self.assertAllEqual(self.phis_symbols, test_qhbm.qnn.symbols)
#     self.assertAllEqual(
#         tfq.from_tensor(tfq.convert_to_tensor([self.u])),
#         tfq.from_tensor(test_qhbm.qnn.pqc(resolve=False)),
#     )
#     self.assertAllEqual(
#         tfq.from_tensor(tfq.convert_to_tensor([self.u**-1])),
#         tfq.from_tensor(test_qhbm.qnn.inverse_pqc(resolve=False)),
#     )
#     self.assertAllEqual(self.raw_qubits, test_qhbm.raw_qubits)
#     self.assertAllEqual(self.bit_symbols, test_qhbm.qnn._bit_symbols)
#     self.assertEqual(self.raw_bit_circuit,
#                      tfq.from_tensor(test_qhbm.qnn._bit_circuit)[0])

#   def test_copy(self):
#     """Confirms copy works correctly."""
#     test_ebm = ebm.Bernoulli(self.num_bits)
#     test_qnn = qnn.QNN(self.u)
#     test_qhbm = qhbm.QHBM(test_ebm, test_qnn, self.name)
#     qhbm_copy = test_qhbm.copy()
#     self.assertEqual(test_qhbm.name, qhbm_copy.name)
#     self.assertAllClose(test_qhbm.trainable_variables,
#                         qhbm_copy.trainable_variables)
#     self.assertAllEqual(test_qhbm.qnn.symbols, qhbm_copy.qnn.symbols)
#     self.assertAllEqual(
#         tfq.from_tensor(test_qhbm.qnn.pqc(resolve=False)),
#         tfq.from_tensor(qhbm_copy.qnn.pqc(resolve=False)))
#     self.assertAllEqual(
#         tfq.from_tensor(test_qhbm.qnn.inverse_pqc(resolve=False)),
#         tfq.from_tensor(qhbm_copy.qnn.inverse_pqc(resolve=False)))
#     self.assertAllEqual(test_qhbm.raw_qubits, qhbm_copy.raw_qubits)
#     self.assertAllEqual(test_qhbm.qnn._bit_symbols, qhbm_copy.qnn._bit_symbols)
#     self.assertEqual(
#         tfq.from_tensor(test_qhbm.qnn._bit_circuit),
#         tfq.from_tensor(qhbm_copy.qnn._bit_circuit))

# def get_basic_qhbm(is_analytic=False):
#   """Returns a basic QHBM for testing."""
#   num_bits = 3
#   initial_thetas = 0.5 * tf.constant([-23, 0, 17], dtype=tf.float32)
#   test_ebm = ebm.Bernoulli(num_bits, is_analytic=is_analytic)
#   test_ebm.kernel.assign(initial_thetas)
#   initial_phis = tf.constant([1.2, -2.5])
#   phis_symbols = [sympy.Symbol(s) for s in ["s_static_0", "s_static_1"]]
#   u = cirq.Circuit()
#   qubits = cirq.GridQubit.rect(1, num_bits)
#   for s in phis_symbols:
#     for q in qubits:
#       u += cirq.X(q)**s
#   test_qnn = qnn.QNN(u, is_analytic=is_analytic)
#   test_qnn.values.assign(initial_phis)
#   name = "static_qhbm"
#   return qhbm.QHBM(test_ebm, test_qnn, name=name)

# class QHBMBasicFunctionTest(tf.test.TestCase):
#   """Test methods of the QHBM class with a simple QHBM."""

#   def test_sample_bitstrings(self):
#     """Confirm only the middle bit alternates."""
#     num_samples = int(1e6)
#     test_qhbm = get_basic_qhbm()
#     test_bitstrings, test_counts = test_qhbm.ebm.sample(num_samples)
#     self.assertTrue(
#         test_util.check_bitstring_exists(
#             tf.constant([0, 0, 1], dtype=tf.int8), test_bitstrings))
#     self.assertTrue(
#         test_util.check_bitstring_exists(
#             tf.constant([0, 1, 1], dtype=tf.int8), test_bitstrings))
#     # Sanity check that absent bitstring is really missing.
#     self.assertFalse(
#         test_util.check_bitstring_exists(
#             tf.constant([0, 0, 0], dtype=tf.int8), test_bitstrings))
#     # Only the two expected bitstrings should exist.
#     self.assertAllEqual(tf.shape(test_bitstrings), [2, 3])
#     self.assertEqual(tf.reduce_sum(test_counts), num_samples)
#     self.assertAllClose(1.0, test_counts[0] / test_counts[1], atol=2e-3)

#   def test_sample_state_circuits(self):
#     """Confirm circuits are sampled correctly."""
#     num_samples = int(1e6)
#     test_qhbm = get_basic_qhbm()
#     test_circuit_samples, _ = test_qhbm.circuits(num_samples)
#     # Circuits with the allowed-to-be-sampled bitstrings prepended.
#     resolved_u_t = test_qhbm.qnn.pqc(resolve=True)
#     resolved_u = tfq.from_tensor(resolved_u_t)[0]
#     expected_circuit_samples = [
#         cirq.Circuit(
#             cirq.X(test_qhbm.raw_qubits[0])**0,
#             cirq.X(test_qhbm.raw_qubits[1])**0,
#             cirq.X(test_qhbm.raw_qubits[2]),
#         ) + resolved_u,
#         cirq.Circuit(
#             cirq.X(test_qhbm.raw_qubits[0])**0,
#             cirq.X(test_qhbm.raw_qubits[1]),
#             cirq.X(test_qhbm.raw_qubits[2]),
#         ) + resolved_u,
#     ]
#     # Check that both circuits are generated.
#     test_circuit_samples_deser = tfq.from_tensor(test_circuit_samples)
#     self.assertTrue(
#         any([
#             expected_circuit_samples[0] == test_circuit_samples_deser[0],
#             expected_circuit_samples[0] == test_circuit_samples_deser[1],
#         ]))
#     self.assertTrue(
#         any([
#             expected_circuit_samples[1] == test_circuit_samples_deser[0],
#             expected_circuit_samples[1] == test_circuit_samples_deser[1],
#         ]))

#   def test_sample_unresolved_state_circuits(self):
#     """Confirm unresolved circuits are sampled correctly."""
#     # TODO(b/182904206)

#   def test_sample_pulled_back_bitstrings(self):
#     """Ensures pulled back bitstrings are correct."""
#     test_qhbm = get_basic_qhbm()
#     # This setting reduces test_qhbm.u to a bit flip on every qubit.
#     test_qhbm.qnn.values.assign([0.5, 0.5])
#     circuits = tfq.convert_to_tensor([
#         cirq.Circuit(
#             cirq.X(test_qhbm.raw_qubits[0]), cirq.X(test_qhbm.raw_qubits[2])),
#         cirq.Circuit(
#             cirq.X(test_qhbm.raw_qubits[1]), cirq.X(test_qhbm.raw_qubits[2])),
#     ])
#     n_samples_0 = int(1e4)
#     n_samples_1 = int(2e4)
#     counts = tf.constant([n_samples_0, n_samples_1])
#     ragged_samples = test_qhbm.qnn.pulled_back_sample(
#         circuits, counts, reduce=False, unique=False)
#     test_samples_0 = ragged_samples[0].to_tensor()
#     test_samples_1 = ragged_samples[1].to_tensor()
#     self.assertEqual(n_samples_0, test_samples_0.shape[0])
#     self.assertEqual(n_samples_1, test_samples_1.shape[0])
#     uniques_0, _ = util.unique_bitstrings_with_counts(test_samples_0)
#     uniques_1, _ = util.unique_bitstrings_with_counts(test_samples_1)
#     self.assertEqual(1, uniques_0.shape[0])
#     self.assertEqual(1, uniques_1.shape[0])
#     self.assertAllEqual(tf.constant([0, 1, 0], dtype=tf.int8), uniques_0[0])
#     self.assertAllEqual(tf.constant([1, 0, 0], dtype=tf.int8), uniques_1[0])

# def get_exact_qhbm():
#   """Returns a basic ExactQHBM for testing."""
#   return get_basic_qhbm(is_analytic=True)

# class ExactQHBMBasicFunctionTest(tf.test.TestCase):
#   """Test methods of the exact QHBM class with a simple QHBM."""

#   all_bitstrings = tf.constant([
#       [0, 0, 0],
#       [0, 0, 1],
#       [0, 1, 0],
#       [0, 1, 1],
#       [1, 0, 0],
#       [1, 0, 1],
#       [1, 1, 0],
#       [1, 1, 1],
#   ])

#   def test_copy(self):
#     """Confirms copy works correctly."""
#     test_qhbm = get_exact_qhbm()
#     qhbm_copy = test_qhbm.copy()
#     self.assertEqual(test_qhbm.name, qhbm_copy.name)
#     self.assertAllClose(test_qhbm.trainable_variables,
#                         qhbm_copy.trainable_variables)
#     self.assertAllEqual(test_qhbm.qnn.symbols, qhbm_copy.qnn.symbols)
#     self.assertAllEqual(
#         tfq.from_tensor(test_qhbm.qnn.pqc(resolve=False)),
#         tfq.from_tensor(qhbm_copy.qnn.pqc(resolve=False)))
#     self.assertAllEqual(
#         tfq.from_tensor(test_qhbm.qnn.inverse_pqc(resolve=False)),
#         tfq.from_tensor(qhbm_copy.qnn.inverse_pqc(resolve=False)))
#     self.assertAllEqual(test_qhbm.raw_qubits, qhbm_copy.raw_qubits)
#     self.assertAllEqual(test_qhbm.qnn._bit_symbols, qhbm_copy.qnn._bit_symbols)
#     self.assertEqual(
#         tfq.from_tensor(test_qhbm.qnn._bit_circuit),
#         tfq.from_tensor(qhbm_copy.qnn._bit_circuit))

#   def test_all_energies(self):
#     """Confirms that each bitstring energy is correct."""
#     test_qhbm = get_exact_qhbm()
#     test_energies = test_qhbm.ebm.energies()
#     for n_b, b in enumerate(self.all_bitstrings):
#       self.assertAllClose(
#           test_qhbm.ebm.energy(tf.expand_dims(b, 0))[0],
#           test_energies[n_b],
#           atol=ATOL)

#   def test_log_partition_function(self):
#     """Confirms the logarithm of the partition function is correct.

#         The basic energy function used for testing is independent between bits,
#         so that E(b) = sum_i E_i(b[i]).  The partition function for such an E is
#         Z = sum_b e^{-E(b)}
#           = sum_b e^{-sum_i E_i(b[i])}
#           = sum_b prod_i e^{-E_i(b[i])}
#           = prod_i (e^{-E_i(0)) + e^{-E_i(1)})
#         and for the simple energy function, E_i(b[i]) = theta[i] * (1 - 2 *
#         b[i]),
#         Z = prod_i (e^{-theta[i]) + e^{theta[i]})
#         """

#     def base_val(t):
#       return tf.math.exp(-t) + tf.math.exp(t)

#     test_qhbm = get_exact_qhbm()
#     log_partition_expect = tf.math.log(
#         tf.reduce_prod(
#             [base_val(theta) for theta in test_qhbm.ebm.kernel.numpy()]))
#     test_log_partition = test_qhbm.log_partition_function()
#     self.assertAllClose(log_partition_expect, test_log_partition, atol=ATOL)

#   def test_entropy_function(self):
#     """Confirms the entropy of the QHBM is correct."""
#     test_qhbm = get_exact_qhbm()
#     entropy_expect = tf.reduce_sum(
#         tfp.distributions.Bernoulli(logits=2 * test_qhbm.ebm.kernel).entropy())
#     test_entropy = test_qhbm.entropy()
#     self.assertAllClose(entropy_expect, test_entropy, atol=ATOL)

#   def test_eigvals(self):
#     """Confirms the eigenvalues of the QHBM are correct.

#         Each eigenvalue is the exponential of the energy of a bitstring.
#         """
#     test_qhbm = get_exact_qhbm()
#     this_eigval_test = test_qhbm.probabilities()
#     partition_function = tf.math.exp(test_qhbm.log_partition_function())
#     for n_b, b in enumerate(self.all_bitstrings):
#       eigval_expect = (
#           tf.math.exp(-1.0 * test_qhbm.ebm.energy(b)) / partition_function)
#       self.assertAllClose(eigval_expect, this_eigval_test[n_b], atol=ATOL)

#   def test_unitary_matrix(self):
#     """Confirms the diagonalizing unitary of the QHBM is correct."""
#     test_qhbm = get_exact_qhbm()
#     # This setting reduces test_qhbm.u to a bit flip on every qubit.
#     # Thus, the list of eigenvectors should be the reverse list of basis vectors
#     test_qhbm.qnn.values.assign([0.5, 0.5])
#     eig_list_expect = tf.one_hot([7 - i for i in range(8)], 8)
#     this_eigvec_test = tf.transpose(test_qhbm.unitary_matrix())
#     self.assertAllClose(eig_list_expect, this_eigvec_test)

#     # Test with the internal parameters.
#     test_qhbm = get_exact_qhbm()
#     cirq_circuit = tfq.from_tensor(test_qhbm.qnn.pqc(resolve=True))[0]
#     cirq_unitary = cirq.unitary(cirq_circuit)
#     qhbm_unitary = test_qhbm.unitary_matrix()
#     self.assertAllClose(cirq_unitary, qhbm_unitary)

#   def test_density_matrix(self):
#     """Confirms the density matrix represented by the QHBM is correct."""
#     # Check density matrix of Bell state.
#     n_bits = 2
#     test_ebm = ebm.Bernoulli(n_bits, is_analytic=True)
#     test_ebm.kernel.assign(tf.constant([-10, -10],
#                                        dtype=tf.float32))  # pin at |00>
#     qubits = cirq.GridQubit.rect(1, n_bits)
#     test_u = cirq.Circuit([cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1])])
#     test_qnn = qnn.QNN(test_u, is_analytic=True)
#     test_qhbm = qhbm.QHBM(test_ebm, test_qnn, name="bell")
#     expected_dm = tf.constant(
#         [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]],
#         tf.complex64,
#     )
#     test_dm = test_qhbm.density_matrix()
#     self.assertAllClose(expected_dm, test_dm, atol=ATOL)

#   def test_fidelity(self):
#     """Confirms the fidelity of the QHBM against another matrix is correct."""
#     # TODO(zaqqwerty): Add test where unitary is not equal to transpose
#     # The fidelity of a QHBM with itself is 1.0
#     test_qhbm = get_exact_qhbm()
#     dm = test_qhbm.density_matrix()
#     self.assertAllClose(1.0, test_qhbm.fidelity(dm), atol=ATOL)

#   def test_trainable_variables(self):
#     test_qhbm = get_exact_qhbm()

#     for i in range(len(test_qhbm.trainable_variables)):
#       if i < len(test_qhbm.ebm.trainable_variables):
#         self.assertAllEqual(test_qhbm.ebm.trainable_variables[i],
#                             test_qhbm.trainable_variables[i])
#       else:
#         self.assertAllEqual(
#             test_qhbm.qnn.trainable_variables[
#                 i - len(test_qhbm.ebm.trainable_variables)],
#             test_qhbm.trainable_variables[i])

#     variables = [
#         tf.random.uniform(tf.shape(v)) for v in test_qhbm.trainable_variables
#     ]
#     test_qhbm.trainable_variables = variables
#     for i in range(len(test_qhbm.trainable_variables)):
#       self.assertAllEqual(variables[i], test_qhbm.trainable_variables[i])

#     variables = [tf.Variable(v) for v in variables]
#     test_qhbm.trainable_variables = variables
#     for i in range(len(test_qhbm.trainable_variables)):
#       self.assertAllEqual(variables[i], test_qhbm.trainable_variables[i])

# if __name__ == "__main__":
#   print("Running qhbm_test.py ...")
#   tf.test.main()
