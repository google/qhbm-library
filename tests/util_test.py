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
"""Tests for util.py."""
import math

import cirq
import tensorflow as tf
import tensorflow_quantum as tfq

from qhbmlib import util
from qhbmlib.tests import test_util

# ============================================================================ #
# Density matrix utilities.
# ============================================================================ #


class PureStateTensorToDensityMatrixTest(tf.test.TestCase):
  """Test pure_state_tensor_to_density_matrix from the qhbm library."""

  def test_simple_states(self):
    """Test one-qubit density matrix, dm = 0.75|0><0| + 0.25|1><1|."""
    test_tensor = tf.constant([[1, 0], [1, 0], [1, 0], [0, 1]],
                              dtype=tf.dtypes.complex64)
    test_density = util.pure_state_tensor_to_density_matrix(
        test_tensor, tf.ones([4]))
    ref_density = tf.constant([[0.75, 0], [0, 0.25]], dtype=tf.dtypes.complex64)
    self.assertAllClose(test_density, ref_density)

  def test_len_4_states(self):
    """Test two-qubit states."""
    test_tensor = tf.constant(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]],
        dtype=tf.dtypes.complex64,
    )
    test_density = util.pure_state_tensor_to_density_matrix(
        test_tensor, tf.ones([5]))
    ref_density = tf.constant(
        [[0.2, 0, 0, 0], [0, 0.2, 0, 0], [0, 0, 0.4, 0], [0, 0, 0, 0.2]],
        dtype=tf.dtypes.complex64,
    )
    self.assertAllClose(test_density, ref_density)

  def test_complex_states(self):
    """Test complex state inputs."""
    test_tensor = tf.constant(
        [
            [math.sqrt(0.5) + 0j, math.sqrt(0.5) * 1j],
            [math.sqrt(0.5) + 0j, math.sqrt(0.5) * 1j],
            [0j, 1j],
        ],
        dtype=tf.dtypes.complex64,
    )
    test_density = util.pure_state_tensor_to_density_matrix(
        test_tensor, tf.ones([3]))
    ref_density = tf.constant(
        [[(1 / 3) + 0j, -1j * (1 / 3)], [(1 / 3) * 1j, (1 / 3) * 2 + 0j]],
        dtype=tf.dtypes.complex64,
    )
    self.assertAllClose(test_density, ref_density)

  def test_differing_counts(self):
    """Confirm differing counts are interpreted correctly."""
    test_tensor = tf.constant([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                              dtype=tf.dtypes.complex64)
    test_counts = tf.constant([1, 7, 2])
    test_density = util.pure_state_tensor_to_density_matrix(
        test_tensor, test_counts)
    ref_density = tf.constant(
        [[0.1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.7, 0], [0, 0, 0, 0.2]],
        dtype=tf.dtypes.complex64,
    )
    self.assertAllClose(test_density, ref_density)


class CircuitsAndCountsToDensityMatrixTest(tf.test.TestCase):
  """Test circuits_and_counts_to_density_matrix from the qhbm library."""

  def test_identity(self):
    """Confirm that the all-identities circuit yields the zero state density."""
    num_qubits = 4
    qubits = cirq.GridQubit.rect(1, num_qubits)
    id_circ = cirq.Circuit([cirq.I(q) for q in qubits])
    dm_vec = tf.one_hot(0, 2**num_qubits, dtype=tf.complex64)
    expected_density = tf.reshape(dm_vec, [2**num_qubits, 1]) * tf.reshape(
        dm_vec, [1, 2**num_qubits])
    counts = tf.constant([1])
    test_density = util.circuits_and_counts_to_density_matrix(
        tfq.convert_to_tensor([id_circ]), counts)
    self.assertAllClose(test_density, expected_density)


class FidelityTest(tf.test.TestCase):
  """Test fidelity from the qhbm library."""

  def test_high_fidelity(self):
    """Test the case where density matrices match."""
    test_dm1 = tf.constant([[0.5, -0.5], [-0.5, 0.5]])
    test_dm2 = tf.constant([[0.5, -0.5], [-0.5, 0.5]])
    test_fidelity = util.fidelity(test_dm1, test_dm2)
    ref_fidelity = 1.0
    self.assertAllClose(test_fidelity, ref_fidelity)

  def test_low_fidelity(self):
    """Test the case where the density matrices do not match."""
    test_dm1 = tf.constant([[0.9999999, 0], [0, 0.0000001]])
    test_dm2 = tf.constant([[0.0000001, 0], [0, 0.9999999]])
    test_fidelity = util.fidelity(test_dm1, test_dm2)
    ref_fidelity = 0.0
    self.assertAllClose(test_fidelity, ref_fidelity)

  def test_larger_matrix(self):
    """Confirm correct operation on large matrices."""
    size = 2**8
    test_dm1 = tf.eye(size) / size
    test_dm2 = tf.eye(size) / size
    test_fidelity = util.fidelity(test_dm1, test_dm2)
    ref_fidelity = 1.0
    self.assertAllClose(test_fidelity, ref_fidelity)


class FastFidelity(tf.test.TestCase):
  """Test fidelity_eigh from the qhbm library."""

  def test_high_fidelity(self):
    """Test the case where density matrices match."""
    test_dm1 = tf.constant([[0.5, -0.5], [-0.5, 0.5]])
    test_dm2 = tf.constant([[0.5, -0.5], [-0.5, 0.5]])
    test_fidelity = util.fast_fidelity(test_dm1, test_dm2)
    ref_fidelity = util.fidelity(test_dm1, test_dm2)
    self.assertAllClose(test_fidelity, ref_fidelity)

  def test_low_fidelity(self):
    """Test the case where the density matrices do not match."""
    test_dm1 = tf.constant([[0.9999999, 0], [0, 0.0000001]])
    test_dm2 = tf.constant([[0.0000001, 0], [0, 0.9999999]])
    test_fidelity = util.fast_fidelity(test_dm1, test_dm2)
    ref_fidelity = util.fidelity(test_dm1, test_dm2)
    self.assertAllClose(test_fidelity, ref_fidelity)

  def test_larger_matrix(self):
    """Confirm correct operation on large matrices."""
    size = 2**8
    test_dm1 = tf.eye(size) / size
    test_dm2 = tf.eye(size) / size
    test_fidelity = util.fast_fidelity(test_dm1, test_dm2)
    ref_fidelity = util.fidelity(test_dm1, test_dm2)
    self.assertAllClose(test_fidelity, ref_fidelity)

  def test_pure_quantum_states(self):
    """Confirm correct operation on large pure quantum states."""
    num_qubit = 5
    # Too slow. just test 1 sample.
    test_dm1 = test_util.generate_pure_random_density_operator(num_qubit)
    test_dm2 = test_util.generate_pure_random_density_operator(num_qubit)
    test_fidelity = util.fast_fidelity(test_dm1, test_dm2)
    ref_fidelity = util.fidelity(test_dm1, test_dm2)
    self.assertAllClose(test_fidelity, ref_fidelity, atol=1e-4)


class FastFidelityMixedStates(tf.test.TestCase):

  def test_mixed_quantum_states(self):
    """Confirm correct operation on large mixed quantum states."""
    num_qubit = 5
    # Too slow. just test 1 sample.
    test_dm1, _ = test_util.generate_mixed_random_density_operator(num_qubit)
    test_dm2, _ = test_util.generate_mixed_random_density_operator(num_qubit)
    test_fidelity = util.fast_fidelity(test_dm1, test_dm2)
    ref_fidelity = util.fidelity(test_dm1, test_dm2)
    self.assertAllClose(test_fidelity, ref_fidelity, atol=1e-4)


class FastFidelityMixedStatesWithSameBasis(tf.test.TestCase):

  def test_mixed_quantum_states_with_same_basis(self):
    """Confirm correct operation on mixed quantum states with same basis."""
    num_qubit = 5
    # Too slow. just test 1 sample.
    test_dm1, test_dm2 = test_util.generate_mixed_random_density_operator_pair(
        num_qubit)
    test_fidelity = util.fast_fidelity(test_dm1, test_dm2)
    ref_fidelity = util.fidelity(test_dm1, test_dm2)
    self.assertAllClose(test_fidelity, ref_fidelity, atol=1e-4)


class FastFidelityMixedStatesWithSameButPermutedBasis(tf.test.TestCase):

  def test_mixed_quantum_states_with_same_but_permuted_basis(self):
    """Confirm operations on mixed quantum states with same permuted basis."""
    num_qubit = 5
    for _ in range(3):  # test 3 samples
      test_dm1, test_dm2 = test_util.generate_mixed_random_density_operator_pair(
          num_qubit, perm_basis=True)
      test_fidelity = util.fast_fidelity(test_dm1, test_dm2)
      ref_fidelity = util.fidelity(test_dm1, test_dm2)
      self.assertAllClose(test_fidelity, ref_fidelity, atol=1e-4)


class NumpyFidelity(tf.test.TestCase):
  """Test np_fidelity from the qhbm library."""

  def test_high_fidelity(self):
    """Test the case where density matrices match."""
    test_dm1 = tf.constant([[0.5, -0.5], [-0.5, 0.5]])
    test_dm2 = tf.constant([[0.5, -0.5], [-0.5, 0.5]])
    test_fidelity = util.np_fidelity(test_dm1, test_dm2)
    ref_fidelity = util.fidelity(test_dm1, test_dm2)
    self.assertAllClose(test_fidelity, ref_fidelity)

  def test_low_fidelity(self):
    """Test the case where the density matrices do not match."""
    test_dm1 = tf.constant([[0.9999999, 0], [0, 0.0000001]])
    test_dm2 = tf.constant([[0.0000001, 0], [0, 0.9999999]])
    test_fidelity = util.np_fidelity(test_dm1, test_dm2)
    ref_fidelity = util.fidelity(test_dm1, test_dm2)
    self.assertAllClose(test_fidelity, ref_fidelity)

  def test_larger_matrix(self):
    """Confirm correct operation on large matrices."""
    size = 2**8
    test_dm1 = tf.eye(size) / size
    test_dm2 = tf.eye(size) / size
    test_fidelity = util.np_fidelity(test_dm1, test_dm2)
    ref_fidelity = util.fidelity(test_dm1, test_dm2)
    self.assertAllClose(test_fidelity, ref_fidelity)

  # TODO(b/181024487)
  # def test_pure_quantum_states(self):
  #   """Confirm correct operation on large pure quantum states."""
  #   num_qubit = 5
  #   # Too slow. just test 1 sample.
  #   test_dm1 = generate_pure_random_density_operator(num_qubit)
  #   test_dm2 = generate_pure_random_density_operator(num_qubit)
  #   test_fidelity = util.np_fidelity(test_dm1, test_dm2)
  #   ref_fidelity = util.fidelity(test_dm1, test_dm2)
  #   self.assertAllClose(test_fidelity, ref_fidelity)

  # TODO(b/181024487)


# class NumpyFidelityMixedStates(tf.test.TestCase):

#   def test_mixed_quantum_states(self):
#     """Confirm correct operation on large mixed quantum states."""
#     num_qubit = 5
#     # Too slow. just test 1 sample.
#     test_dm1 = generate_mixed_random_density_operator(num_qubit)
#     test_dm2 = generate_mixed_random_density_operator(num_qubit)
#     test_fidelity = util.np_fidelity(test_dm1, test_dm2)
#     ref_fidelity = util.fidelity(test_dm1, test_dm2)
#     self.assertAllClose(test_fidelity, ref_fidelity)

# TODO(b/181024487)
# class NumpyFidelityMixedStatesWithSameBasis(tf.test.TestCase):

#   def test_mixed_quantum_states_with_same_basis(self):
#     """Confirm correct operation on mixed quantum states with same basis."""
#     num_qubit = 5
#     # Too slow. just test 1 sample.
#     test_dm1, test_dm2 = generate_mixed_random_density_operator_pair(
#         num_qubit)
#     test_fidelity = util.np_fidelity(test_dm1, test_dm2)
#     ref_fidelity = util.fidelity(test_dm1, test_dm2)
#     self.assertAllClose(test_fidelity, ref_fidelity)

# TODO(b/181024487)
# class NumpyFidelityMixedStatesWithSameButPermutedBasis(tf.test.TestCase):

#   def test_mixed_quantum_states_with_same_but_permuted_basis(self):
#     """Confirm operations on mixed quantum states with same permuted basis."""
#     num_qubit = 5
#     for _ in range(3):  # test 3 samples
#       test_dm1, test_dm2 = generate_mixed_random_density_operator_pair(
#           num_qubit, perm_basis=True)
#       test_fidelity = util.np_fidelity(test_dm1, test_dm2)
#       ref_fidelity = util.fidelity(test_dm1, test_dm2)
#       self.assertAllClose(test_fidelity, ref_fidelity)


class GetThermalStateTest(tf.test.TestCase):
  """Test get_thermal_state from the qhbm library."""

  def test_specific_matrix(self):
    """Confirm correct thermal state for explicit matrix."""
    xz = tf.constant(
        [[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, -1, 0, 0]],
        dtype=tf.complex128,
    )
    beta = tf.constant(1.3, dtype=tf.float64)
    direct_exp = tf.linalg.expm(-tf.cast(beta, tf.complex128) * xz)
    direct_thermal = direct_exp / tf.linalg.trace(direct_exp)
    stable_thermal = util.get_thermal_state(beta, xz)
    self.assertAllClose(direct_thermal, stable_thermal, atol=1e-10)

  def test_many_matrices(self):
    """Confirm stable function is equal to the simple thermal state calc."""
    beta_list = [0.1, 0.8, 2.9, 27]
    num_qubits_list = [2, 3, 4, 5, 6, 7]
    for num_qubits in num_qubits_list:
      qubits = cirq.GridQubit.rect(1, num_qubits)
      for beta in beta_list:
        beta_t = tf.constant(beta, dtype=tf.float64)
        pauli_sum = test_util.get_random_pauli_sum(qubits)
        pauli_matrix = tf.constant(pauli_sum.matrix(), dtype=tf.complex128)
        direct_exp = tf.linalg.expm(-beta * pauli_matrix)
        direct_thermal = direct_exp / tf.linalg.trace(direct_exp)
        stable_thermal = util.get_thermal_state(beta_t, pauli_matrix)
        self.assertAllClose(direct_thermal, stable_thermal, atol=1e-10)


class LogPartitionFunctionTest(tf.test.TestCase):
  """Test log_partition_function from the qhbm library."""

  def test_specific_matrix(self):
    """Confirm log partition function for specific Hamiltonian."""
    yz = tf.constant(
        [[0, 0, -1j, 0], [0, 0, 0, 1j], [1j, 0, 0, 0], [0, -1j, 0, 0]],
        dtype=tf.complex128,
    )
    beta = tf.constant(3.8, dtype=tf.float64)
    direct_exp = tf.linalg.expm(-tf.cast(beta, tf.complex128) * yz)
    partition_function = tf.linalg.trace(direct_exp)
    actual_log_partition = tf.math.log(partition_function)
    test_log_partition = util.log_partition_function(beta, yz)
    self.assertAllClose(actual_log_partition, test_log_partition, 1e-10)

  def test_many_matrices(self):
    """Confirm log parition function for random Hamiltonians."""
    num_trials_per_qubit = 3
    num_qubits_list = [1, 2, 3, 4, 5]
    h_max = 0.25  # can not make too large or direct expm will get nan sometimes
    h_min = -h_max
    for num_qubits in num_qubits_list:
      for _ in range(num_trials_per_qubit):
        # Get random components.
        random_mat = tf.random.uniform(
            [2**num_qubits, 2**num_qubits],
            minval=h_min,
            maxval=h_max,
            dtype=tf.float64,
        )
        random_upper_triangular = tf.linalg.band_part(random_mat, 0, -1)
        random_lower_triangular = tf.linalg.band_part(random_mat, -1, 0)
        random_diagonal = tf.linalg.band_part(random_mat, 0, 0)

        # Build random Hamiltonian out of random matrix components.
        random_real_part = (
            random_lower_triangular +
            tf.linalg.adjoint(random_lower_triangular) + random_diagonal)
        random_upper_triangular_imag = (
            tf.cast(random_upper_triangular, tf.complex128) * 1j)
        random_imag_part = random_upper_triangular_imag + tf.linalg.adjoint(
            random_upper_triangular_imag)
        random_h = tf.cast(random_real_part, tf.complex128) + random_imag_part

        # Compare stable version to direct exponentiation.
        beta = tf.random.uniform([], dtype=tf.float64)
        direct_exp = tf.linalg.expm(-tf.cast(beta, tf.complex128) * random_h)
        partition_function = tf.linalg.trace(direct_exp)
        actual_log_partition = tf.math.log(partition_function)
        test_log_partition = util.log_partition_function(beta, random_h)
        self.assertAllClose(actual_log_partition, test_log_partition, 1e-10)


class EntropyTest(tf.test.TestCase):
  """Test entropy from the qhbm library."""

  def test_pure_state(self):
    """Entropy of pure states should be zero."""
    num_trials_per_qubit = 3
    num_qubits_list = [1, 2, 3, 4, 5]
    for num_qubits in num_qubits_list:
      for _ in range(num_trials_per_qubit):
        test_dm = test_util.generate_pure_random_density_operator(num_qubits)
        self.assertAllClose(0, util.entropy(test_dm), 1e-10)

  def test_slightly_mixed_state(self):
    """Confirm entropy of dm is same as for the probabilities of the mixture."""
    num_mixin_list = [2, 3, 4]
    num_qubits_list = [2, 3, 4, 5]
    for num_qubits in num_qubits_list:
      for num_mixin in num_mixin_list:
        total_dm, probs = test_util.generate_mixed_random_density_operator(
            num_qubits, num_mixtures=num_mixin)
        # Entropy of total_dm is the same as that of the mixin distribution.
        actual_entropy = test_util.stable_classical_entropy(probs)
        test_entropy = util.entropy(total_dm)
        self.assertAllClose(actual_entropy, test_entropy, 1e-10)

  def test_fully_mixed_state(self):
    """Confirm entropy of completely mixed state.

        On N qubits, the completely mixed state is rho = I_{2^N}/(2^N), the
        identity
        matrix on N qubits divided by 2^N. The entropy is
        S(M) = -Tr[rho ln rho] = -sum_i 2^-N ln 2^-N = -(2^N) (2^-N) ln 2^-N
             = - ln 2^-N = ln 2^N = N ln 2
        """
    num_qubits_list = [1, 2, 3, 4, 5]
    for num_qubits in num_qubits_list:
      rho = tf.eye(2**num_qubits, dtype=tf.complex128) / (2.0**num_qubits)
      test_entropy = util.entropy(rho)
      actual_entropy = num_qubits * tf.math.log(2.0)
      self.assertAllClose(actual_entropy, test_entropy, 1e-10)


# ============================================================================ #
# Bitstring utilities.
# ============================================================================ #

if __name__ == "__main__":
  print("Running util_test.py ...")
  tf.test.main()
