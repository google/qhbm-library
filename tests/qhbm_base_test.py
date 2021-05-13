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
"""Tests for the qhbm_base module."""

import cirq
import sympy
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_quantum as tfq

from qhbm_library import qhbm_base
from tests import test_util


# Global tolerance, set for float32.
ATOL = 1e-5


class BuildBitCircuitTest(tf.test.TestCase):
    """Test build_bit_circuit from the qhbm library."""

    def test_build_bit_circuit(self):
        """Confirm correct bit injector circuit creation."""
        my_qubits = [cirq.GridQubit(0, 2), cirq.GridQubit(1, 4), cirq.GridQubit(2, 2)]
        identifier = "build_bit_test"
        test_circuit, test_symbols = qhbm_base.build_bit_circuit(my_qubits, identifier)
        expected_symbols = list(
            sympy.symbols(
                "_bit_build_bit_test_0 _bit_build_bit_test_1 _bit_build_bit_test_2"
            )
        )
        expected_circuit = cirq.Circuit(
            [cirq.X(q) ** s for q, s in zip(my_qubits, expected_symbols)]
        )
        self.assertAllEqual(test_symbols, expected_symbols)
        self.assertEqual(test_circuit, expected_circuit)


class UniqueWithCountsTest(tf.test.TestCase):
    """Test unique_with_counts from the qhbm library."""

    def test_identity(self):
        # Case when all entries are unique.
        test_bitstrings = tf.constant([[1], [0]], dtype=tf.int8)
        test_y, test_idx, test_count = qhbm_base.unique_with_counts(test_bitstrings)
        for i in tf.range(test_bitstrings.shape[0]):
            self.assertAllEqual(test_y[test_idx[i]], test_bitstrings[i])
        self.assertAllEqual(test_y, test_bitstrings)
        self.assertAllEqual(test_idx, tf.constant([0, 1]))
        self.assertAllEqual(test_count, tf.constant([1, 1]))

    def test_short(self):
        # Case when bitstrings are length 1.
        test_bitstrings = tf.constant(
            [
                [0],
                [1],
                [0],
                [1],
                [1],
                [0],
                [1],
                [1],
            ],
            dtype=tf.int8,
        )
        test_y, test_idx, test_count = qhbm_base.unique_with_counts(test_bitstrings)
        for i in tf.range(test_bitstrings.shape[0]):
            self.assertAllEqual(test_y[test_idx[i]], test_bitstrings[i])
        self.assertAllEqual(test_y, tf.constant([[0], [1]]))
        self.assertAllEqual(test_idx, tf.constant([0, 1, 0, 1, 1, 0, 1, 1]))
        self.assertAllEqual(test_count, tf.constant([3, 5]))

    def test_long(self):
        # Case when bitstrings are of length > 1.
        test_bitstrings = tf.constant(
            [
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 0, 1],
            ],
            dtype=tf.int8,
        )
        test_y, test_idx, test_count = qhbm_base.unique_with_counts(test_bitstrings)
        for i in tf.range(test_bitstrings.shape[0]):
            self.assertAllEqual(test_y[test_idx[i]], test_bitstrings[i])
        self.assertAllEqual(test_y, tf.constant([[1, 0, 1], [1, 1, 1], [0, 1, 1]]))
        self.assertAllEqual(test_idx, tf.constant([0, 1, 2, 0, 1, 2, 0, 0]))
        self.assertAllEqual(test_count, tf.constant([4, 2, 2]))


class InputChecksTest(tf.test.TestCase):
    """Tests all the input checking functions used for QHBMs."""

    def test_upgrade_initial_values(self):
        """Confirms lists of values are properly upgraded to variables."""
        # Test allowed inputs.
        true_list = [-5.1, 2.8, -3.4, 4.8]
        true_tensor = tf.constant(true_list, dtype=tf.float32)
        true_variable = tf.Variable(true_tensor)
        self.assertAllClose(
            qhbm_base.upgrade_initial_values(true_list), true_variable, atol=ATOL
        )
        self.assertAllClose(
            qhbm_base.upgrade_initial_values(true_tensor), true_variable, atol=ATOL
        )
        self.assertAllClose(
            qhbm_base.upgrade_initial_values(true_variable), true_variable, atol=ATOL
        )
        # Check for bad inputs.
        with self.assertRaisesRegex(TypeError, "numeric type"):
            _ = qhbm_base.upgrade_initial_values("junk")
        with self.assertRaisesRegex(ValueError, "must be 1D"):
            _ = qhbm_base.upgrade_initial_values([[5.2]])

    def test_check_function(self):
        """Confirms only allowed functions pass the checks."""

        def base_func(a: tf.Tensor, b: tf.Tensor):
            return a + b

        def analytic_func(a: tf.Tensor):
            return a

        self.assertEqual(base_func, qhbm_base.check_base_function(base_func))
        with self.assertRaisesRegex(TypeError, "two argument"):
            _ = qhbm_base.check_base_function(analytic_func)

    def test_upgrade_symbols(self):
        """Confirms symbols are upgraded appropriately."""
        true_symbol_names = ["test1", "a", "my_symbol", "MySymbol2"]
        values = tf.constant([0 for _ in true_symbol_names])
        true_symbols = [sympy.Symbol(s) for s in true_symbol_names]
        true_symbols_t = tf.constant(true_symbol_names, dtype=tf.string)
        self.assertAllEqual(
            true_symbols_t, qhbm_base.upgrade_symbols(true_symbols, values)
        )
        # Test bad inputs.
        with self.assertRaisesRegex(TypeError, "must be `sympy.Symbol`"):
            _ = qhbm_base.upgrade_symbols(true_symbols[:-1] + ["bad"], values)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            _ = qhbm_base.upgrade_symbols(true_symbols[:-1] + true_symbols[:1], values)
        with self.assertRaisesRegex(ValueError, "symbol for every value"):
            _ = qhbm_base.upgrade_symbols(true_symbols, values[:-1])
        with self.assertRaisesRegex(TypeError, "must be an iterable"):
            _ = qhbm_base.upgrade_symbols(5, values)

    def test_upgrade_circuit(self):
        """Confirms circuits are upgraded appropriately."""
        qubits = cirq.GridQubit.rect(1, 5)
        true_symbol_names = ["a", "b"]
        true_symbols = [sympy.Symbol(s) for s in true_symbol_names]
        true_symbols_t = tf.constant(true_symbol_names, dtype=tf.string)
        true_circuit = cirq.Circuit()
        for q in qubits:
            for s in true_symbols:
                true_circuit += cirq.X(q) ** s
        true_circuit_t = tfq.convert_to_tensor([true_circuit])
        self.assertEqual(
            tfq.from_tensor(true_circuit_t),
            tfq.from_tensor(qhbm_base.upgrade_circuit(true_circuit, true_symbols_t)),
        )
        # Test bad inputs.
        with self.assertRaisesRegex(TypeError, "must be a `cirq.Circuit`"):
            _ = qhbm_base.upgrade_circuit("junk", true_symbols_t)
        with self.assertRaisesRegex(TypeError, "must be a `tf.Tensor`"):
            _ = qhbm_base.upgrade_circuit(true_circuit, true_symbol_names)
        with self.assertRaisesRegex(TypeError, "dtype `tf.string`"):
            _ = qhbm_base.upgrade_circuit(true_circuit, tf.constant([5.5]))
        with self.assertRaisesRegex(ValueError, "must contain"):
            _ = qhbm_base.upgrade_circuit(true_circuit, tf.constant(["a", "junk"]))
        with self.assertRaisesRegex(ValueError, "Empty circuit"):
            _ = qhbm_base.upgrade_circuit(
                cirq.Circuit(), tf.constant([], dtype=tf.string)
            )


class QHBMTest(tf.test.TestCase):
    """Tests the base QHBM class."""

    num_bits = 5
    initial_thetas = tf.random.uniform([num_bits], minval=-1.0)
    raw_phis_symbols = [sympy.Symbol("s0"), sympy.Symbol("s1")]
    phis_symbols = tf.constant([str(s) for s in raw_phis_symbols])
    initial_phis = tf.random.uniform([len(phis_symbols)], minval=-1.0)
    raw_qubits = cirq.GridQubit.rect(1, num_bits)
    u = cirq.Circuit()
    for s in raw_phis_symbols:
        for q in raw_qubits:
            u += cirq.X(q) ** s
    name = "TestQHBM"
    raw_bit_circuit, raw_bit_symbols = qhbm_base.build_bit_circuit(raw_qubits, name)
    bit_symbols = tf.constant([str(s) for s in raw_bit_symbols])
    bit_and_u = tfq.layers.AddCircuit()(raw_bit_circuit, append=u)

    def test_init(self):
        """Confirms QHBM is initialized correctly."""
        energy, sampler = test_util.get_ebm_functions(self.num_bits)
        test_qhbm = qhbm_base.QHBM(
            self.initial_thetas,
            energy,
            sampler,
            self.initial_phis,
            self.raw_phis_symbols,
            self.u,
            self.name,
        )
        self.assertEqual(self.name, test_qhbm.name)
        self.assertAllClose(self.initial_thetas, test_qhbm.thetas)
        self.assertEqual(energy, test_qhbm.energy_function)
        self.assertEqual(sampler, test_qhbm.sampler_function)
        self.assertAllClose(self.initial_phis, test_qhbm.phis)
        self.assertAllEqual(self.phis_symbols, test_qhbm.phis_symbols)
        self.assertAllEqual(
            tfq.from_tensor(tfq.convert_to_tensor([self.u])),
            tfq.from_tensor(test_qhbm.u),
        )
        self.assertAllEqual(
            tfq.from_tensor(tfq.convert_to_tensor([self.u ** -1])),
            tfq.from_tensor(test_qhbm.u_dagger),
        )
        self.assertAllEqual(self.raw_qubits, test_qhbm.raw_qubits)
        self.assertAllEqual(self.bit_symbols, test_qhbm.bit_symbols)
        self.assertEqual(
            tfq.from_tensor(self.bit_and_u), tfq.from_tensor(test_qhbm.bit_and_u)
        )

    def test_copy(self):
        """Confirms copy works correctly."""
        energy, sampler = test_util.get_ebm_functions(self.num_bits)
        test_qhbm = qhbm_base.QHBM(
            self.initial_thetas,
            energy,
            sampler,
            self.initial_phis,
            self.raw_phis_symbols,
            self.u,
            self.name,
        )
        qhbm_copy = test_qhbm.copy()
        self.assertEqual(test_qhbm.name, qhbm_copy.name)
        self.assertAllClose(test_qhbm.thetas, qhbm_copy.thetas)
        self.assertEqual(test_qhbm.energy_function, qhbm_copy.energy_function)
        self.assertEqual(test_qhbm.sampler_function, qhbm_copy.sampler_function)
        self.assertAllClose(test_qhbm.phis, qhbm_copy.phis)
        self.assertAllEqual(test_qhbm.phis_symbols, qhbm_copy.phis_symbols)
        self.assertAllEqual(tfq.from_tensor(test_qhbm.u), tfq.from_tensor(qhbm_copy.u))
        self.assertAllEqual(
            tfq.from_tensor(test_qhbm.u_dagger), tfq.from_tensor(qhbm_copy.u_dagger)
        )
        self.assertAllEqual(test_qhbm.raw_qubits, qhbm_copy.raw_qubits)
        self.assertAllEqual(test_qhbm.bit_symbols, qhbm_copy.bit_symbols)
        self.assertEqual(
            tfq.from_tensor(test_qhbm.bit_and_u), tfq.from_tensor(qhbm_copy.bit_and_u)
        )


def get_basic_qhbm():
    """Returns a basic QHBM for testing."""
    num_bits = 3
    initial_thetas = tf.constant([-23, 0, 17], dtype=tf.float32)
    energy, sampler = test_util.get_ebm_functions(num_bits)
    initial_phis = tf.constant([1.2, -2.5])
    phis_symbols = [sympy.Symbol(s) for s in ["s_static_0", "s_static_1"]]
    u = cirq.Circuit()
    qubits = cirq.GridQubit.rect(1, num_bits)
    for s in phis_symbols:
        for q in qubits:
            u += cirq.X(q) ** s
    name = "static_qhbm"
    return qhbm_base.QHBM(
        initial_thetas, energy, sampler, initial_phis, phis_symbols, u, name
    )


class QHBMBasicFunctionTest(tf.test.TestCase):
    """Test methods of the QHBM class with a simple QHBM."""

    def check_bitstring_exists(self, bitstring, bitstring_list):
        """True if `bitstring` is an entry of `bitstring_list`."""
        return tf.math.reduce_any(
            tf.reduce_all(tf.math.equal(bitstring, bitstring_list), 1)
        )

    def test_sample_bitstrings(self):
        """Confirm only the middle bit alternates."""
        num_samples = int(1e6)
        test_qhbm = get_basic_qhbm()
        test_bitstrings, test_counts = test_qhbm.sample_bitstrings(num_samples)
        self.assertTrue(
            self.check_bitstring_exists(
                tf.constant([0, 0, 1], dtype=tf.int8), test_bitstrings
            )
        )
        self.assertTrue(
            self.check_bitstring_exists(
                tf.constant([0, 1, 1], dtype=tf.int8), test_bitstrings
            )
        )
        # Sanity check that absent bitstring is really missing.
        self.assertFalse(
            self.check_bitstring_exists(
                tf.constant([0, 0, 0], dtype=tf.int8), test_bitstrings
            )
        )
        # Only the two expected bitstrings should exist.
        self.assertAllEqual(tf.shape(test_bitstrings), [2, 3])
        self.assertEqual(tf.reduce_sum(test_counts), num_samples)
        self.assertAllClose(1.0, test_counts[0] / test_counts[1], atol=1e-3)

    def test_sample_state_circuits(self):
        """Confirm circuits are sampled correctly."""
        num_samples = int(1e6)
        test_qhbm = get_basic_qhbm()
        test_circuit_samples, _, _ = test_qhbm.sample_state_circuits(num_samples)
        # Circuits with the allowed-to-be-sampled bitstrings prepended.
        resolved_u_t = tfq.resolve_parameters(
            test_qhbm.u, test_qhbm.phis_symbols, tf.expand_dims(test_qhbm.phis, 0)
        )
        resolved_u = tfq.from_tensor(resolved_u_t)[0]
        expected_circuit_samples = [
            cirq.Circuit(
                cirq.X(test_qhbm.raw_qubits[0]) ** 0,
                cirq.X(test_qhbm.raw_qubits[1]) ** 0,
                cirq.X(test_qhbm.raw_qubits[2]),
            )
            + resolved_u,
            cirq.Circuit(
                cirq.X(test_qhbm.raw_qubits[0]) ** 0,
                cirq.X(test_qhbm.raw_qubits[1]),
                cirq.X(test_qhbm.raw_qubits[2]),
            )
            + resolved_u,
        ]
        # Check that both circuits are generated.
        test_circuit_samples_deser = tfq.from_tensor(test_circuit_samples)
        self.assertTrue(
            any(
                [
                    expected_circuit_samples[0] == test_circuit_samples_deser[0],
                    expected_circuit_samples[0] == test_circuit_samples_deser[1],
                ]
            )
        )
        self.assertTrue(
            any(
                [
                    expected_circuit_samples[1] == test_circuit_samples_deser[0],
                    expected_circuit_samples[1] == test_circuit_samples_deser[1],
                ]
            )
        )

    def test_sample_unresolved_state_circuits(self):
        """Confirm unresolved circuits are sampled correctly."""
        # TODO(b/182904206)

    def test_sample_pulled_back_bitstrings(self):
        """Ensures pulled back bitstrings are correct."""
        test_qhbm = get_basic_qhbm()
        # This setting reduces test_qhbm.u to a bit flip on every qubit.
        test_qhbm.phis = tf.Variable([0.5, 0.5])
        circuits = tfq.convert_to_tensor(
            [
                cirq.Circuit(
                    cirq.X(test_qhbm.raw_qubits[0]), cirq.X(test_qhbm.raw_qubits[2])
                ),
                cirq.Circuit(
                    cirq.X(test_qhbm.raw_qubits[1]), cirq.X(test_qhbm.raw_qubits[2])
                ),
            ]
        )
        n_samples_0 = int(1e4)
        n_samples_1 = int(2e4)
        counts = tf.constant([n_samples_0, n_samples_1])
        ragged_samples = test_qhbm.sample_pulled_back_bitstrings(circuits, counts)
        test_samples_0 = ragged_samples[0].to_tensor()
        test_samples_1 = ragged_samples[1].to_tensor()
        self.assertEqual(n_samples_0, test_samples_0.shape[0])
        self.assertEqual(n_samples_1, test_samples_1.shape[0])
        uniques_0, _, _ = qhbm_base.unique_with_counts(test_samples_0)
        uniques_1, _, _ = qhbm_base.unique_with_counts(test_samples_1)
        self.assertEqual(1, uniques_0.shape[0])
        self.assertEqual(1, uniques_1.shape[0])
        self.assertAllEqual(tf.constant([0, 1, 0], dtype=tf.int8), uniques_0[0])
        self.assertAllEqual(tf.constant([1, 0, 0], dtype=tf.int8), uniques_1[0])

    def test_energy_and_energy_grad(self):
        """Confirms energies are correct.

        The simple energy function is
        energy(thetas, b) = sum_i thetas[i] * (2*b[i]

        The derivative of this function with respect to `thetas` is
        [b[i] for i in len(thetas)]
        """
        test_qhbm = get_basic_qhbm()
        test_thetas = tf.identity(test_qhbm.thetas)
        for b in [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]:
            b = tf.constant(b, dtype=tf.float32)
            energy_expect = (
                test_thetas[0] * (1.0 - 2.0 * b[0])
                + test_thetas[1] * (1.0 - 2.0 * b[1])
                + test_thetas[2] * (1.0 - 2.0 * b[2])
            )
            grad_expect = [1.0 - 2.0 * b[0], 1.0 - 2.0 * b[1], 1.0 - 2.0 * b[2]]
            test_energy, test_grad = test_qhbm.energy_and_energy_grad(b)
            self.assertEqual(energy_expect, test_energy)
            self.assertAllEqual(grad_expect, test_grad)

    def test_pulled_back_energy_expectation(self):
        """Tests pulled back energy expectation."""
        test_qhbm = get_basic_qhbm()
        # This setting reduces test_qhbm.u to a bit flip on every qubit.
        test_qhbm.phis = tf.Variable([0.5, 0.5])
        circuits = tfq.convert_to_tensor(
            [
                cirq.Circuit(
                    cirq.X(test_qhbm.raw_qubits[0]), cirq.X(test_qhbm.raw_qubits[2])
                ),
                cirq.Circuit(
                    cirq.X(test_qhbm.raw_qubits[1]), cirq.X(test_qhbm.raw_qubits[2])
                ),
            ]
        )
        n_samples_0 = int(1e4)
        n_samples_1 = int(2e4)
        counts = tf.constant([n_samples_0, n_samples_1])
        test_energy = test_qhbm.pulled_back_energy_expectation(circuits, counts)
        # Get the individual energies of the pulled back bitstrings.
        e_0 = test_qhbm.energy_function(
            test_qhbm.thetas, tf.constant([0, 1, 0], dtype=tf.int8)
        )
        e_1 = test_qhbm.energy_function(
            test_qhbm.thetas, tf.constant([1, 0, 0], dtype=tf.int8)
        )
        e_avg = (n_samples_0 * e_0 + n_samples_1 * e_1) / (n_samples_0 + n_samples_1)
        self.assertEqual(e_avg, test_energy)


def get_exact_qhbm():
    """Returns a basic ExactQHBM for testing."""
    qhbm = get_basic_qhbm()
    return qhbm_base.ExactQHBM(
        qhbm.thetas,
        qhbm.energy_function,
        qhbm.sampler_function,
        qhbm.phis,
        [sympy.Symbol(s.decode("utf-8")) for s in qhbm.phis_symbols.numpy()],
        tfq.from_tensor(qhbm.u)[0],
        qhbm.name,
    )


class ExactQHBMBasicFunctionTest(tf.test.TestCase):
    """Test methods of the exact QHBM class with a simple QHBM."""

    all_bitstrings = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]

    def test_exact_qhbm_init(self):
        """Confirms all bitstrings exist."""
        test_qhbm = get_exact_qhbm()
        for n_b, b in enumerate(self.all_bitstrings):
            self.assertAllEqual(b, test_qhbm.all_strings[n_b])

    def test_copy(self):
        """Confirms copy works correctly."""
        test_qhbm = get_exact_qhbm()
        qhbm_copy = test_qhbm.copy()
        self.assertEqual(test_qhbm.name, qhbm_copy.name)
        self.assertAllClose(test_qhbm.thetas, qhbm_copy.thetas)
        self.assertEqual(test_qhbm.energy_function, qhbm_copy.energy_function)
        self.assertEqual(test_qhbm.sampler_function, qhbm_copy.sampler_function)
        self.assertAllClose(test_qhbm.phis, qhbm_copy.phis)
        self.assertAllEqual(test_qhbm.phis_symbols, qhbm_copy.phis_symbols)
        self.assertAllEqual(tfq.from_tensor(test_qhbm.u), tfq.from_tensor(qhbm_copy.u))
        self.assertAllEqual(
            tfq.from_tensor(test_qhbm.u_dagger), tfq.from_tensor(qhbm_copy.u_dagger)
        )
        self.assertAllEqual(test_qhbm.raw_qubits, qhbm_copy.raw_qubits)
        self.assertAllEqual(test_qhbm.bit_symbols, qhbm_copy.bit_symbols)
        self.assertEqual(
            tfq.from_tensor(test_qhbm.bit_and_u), tfq.from_tensor(qhbm_copy.bit_and_u)
        )
        for n_b, b in enumerate(self.all_bitstrings):
            self.assertAllEqual(b, test_qhbm.all_strings[n_b])

    def test_all_energies(self):
        """Confirms that each bitstring energy is correct."""
        test_qhbm = get_exact_qhbm()
        energy_function, _ = test_util.get_ebm_functions(3)
        test_energies = test_qhbm.all_energies()
        for n_b, b in enumerate(self.all_bitstrings):
            self.assertAllClose(
                energy_function(test_qhbm.thetas, b), test_energies[n_b], atol=ATOL
            )

    def test_log_partition_function(self):
        """Confirms the logarithm of the partition function is correct.

        The basic energy function used for testing is independent between bits,
        so that E(b) = sum_i E_i(b[i]).  The partition function for such an E is
        Z = sum_b e^{-E(b)}
          = sum_b e^{-sum_i E_i(b[i])}
          = sum_b prod_i e^{-E_i(b[i])}
          = prod_i (e^{-E_i(0)) + e^{-E_i(1)})
        and for the simple energy function, E_i(b[i]) = theta[i] * (1 - 2 * b[i]),
        Z = prod_i (e^{-theta[i]) + e^{theta[i]})
        """

        def base_val(t):
            return tf.math.exp(-t) + tf.math.exp(t)

        test_qhbm = get_exact_qhbm()
        log_partition_expect = tf.math.log(
            tf.reduce_prod([base_val(theta) for theta in test_qhbm.thetas.numpy()])
        )
        test_log_partition = test_qhbm.log_partition_function()
        self.assertAllClose(log_partition_expect, test_log_partition, atol=ATOL)

    def test_entropy_function(self):
        """Confirms the entropy of the QHBM is correct."""
        test_qhbm = get_exact_qhbm()
        entropy_expect = tf.reduce_sum(
            tfp.distributions.Bernoulli(logits=test_qhbm.thetas).entropy()
        )
        test_entropy = test_qhbm.entropy_function()
        self.assertAllClose(entropy_expect, test_entropy, atol=ATOL)

    def test_eigvals(self):
        """Confirms the eigenvalues of the QHBM are correct.

        Each eigenvalue is the exponential of the energy of a bitstring.
        """
        test_qhbm = get_exact_qhbm()
        this_eigval_test = test_qhbm.eigvals()
        partition_function = tf.math.exp(test_qhbm.log_partition_function())
        for n_b, b in enumerate(self.all_bitstrings):
            eigval_expect = (
                tf.math.exp(-1.0 * test_qhbm.energy_and_energy_grad(b)[0])
                / partition_function
            )
            self.assertAllClose(eigval_expect, this_eigval_test[n_b], atol=ATOL)

    def test_unitary_matrix(self):
        """Confirms the diagonalizing unitary of the QHBM is correct."""
        test_qhbm = get_exact_qhbm()
        # This setting reduces test_qhbm.u to a bit flip on every qubit.
        # Thus, the list of eigenvectors should be the reverse list of basis vectors
        test_qhbm.phis = tf.Variable([0.5, 0.5])
        eig_list_expect = tf.one_hot([7 - i for i in range(8)], 8)
        this_eigvec_test = tf.transpose(test_qhbm.unitary_matrix())
        self.assertAllClose(eig_list_expect, this_eigvec_test)

        # Test with the internal parameters.
        test_qhbm = get_exact_qhbm()
        cirq_circuit = tfq.from_tensor(
            tfq.resolve_parameters(
                test_qhbm.u, test_qhbm.phis_symbols, tf.expand_dims(test_qhbm.phis, 0)
            )
        )[0]
        cirq_unitary = cirq.unitary(cirq_circuit)
        qhbm_unitary = test_qhbm.unitary_matrix()
        self.assertAllClose(cirq_unitary, qhbm_unitary)

    def test_eigvecs(self):
        """Confirms the eigenvectors of the QHBM are correct."""
        test_qhbm = get_exact_qhbm()
        eigvecs_expect = tf.transpose(test_qhbm.unitary_matrix())
        test_eigvecs = test_qhbm.eigvecs()
        self.assertAllClose(eigvecs_expect, test_eigvecs)

    def test_density_matrix(self):
        """Confirms the density matrix represented by the QHBM is correct."""
        # Check density matrix of Bell state.
        n_bits = 2
        energy, sampler = test_util.get_ebm_functions(n_bits)
        initial_thetas = tf.constant([-10, -10], dtype=tf.float32)  # pin at |00>
        qubits = cirq.GridQubit.rect(1, n_bits)
        test_u = cirq.Circuit([cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1])])
        test_qhbm = qhbm_base.ExactQHBM(
            initial_thetas, energy, sampler, tf.constant([]), [], test_u, "bell"
        )
        expected_dm = tf.constant(
            [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]],
            tf.complex64,
        )
        test_dm = test_qhbm.density_matrix()
        self.assertAllClose(expected_dm, test_dm, atol=ATOL)

    def test_fidelity(self):
        """Confirms the fidelity of the QHBM against another matrix is correct."""
        # TODO(zaqqwerty): Add test where unitary is not equal to transpose
        # The fidelity of a QHBM with itself is 1.0
        test_qhbm = get_exact_qhbm()
        dm = test_qhbm.density_matrix()
        self.assertAllClose(1.0, test_qhbm.fidelity(dm), atol=ATOL)


if __name__ == "__main__":
    print("Running qhbm_base_test.py ...")
    tf.test.main()