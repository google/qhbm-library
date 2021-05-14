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
"""Implementation of general QHBMs in TFQ."""

import inspect
import itertools
import numbers
from typing import Any, Callable, Iterable, List, Union

import cirq
import sympy
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_quantum as tfq


def build_bit_circuit(qubits, ident):
    """Returns exponentiated X gate on each qubit and the exponent symbols."""
    circuit = cirq.Circuit()
    symbols = []
    for n, q in enumerate(qubits):
        new_bit = sympy.Symbol("_bit_{0}_{1}".format(ident, n))
        circuit += cirq.X(q) ** new_bit
        symbols.append(new_bit)
    return circuit, symbols


@tf.function
def unique_with_counts(input_bitstrings, out_idx=tf.int32):
    """Extract the unique bitstrings in the given bitstring tensor.

    Works by converting each bitstring to a 64 bit integer, then using built-in
    `tf.unique_with_counts` on this 1-D array, then mapping these integers back to
    bitstrings. The inputs and outputs are to be related by the same invariants as
    those of `tf.unique_with_counts`,
    y[idx[i]] = input_bitstrings[i] for i in [0, 1,...,rank(input_bitstrings) - 1]

    TODO(zaqqwerty): the signature and return values are designed to be similar
    to those of tf.unique_with_counts.  This function is needed because
    `tf.unique_with_counts` does not work on 2-D tensors.  When it begins to work
    on 2-D tensors, then this function will be deprecated.

    Args:
      input_bitstrings: 2-D `tf.Tensor` of dtype `int8`.  This tensor is
        interpreted as a list of bitstrings.  Bitstrings are required to be
        64 bits or fewer.
      out_idx: An optional `tf.DType` from: `tf.int32`, `tf.int64`. Defaults to
        `tf.int32`.  Specified type of idx and count outputs.

    Returns:
      y: 2-D `tf.Tensor` of dtype `int8` containing the unique 0-axis entries of
        `input_bitstrings`.
      idx: 1-D `tf.Tensor` of dtype `out_idx` such that `idx[i]` is the index in
        `y` containing the value `input_bitstrings[i]`.
      count: 1-D `tf.Tensor` of dtype `out_idx` such that `count[i]` is the number
        of occurences of `y[i]` in `input_bitstrings`.
    """
    print("retracing: unique_with_counts")
    # Convert bitstrings to integers and uniquify those integers.
    input_shape = tf.shape(input_bitstrings)
    mask = tf.cast(input_bitstrings, dtype=tf.int64)
    base = tf.bitwise.left_shift(
        mask, tf.range(tf.cast(input_shape[1], dtype=tf.int64), dtype=tf.int64)
    )
    ints_equiv = tf.reduce_sum(base, 1)
    _, idx, count = tf.unique_with_counts(ints_equiv, out_idx=out_idx)

    # Convert unique integers to corresponding unique bitstrings.
    y = tf.zeros((tf.shape(count)[0], input_shape[1]), dtype=tf.int8)
    y = tf.tensor_scatter_nd_update(y, tf.expand_dims(idx, axis=1), input_bitstrings)

    return y, idx, count


def upgrade_initial_values(
    initial_values: Union[List[numbers.Real], tf.Tensor, tf.Variable]
) -> tf.Variable:
    """Upgrades the given values to a tf.Variable.

    Args:
      initial_values: Numeric values to upgrade.

    Returns:
      The input values upgraded to a fresh `tf.Variable` of dtype `tf.float32`.
    """
    if isinstance(initial_values, tf.Variable):
        initial_values = initial_values.read_value()
    if isinstance(initial_values, (List, tf.Tensor)):
        initial_values = tf.Variable(
            tf.cast(initial_values, tf.float32), dtype=tf.float32
        )
        if len(tf.shape(initial_values)) != 1:
            raise ValueError("Values for QHBMs must be 1D.")
        return initial_values
    raise TypeError(f"Input needs to be a numeric type, got {type(initial_values)}")


def check_base_function(
    fn: Callable[[tf.Tensor, tf.Tensor], Any]
) -> Callable[[tf.Tensor, tf.Tensor], Any]:
    """Checks that a given function is valid for the base QHBM class.

    The base functions take the `theta` parameters of the classical distribution
    and either a bitstring or a number, all as `tf.Tensor`s.

    Args:
      fn: Function to be checked.

    Returns:
      fn: The input `fn` passed through unchanged.
    """
    sig = inspect.signature(fn)
    if len(sig.parameters) != 2:
        raise TypeError("`fn` must be a two argument Callable.")
    return fn


def upgrade_symbols(
    symbols: Union[Iterable[sympy.Symbol], tf.Tensor],
    values: Union[tf.Tensor, tf.Variable],
) -> tf.Tensor:
    """Upgrades symbols and checks for correct shape.

    For a QHBM, there must be a value associated with each symbol.  So this
    function checks that the shape of `values` is the same as that of `symbols`.

    Args:
      symbols: Iterable of `sympy.Symbol`s to upgrade.
      values: Values corresponding to the symbols.

    Returns:
      `tf.Tensor` containing the string representations of the input `symbols`.
    """
    if isinstance(symbols, Iterable):
        if not all([isinstance(s, sympy.Symbol) for s in symbols]):
            raise TypeError("Each entry of `symbols` must be `sympy.Symbol`.")
        symbols_partial_upgrade = [str(s) for s in symbols]
        if len(set(symbols_partial_upgrade)) != len(symbols):
            raise ValueError("All entries of `symbols` must be unique.")
        symbols_upgrade = tf.constant(symbols_partial_upgrade, dtype=tf.string)
        if tf.shape(symbols_upgrade) != tf.shape(values):
            raise ValueError("There must be a symbol for every value.")
        return symbols_upgrade
    raise TypeError("`symbols` must be an iterable of `sympy.Symbol`s.")


def upgrade_circuit(circuit: cirq.Circuit, symbols: tf.Tensor) -> tf.Tensor:
    """Upgrades a circuit and confirms all symbols are present.

    Args:
      circuit: Circuit to convert to tensor.
      symbols: Tensor of strings which are the symbols in `circuit`.

    Returns:
      Single entry 1D tensor of strings representing the input `circuit`.
    """
    if not isinstance(circuit, cirq.Circuit):
        raise TypeError(f"`circuit` must be a `cirq.Circuit`, got {type(circuit)}")
    if not isinstance(symbols, tf.Tensor):
        raise TypeError("`symbols` must be a `tf.Tensor`")
    if symbols.dtype != tf.string:
        raise TypeError("`symbols` must have dtype `tf.string`")
    if set(tfq.util.get_circuit_symbols(circuit)) != {
        s.decode("utf-8") for s in symbols.numpy()
    }:
        raise ValueError(
            "`circuit` must contain all and only the parameters in `symbols`."
        )
    if not circuit:
        raise ValueError(
            "Empty circuit not allowed. "
            "Instead, use identities on all unused qubits."
        )
    return tfq.convert_to_tensor([circuit])


class QHBM(tf.Module):
    """Class for working with QHBM models in TFQ."""

    def __init__(
        self, initial_thetas, energy, sampler, initial_phis, phis_symbols, u, name
    ):
        """Initializes a QHBM with all the required parameters."""
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        super().__init__(name)
        self.thetas = upgrade_initial_values(initial_thetas)
        self.energy_function = check_base_function(energy)
        self.sampler_function = check_base_function(sampler)
        self.phis = upgrade_initial_values(initial_phis)
        self.phis_symbols = upgrade_symbols(phis_symbols, self.phis)
        self.u = upgrade_circuit(u, self.phis_symbols)
        self.u_dagger = upgrade_circuit(u ** -1, self.phis_symbols)

        self.raw_qubits = sorted(u.all_qubits())
        self.qubits = tf.constant([[q.row, q.col] for q in self.raw_qubits])
        raw_bit_circuit, raw_bit_symbols = build_bit_circuit(self.raw_qubits, name)
        self.bit_symbols = upgrade_symbols(
            raw_bit_symbols, tf.ones([len(self.raw_qubits)])
        )
        self.bit_and_u = tfq.layers.AddCircuit()(raw_bit_circuit, append=u)

        # Simulator backends
        self.tfq_sample_layer = tfq.layers.Sample()

    def copy(self):
        return QHBM(
            self.thetas,
            self.energy_function,
            self.sampler_function,
            self.phis,
            [sympy.Symbol(s.decode("utf-8")) for s in self.phis_symbols.numpy()],
            tfq.from_tensor(self.u)[0],
            self.name,
        )

    @tf.function
    def sample_bitstrings(self, num_samples):
        """Returns unique bitstrings and counts from this QHBM's classical dist.

        Args:
          num_samples: number of bitstrings to sample from the classical probability
              distribution.

        Returns:
          unique_samples:
          counts: 1D `tf.Tensor` of dtype `tf.int32` such that `counts[i]` is the
            number of times `unique_samples[i]` was sampled from `qhbm_sampler`.
        """
        print(f"retracing: sample_bitstrings from {self.name}")
        samples = self.sampler_function(self.thetas, num_samples)
        unique_samples, _, counts = unique_with_counts(samples)
        return unique_samples, counts

    @tf.function
    def sample_state_circuits(self, num_samples):
        """Returns tensor of circuits generating pure state samples from this QHBM.

        Args:
          num_samples: number of pure state samples to draw from this QHBM.

        Returns:
          circuit_samples: 1-D `tf.Tensor` containing the string representations of
              the circuits to generate pures state samples from this QHBM.
          bitstring_samples:
          counts: 1-D `tf.Tensor` of type `tf.int32` such that `counts[i]` says how
              many times circuit `circuit_samples[i]` occurs in the sum used to
              approximate this QHBM.
        """
        print(f"retracing: sample_state_circuits from {self.name}")
        bitstring_samples, counts = self.sample_bitstrings(num_samples)
        circuit_samples = self.state_circuits(bitstring_samples)
        return circuit_samples, bitstring_samples, counts

    @tf.function
    def state_circuits(self, samples):
        print(f"retracing: state_circuits from {self.name}")
        u_model_concrete = tfq.resolve_parameters(
            self.bit_and_u, self.phis_symbols, tf.expand_dims(self.phis, 0)
        )
        tiled_u_model_concrete = tf.tile(u_model_concrete, [tf.shape(samples)[0]])
        circuit_samples = tfq.resolve_parameters(
            tiled_u_model_concrete, self.bit_symbols, samples
        )
        return circuit_samples

    @tf.function
    def sample_unresolved_state_circuits(self, num_samples):
        """Returns tensor of circuits generating pure state samples from this QHBM.

        Args:
          num_samples: number of pure state samples to draw from this QHBM.

        Returns:
          circuit_samples: 1-D `tf.Tensor` containing the string representations of
              the circuits to generate pures state samples from this QHBM.
          bitstring_samples:
          counts: 1-D `tf.Tensor` of type `tf.int32` such that `counts[i]` says how
              many times circuit `circuit_samples[i]` occurs in the sum used to
              approximate this QHBM.
        """
        print(f"retracing: sample_unresolved_state_circuits from {self.name}")
        # Inject the unique bitstrings
        bitstring_samples, counts = self.sample_bitstrings(num_samples)
        tiled_u_model = tf.tile(self.bit_and_u, [tf.shape(counts)[0]])
        circuit_samples = tfq.resolve_parameters(
            tiled_u_model, self.bit_symbols, bitstring_samples
        )
        return circuit_samples, bitstring_samples, counts

    @tf.function
    def sample_pulled_back_bitstrings(self, circuit_samples, counts):
        """Returns samples from the pulled back data distribution.

        The inputs represent the data density matrix. The inverse of this QHBM's
        unitary is appended to create the set of circuits representing the
        pulled back data density matrix. Then, the requested number of bitstrings
        are sampled from each circuit.

        Args:
          circuit_samples: 1-D `tf.Tensor` of type `tf.string` which contains
            circuits serialized by `tfq.convert_to_tensor`. These represent pure
            state samples from the data density matrix. Each entry should be unique.
          counts: 1-D `tf.Tensor` of type `tf.int32`, must be the same size as
            `circuit_samples`. Contains the number of samples to draw from each
            circuit.

        Returns:
          ragged_samples: `tf.RaggedTensor` of DType `tf.int8` structured such
              that `ragged_samples[i]` contains `counts[i]` bitstrings.
        """
        print(f"retracing: sample_pulled_back_bitstrings for {self.name}")
        pulled_back_circuits = tfq.layers.AddCircuit()(
            circuit_samples,
            append=tf.tile(self.u_dagger, [tf.shape(circuit_samples)[0]]),
        )
        raw_samples = self.tfq_sample_layer(
            pulled_back_circuits,
            symbol_names=self.phis_symbols,
            symbol_values=tf.tile(
                tf.expand_dims(self.phis, 0), [tf.shape(counts)[0], 1]
            ),
            repetitions=tf.expand_dims(tf.math.reduce_max(counts), 0),
        )
        num_samples_mask = tf.cast((tf.ragged.range(counts) + 1).to_tensor(), tf.bool)
        ragged_samples = tf.ragged.boolean_mask(raw_samples, num_samples_mask)
        return ragged_samples

    @tf.function
    def energy_and_energy_grad(self, bitstring):
        """Calculates the energy its derivative given a bitstring.

        Args:
          bitstring: `tf.Tensor` of type `tf.int8` which represents the bitstring.

        Returns:
          energy: The energy of this bitstring against the QHBMs energy function.
          grad: The derivative of `energy` with respect to the `thetas` parameters
            of this QHBM.
        """
        print(f"retracing: energy_and_energy_grad for {self.name}")
        new_dup_thetas = tf.identity(self.thetas)  # Need fresh tensor for gradient
        with tf.GradientTape() as tape:
            tape.watch(new_dup_thetas)
            energy = self.energy_function(new_dup_thetas, bitstring)
        grad = tape.gradient(energy, new_dup_thetas)
        return energy, grad

    @tf.function
    def pulled_back_energy_expectation(self, circuit_samples, circuit_counts):
        """Calculates the average energy of bitstrings from the pulled back dist."""
        print(f"retracing: pulled_back_energy_expectation for {self.name}")
        ragged_samples_pb = self.sample_pulled_back_bitstrings(
            circuit_samples, circuit_counts
        )
        # safe when all circuits have the same number of qubits
        all_samples_pb = ragged_samples_pb.values.to_tensor()
        unique_samples_pb, _, counts_pb = unique_with_counts(all_samples_pb)
        counts_pb = tf.cast(counts_pb, tf.float32)
        e_list = tf.map_fn(
            lambda x: self.energy_function(self.thetas, x),
            unique_samples_pb,
            fn_output_signature=tf.float32,
        )
        e_list_full = tf.multiply(counts_pb, e_list)
        average_energy = tf.divide(tf.reduce_sum(e_list_full), tf.reduce_sum(counts_pb))
        return average_energy


class ExactQHBM(QHBM):
    """Used for QHBMs with exact access to the classical and quantum models."""

    def __init__(
        self, initial_thetas, energy, sampler, initial_phis, phis_symbols, u, name
    ):
        """Initializes an analytic QHBM."""
        super().__init__(
            initial_thetas, energy, sampler, initial_phis, phis_symbols, u, name
        )
        self.all_strings = tf.constant(
            list(itertools.product([0, 1], repeat=len(self.raw_qubits))), dtype=tf.int8
        )
        self.tfq_unitary_layer = tfq.layers.Unitary()
        self.tfq_expectation_layer = tfq.layers.Expectation()

    def copy(self):
        return ExactQHBM(
            self.thetas,
            self.energy_function,
            self.sampler_function,
            self.phis,
            [sympy.Symbol(s.decode("utf-8")) for s in self.phis_symbols.numpy()],
            tfq.from_tensor(self.u)[0],
            self.name,
        )

    @tf.function
    def all_energies(self):
        """Returns the energy of every bitstring spanned by this QHBM."""
        print(f"retracing: all_energies for {self.name}")
        # TODO(martantonio): get code to be nearly as fast but with less memory
        # overhead.  tf.map_fn seems to get too CPU fragmented.
        return tf.vectorized_map(
            lambda x: self.energy_function(self.thetas, x), self.all_strings
        )

    @tf.function
    def log_partition_function(self):
        """Calculates the logarithm of the partition function of this QHBM.

        Returns:
          Scalar `tf.Tensor` of DType `tf.float32` which is the logarithm of the
            partition function of the EBM.
        """
        print(f"retracing: log_partition_function for {self.name}")
        negative_energies = -1.0 * self.all_energies()
        return tf.reduce_logsumexp(negative_energies)

    @tf.function
    def entropy_function(self):
        """Calculates the entropy of this QHBM.

        Returns:
          Scalar `tf.Tensor` of DType `tf.float32` which is the entropy of the EBM.
        """
        print(f"retracing: entropy_function for {self.name}")
        negative_energies = -1.0 * self.all_energies()
        return tfp.distributions.Categorical(logits=negative_energies).entropy()

    @tf.function
    def eigvals(self):
        """Returns the eigenvalues of the density matrix this QHBM represents.

        The eigenvalues are the probabilities of the EBM, which are the exponentials
        of the energies divided by the partition function.
        """
        print(f"retracing: eigvals for {self.name}")
        negative_energies = -1.0 * self.all_energies()
        return tf.math.softmax(negative_energies)

    @tf.function
    def unitary_matrix(self):
        """Returns diagonalizing unitary of the density matrix this QHBM represents.

        From the wikipedia page on diagonalization,
        https://en.wikipedia.org/wiki/Diagonalizable_matrix :
        A is diagonalizable if there exists P such that P^{-1} A P is diagonal.
        P is then the unitary diagonalizing A.

        In the case of QHBMs, we have A = UDU^{-1}, where D is the matrix whose
        diagonal is the list of probabilities, and U is the quantum circuit.
        Thus U is a valid choice for P, since then: P^{-1} UDU^{-1} P = D.
        """
        print(f"retracing: unitary_matrix for {self.name}")
        return self.tfq_unitary_layer(
            self.u,
            symbol_names=self.phis_symbols,
            symbol_values=tf.expand_dims(self.phis, 0),
        ).to_tensor()[0]

    @tf.function
    def eigvecs(self):
        """Returns the list of eigenvectors of this QHBM."""
        return tf.transpose(self.unitary_matrix())

    @tf.function
    def density_matrix(self):
        """Returns the exact density matrix represented by this QHBM."""
        print(f"retracing: density_matrix for {self.name}")
        probabilities = tf.cast(self.eigvals(), tf.complex64)
        unitary = self.unitary_matrix()
        u_mul_probs = tf.multiply(
            unitary,
            tf.tile(tf.expand_dims(probabilities, axis=0), (tf.shape(unitary)[0], 1)),
        )
        return tf.linalg.matmul(u_mul_probs, tf.linalg.adjoint(unitary))

    @tf.function
    def fidelity(self, sigma: tf.Tensor):
        """Calculate the fidelity between a QHBM and a density matrix.

        Args:
          sigma: 2-D `tf.Tensor` of dtype `complex64` representing the right density
            matrix in the fidelity calculation.

        Returns:
          A scalar `tf.Tensor` which is the fidelity between the density matrix
            represented by this QHBM and `sigma`.
        """
        print(f"retracing: fidelity for {self.name}")
        e_rho = tf.cast(self.eigvals(), tf.complex128)
        v_rho = tf.cast(self.unitary_matrix(), tf.complex128)
        sqrt_e_rho = tf.sqrt(e_rho)
        v_rho_sqrt_e_rho = tf.multiply(
            v_rho, tf.tile(tf.expand_dims(sqrt_e_rho, 0), (tf.shape(v_rho)[0], 1))
        )
        rho_sqrt = tf.linalg.matmul(v_rho_sqrt_e_rho, tf.linalg.adjoint(v_rho))
        omega = tf.linalg.matmul(
            tf.linalg.matmul(rho_sqrt, tf.cast(sigma, tf.complex128)), rho_sqrt
        )
        # TODO(zaqqwerty): find convincing proof that omega is hermitian,
        # in order to go back to eigvalsh.
        e_omega = tf.linalg.eigvals(omega)
        return tf.cast(
            tf.math.abs(tf.math.reduce_sum(tf.math.sqrt(e_omega))) ** 2, tf.float32
        )


# TODO(#16)
# class QHBMQ(tf.Module):
#     def __init__(
#         self,
#         p,
#         initial_etas,
#         initial_classical_thetas,
#         eta_theta_symbols,
#         energy,
#         sampler,
#         log_partition,
#         entropy,
#         initial_phis,
#         phis_symbols,
#         u,
#         name,
#     ):
#         """Initialize a QHBM with all the required parameters."""
#         if not isinstance(name, str):
#             raise TypeError("name must be a string")
#         super().__init__(name)

#         self.p = tf.constant(p, dtype=tf.int32)

#         if isinstance(initial_etas, tf.Variable):
#             self.etas = tf.Variable(
#                 tf.cast(initial_etas.read_value(), tf.float32), dtype=tf.float32
#             )
#         elif isinstance(initial_etas, (list, np.ndarray, tf.Tensor)):
#           self.etas = tf.Variable(tf.cast(initial_etas, tf.float32), dtype=tf.float32)
#         else:
#             raise TypeError(
#                 "initial_etas needs to be a numeric type, got {}".format(
#                     type(initial_etas)
#                 )
#             )
#         if self.etas.shape[0] != self.p:
#             raise ValueError("must have as many eta values as trotter steps.")

#         if isinstance(initial_classical_thetas, tf.Variable):
#             self.classical_thetas = tf.Variable(
#                 tf.cast(initial_classical_thetas.read_value(), tf.float32),
#                 dtype=tf.float32,
#             )
#         elif isinstance(initial_classical_thetas, (list, np.ndarray, tf.Tensor)):
#             self.classical_thetas = tf.Variable(
#                 tf.cast(initial_classical_thetas, tf.float32), dtype=tf.float32
#             )
#         else:
#             raise TypeError(
#                 "initial_classical_thetas needs to be a numeric type, got {}".format(
#                     type(initial_classical_thetas)
#                 )
#             )

#         if not isinstance(eta_theta_symbols, (list, tuple)) or not all(
#            [isinstance(s, sympy.Symbol) for row_s in eta_theta_symbols for s in row_s]
#         ):
#             raise TypeError(
#                 "eta_theta_symbols must be a 2-D list or tuple of sympy.Symbols."
#             )
#         self.raw_eta_theta_symbols = copy.deepcopy(eta_theta_symbols)
#         self.eta_theta_symbols = tf.constant(
#             [str(s) for row_s in eta_theta_symbols for s in row_s], dtype=tf.string
#         )
#         if (
#             self.classical_thetas.shape[0] * self.etas.shape[0]
#             != self.eta_theta_symbols.shape[0]
#         ):
#             raise ValueError(
#                 "Due to TFQ limitations, "
#                 "there must be as many eta-theta symbols as etas times thetas."
#             )

#         if not isinstance(energy, collections.Callable):
#             raise TypeError("energy must be a function, got {}".format(type(energy)))
#         self.energy_function = energy
#         if not isinstance(sampler, collections.Callable):
#            raise TypeError("sampler must be a function, got {}".format(type(sampler)))
#         self.sampler_function = sampler
#         if not isinstance(log_partition, collections.Callable):
#             raise TypeError(
#                 "log_partition must be a function, got {}".format(type(log_partition))
#             )
#         self.log_partition_function = log_partition
#         if not isinstance(entropy, collections.Callable):
#            raise TypeError("entropy must be a function, got {}".format(type(entropy)))
#         self.entropy_function = entropy
#         if isinstance(initial_phis, tf.Variable):
#             self.phis = tf.Variable(initial_phis.read_value(), dtype=tf.float32)
#         elif isinstance(initial_phis, (list, np.ndarray, tf.Tensor)):
#             self.phis = tf.Variable(initial_phis, dtype=tf.float32)
#         else:
#             raise TypeError(
#                 "initial_phis needs to be a numeric type, got {}".format(
#                     type(initial_phis)
#                 )
#             )
#         if not isinstance(phis_symbols, (list, tuple)) or not all(
#             [isinstance(s, sympy.Symbol) for s in phis_symbols]
#         ):
#             raise TypeError("phis_symbols must be a list or tuple of sympy.Symbols.")
#         self.raw_phis_symbols = copy.deepcopy(phis_symbols)
#       self.phis_symbols = tf.constant([str(s) for s in phis_symbols], dtype=tf.string)
#         if self.phis.shape != self.phis_symbols.shape:
#             raise ValueError("There must be one symbol per quantum circuit parameter")
#         if not isinstance(u, cirq.Circuit):
#             raise TypeError("u must be a cirq.Circuit, got {}".format(type(u)))
#         self.raw_u = copy.deepcopy(u)
#         if set(tfq.util.get_circuit_symbols(u)) != set(
#             [str(s) for s in self.raw_phis_symbols]
#             + [str(s) for row_s in self.raw_eta_theta_symbols for s in row_s]
#         ):
#             raise ValueError(
#                 "u must contain all and only the symbols "
#                 "in phis_symbols and eta_theta_symbols."
#             )
#         self.u = tfq.convert_to_tensor([u])
#         self.u_dagger = tfq.convert_to_tensor([u ** -1])

#         self.raw_qubits = sorted(u.all_qubits())
#       raw_bit_circuit, raw_bit_symbols = util.build_bit_circuit(self.raw_qubits, name)
#         self.qubits = tf.constant([[q.row, q.col] for q in self.raw_qubits])
#         self.num_qubits = tf.constant(len(self.raw_qubits))
#         self.bit_symbols = tf.constant(
#             [str(s) for s in raw_bit_symbols], dtype=tf.string
#         )
#         self.bit_injector = tfq.convert_to_tensor([raw_bit_circuit])

#     def copy(self):
#         return QHBM(
#             self.p,
#             self.etas,
#             self.classical_thetas,
#             self.eta_theta_symbols,
#             self.energy_function,
#             self.sampler_function,
#             self.log_partition_function,
#             self.entropy_function,
#             self.phis,
#             self.raw_phis_symbols,
#             self.raw_u,
#             self.name,
#         )


# @tf.function
# def get_eta_theta_vals(etas, classical_thetas):
#     """Get the quantum circuit params induced by the classical model."""
#     print("retracing: get_eta_theta_vals")
#     p = tf.shape(etas)[0]
#     num_classical_thetas = tf.shape(classical_thetas)[0]
#     classical_thetas_tiled = tf.tile(tf.expand_dims(classical_thetas, 0), [p, 1])
#     etas_tiled = tf.tile(tf.expand_dims(etas, 1), [1, num_classical_thetas])
#     return tf.reshape(etas_tiled * classical_thetas_tiled, [p * num_classical_thetas])


# @tf.function
# def sample_bitstrings(this_qhbm, num_samples):
#     """Returns unique bitstrings and counts from this QHBM's classical dist.

#     Args:
#       this_qhbm: QHBM representation of a density matrix.
#       num_samples: number of bitstrings to sample from the classical probability
#           distribution parameterizing this QHBM.

#     Returns:
#       unique_samples:
#       counts:
#     """
#     print("retracing: sample_bitstrings")
#     samples = this_qhbm.sampler_function(this_qhbm.classical_thetas, num_samples)
#     unique_samples, _, counts = util.unique_with_counts(samples)
#     return unique_samples, counts


# @tf.function
# def sample_state_circuits(this_qhbm, num_samples):
#     """Returns tensor of circuits generating pure state samples from this QHBM.

#     Args:
#       this_qhbm: QHBM representation of a density matrix.
#       num_samples: number of pure state samples to draw from this QHBM.

#     Returns:
#       circuit_samples: 1-D `tf.Tensor` containing the string representations of
#           the circuits to generate pures state samples from this QHBM.
#       bitstring_samples:
#       counts: 1-D `tf.Tensor` of type `tf.int32` such that `counts[i]` says how
#           many times circuit `circuit_samples[i]` occurs in the sum used to
#           approximate this QHBM.
#     """
#     print("retracing: sample_state_circuits")
#     # Fill in quantum circuit values
#     eta_theta_vals = get_eta_theta_vals(this_qhbm.etas, this_qhbm.classical_thetas)
#     total_vals = tf.concat([eta_theta_vals, this_qhbm.phis], 0)
#    total_symbols = tf.concat([this_qhbm.eta_theta_symbols, this_qhbm.phis_symbols], 0)
#     u_model_concrete = tfq.resolve_parameters(
#         this_qhbm.u, total_symbols, tf.expand_dims(total_vals, 0)
#     )

#     # Inject the unique bitstrings
#     bitstring_samples, counts = sample_bitstrings(this_qhbm, num_samples)
#     tiled_bit_injector = tf.tile(this_qhbm.bit_injector, [tf.shape(counts)[0]])
#     bit_injector_samples = tfq.resolve_parameters(
#         tiled_bit_injector, this_qhbm.bit_symbols, bitstring_samples
#     )
#     circuit_samples = tfq.append_circuit(
#         bit_injector_samples, tf.tile(u_model_concrete, [tf.shape(counts)[0]])
#     )
#     return circuit_samples, bitstring_samples, counts


# @tf.function
# def sample_pulled_back_bitstrings(this_qhbm, circuit_samples, counts):
#     """Returns samples from the pulled back data distribution.

#     The inputs represent the data density matrix. The inverse of this QHBM's
#     unitary is appended to create the set of circuits representing the
#     pulled back data density matrix. Then, the requested number of bitstrings
#     are sampled from each circuit.

#     Args:
#       this_qhbm: QHBM representation of a density matrix.
#     """
#     print("retracing: sample_pulled_back_bitstrings")
#     # Fill in quantum circuit values
#     eta_theta_vals = get_eta_theta_vals(this_qhbm.etas, this_qhbm.classical_thetas)
#     total_vals = tf.concat([eta_theta_vals, this_qhbm.phis], 0)
#    total_symbols = tf.concat([this_qhbm.eta_theta_symbols, this_qhbm.phis_symbols], 0)
#     u_dagger_concrete = tfq.resolve_parameters(
#         this_qhbm.u_dagger, total_symbols, tf.expand_dims(total_vals, 0)
#     )
#     pulled_back_circuits = tfq.append_circuit(
#         circuit_samples, tf.tile(u_dagger_concrete, [tf.shape(circuit_samples)[0]])
#     )
#     raw_samples = tfq.layers.Sample()(
#        pulled_back_circuits, repetitions=tf.expand_dims(tf.math.reduce_max(counts), 0)
#     )
#     num_samples_mask = tf.cast((tf.ragged.range(counts) + 1).to_tensor(), tf.bool)
#     ragged_samples = tf.ragged.boolean_mask(raw_samples, num_samples_mask)
#     return ragged_samples


# @tf.function
# def approximate_density_matrix(this_qhbm, num_samples):
#     """Returns an estimate of the density matrix represented by this QHBM.

#     Args:
#       num_samples: number of pure state samples to use when computing the
#           density matrix represented by this QHBM.

#     Returns:
#       density_matrix: 2-D tensor containing a numeric approximation to the
#           density matrix represented by this QHBM.
#     """
#     print("retracing: approximate_density_matrix")
#     state_circuits, _, counts = sample_state_circuits(this_qhbm, num_samples)
#     return util.circuits_and_counts_to_density_matrix(state_circuits, counts)


# @tf.function
# def energy_and_energy_grad(this_qhbm, bitstring):
#     print("retracing: energy_and_energy_grad")
#     # Need fresh tensor for gradient
#     new_dup_thetas = tf.identity(this_qhbm.classical_thetas)
#     with tf.GradientTape() as tape:
#         tape.watch(new_dup_thetas)
#         this_energy = this_qhbm.energy_function(new_dup_thetas, bitstring)
#     this_grad = tape.gradient(this_energy, new_dup_thetas)
#     return this_energy, this_grad


# @tf.function
# def pulled_back_energy_expectation(this_qhbm, circuit_samples, circuit_counts):
#     """Calculates the average energy of bitstrings from the pulled back dist."""
#     print("retracing: pulled_back_energy_expectation")
#     ragged_samples_pb = sample_pulled_back_bitstrings(
#         this_qhbm, circuit_samples, circuit_counts
#     )
#     # safe when all circuits have the same number of qubits
#     all_samples_pb = ragged_samples_pb.values.to_tensor()
#     unique_samples_pb, _, counts_pb = util.unique_with_counts(all_samples_pb)
#     counts_pb = tf.cast(counts_pb, tf.float32)
#     e_list, _ = tf.map_fn(
#         lambda x: energy_and_energy_grad(this_qhbm, x),
#         unique_samples_pb,
#         fn_output_signature=(tf.float32, tf.float32),
#     )
#     e_list_full = tf.multiply(counts_pb, e_list)
#     average_energy = tf.divide(tf.reduce_sum(e_list_full), tf.reduce_sum(counts_pb))
#     return average_energy
