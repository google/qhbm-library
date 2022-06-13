import json
import os
import time

from absl import app
from absl import flags
from absl import logging
import cirq
from ml_collections.config_flags import config_flags
import numpy as np
from qhbmlib import data
from qhbmlib import models
from qhbmlib import inference
from qhbmlib import utils
from baselines import pqc
from baselines import utils as baselines_utils
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_quantum as tfq

from tensorboard.plugins.hparams import api as hp

tfb = tfp.bijectors
tfd = tfp.distributions

# Output logging
flags.DEFINE_string("experiment_name", "qhbm_experiment", "")
flags.DEFINE_string("output_dir", "/tmp/qhbm_logs/qhbm_experiment", "")
config_flags.DEFINE_config_file("config", "path/to/config",
                                "File containing the training configuration.")

# General model flags.
flags.DEFINE_integer("seed", 42, "Random seed.")

FLAGS = flags.FLAGS


def get_qubit_grid(num_rows, num_cols):
  """Rectangle of qubits returned as a nested list."""
  qubits = []
  for r in range(num_rows):
    qubits.append([cirq.GridQubit(r, c) for c in range(num_cols)])
  return qubits


def get_tfim_hamiltonian(bias, config):
  z_hamiltonian = cirq.PauliSum()
  x_hamiltonian = cirq.PauliSum()
  num_rows = config.dataset.num_rows
  num_cols = config.dataset.num_cols

  if config.dataset.lattice_dim == 1:
    num_sites = num_rows * num_cols
    qubits = cirq.GridQubit.rect(1, num_sites)
    for i in range(num_sites):
      x_hamiltonian -= bias * cirq.X(qubits[i])
      z_hamiltonian -= cirq.Z(qubits[i]) * cirq.Z(qubits[(i + 1) % num_sites])
    return x_hamiltonian, z_hamiltonian

  elif config.dataset.lattice_dim == 2:
    qubits = get_qubit_grid(num_rows, num_cols)
    extended_qubits = get_qubit_grid(num_rows, num_cols)
    for r, row in enumerate(qubits):
      extended_qubits[r].append(row[0])
    extended_qubits.append(qubits[0])
    # Horizontal interactions.
    for row in extended_qubits[:-1]:
      for q0, q1 in zip(row, row[1:]):
        z_hamiltonian -= cirq.Z(q0) * cirq.Z(q1)
    # Vertical interactions.
    for row0, row1 in zip(extended_qubits, extended_qubits[1:]):
      for q0, q1 in zip(row0[:-1], row1):
        z_hamiltonian -= cirq.Z(q0) * cirq.Z(q1)
    for row in qubits:
      for q in row:
        x_hamiltonian -= bias * cirq.X(q)
    return x_hamiltonian, z_hamiltonian


def get_tfim_unitary(x_hamiltonian, z_hamiltonian, config):
  hamiltonian_shards = [x_hamiltonian, z_hamiltonian]
  coefficients = [
        config.dataset.total_time /
        (config.dataset.time_steps * config.dataset.trotter_steps),
        config.dataset.total_time /
        (config.dataset.time_steps * config.dataset.trotter_steps)
    ]
  return tfq.util.exponential(hamiltonian_shards * config.dataset.trotter_steps,
                             coefficients * config.dataset.trotter_steps)


def compute_data_point_metrics(beta=None,
                               target_hamiltonian_matrix=None,
                               prev_target_density_matrix=None,
                               channel_matrix=None):
  if beta is not None and target_hamiltonian_matrix is not None:
    target_density_matrix = baselines_utils.get_thermal_state(beta, target_hamiltonian_matrix)
    target_log_partition_function = baselines_utils.log_partition_function(
        beta, target_hamiltonian_matrix)
  elif prev_target_density_matrix is not None and channel_matrix is not None:
    target_density_matrix = channel_matrix @ prev_target_density_matrix @ tf.linalg.adjoint(channel_matrix)
  target_state_eigvals = tf.linalg.eigvalsh(target_density_matrix)
  target_entropy = -tf.math.reduce_sum(tf.math.multiply_no_nan(tf.math.log(target_state_eigvals), target_state_eigvals))
  if beta is not None and target_hamiltonian_matrix is not None:
    return target_density_matrix, target_entropy, target_log_partition_function
  elif prev_target_density_matrix is not None and channel_matrix is not None:
    return target_density_matrix, target_entropy


def get_initial_qhbm(hamiltonian_shards, config, name):
  """Gets initial untrained QHBM."""
  num_sites = config.dataset.num_rows * config.dataset.num_cols
  num_layers = config.model.circuit_layers

  # energy
  energy_initializer = tf.keras.initializers.RandomNormal(
      mean=config.model.energy_init_mean, stddev=config.model.energy_init_stddev)
  if config.model.energy == "kobe":
    energy = models.KOBE(list(range(num_sites)), config.model.kobe_order, energy_initializer)
  if config.model.ebm == "analytic":
    ebm = inference.AnalyticEnergyInference(
        energy, config.training.num_samples, name=name)

  # circuit
  if config.dataset.lattice_dim == 1:
    raw_qubits = cirq.GridQubit.rect(1, num_sites)
  elif config.dataset.lattice_dim == 2:
    raw_qubits = cirq.GridQubit.rect(config.dataset.num_rows,
                                     config.dataset.num_cols)
  if num_layers == 0:
    u = cirq.Circuit(cirq.I(q) for q in raw_qubits)
  else:
    u = pqc.get_hardware_efficient_model_unitary(
        raw_qubits, num_layers, name)
  circuit_initializer = tf.keras.initializers.RandomNormal(
      mean=config.model.circuit_init_mean, stddev=config.model.circuit_init_stddev)
  if config.model.circuit == "qhea":
    circuit = models.DirectQuantumCircuit(u, circuit_initializer)
  elif config.model.circuit == "qaia":
    circuit = models.QAIA(hamiltonian_shards,
                          energy.operator_shards(raw_qubits),
                          num_layers, circuit_initializer)
    circuit.value_layers_inputs[0][1].assign(energy.post_process[0].kernel)
  if config.model.qnn == "analytic":
    qnn = inference.AnalyticQuantumInference(circuit, name=name)
  elif config.model.qnn == "sampled":
    qnn = inference.SampledQuantumInference(
        circuit, config.training.num_samples, name=name)

  qhbm = inference.QHBM(ebm, qnn)
  return qhbm.modular_hamiltonian, qhbm


def get_optimizer(optimizer, learning_rate):
  if optimizer == "SGD":
    return tf.keras.optimizers.SGD(learning_rate=learning_rate)
  elif optimizer == "Adam":
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def information_matrix(qhbm: inference.QHBM, modular_hamiltonian: models.Hamiltonian,
                       modular_hamiltonian_copy: models.Hamiltonian, config):
  """Estimates the Bogoliubov-Kubo-Mori information matrix.
  Args:
    qhbm: Hamiltonian inference.
    modular_hamiltonian: qhbm model. exp(-modular_hamiltonian)/Z(modular_hamiltonian) = rho.
    modular_hamiltonian_copy: copy of modular_hamiltonian.
    config: config dict.
  Returns:
    The BKM information matrix. This is tr[d_j rho d_k modular_hamiltonian] element-wise
    i.e.
    the Hilbert-Schmidt inner product of a mixture coords tangent vector and
    an exponential coords tangent vector.
  """

  def ebm_block():
    samples = qhbm.e_inference.sample(config.training.num_samples)

    with tf.GradientTape() as tape:
      tape.watch(modular_hamiltonian.energy.trainable_variables[0])
      energies = modular_hamiltonian.energy(samples)
    energy_jac = tape.jacobian(energies, modular_hamiltonian.energy.trainable_variables[0])
    avg_energy_grad = tf.reduce_mean(energy_jac, axis=0)
    centered_energy_jac = energy_jac - avg_energy_grad
    # sample-based approximation of covariance of energy grads.
    return tf.matmul(
        centered_energy_jac, centered_energy_jac,
        transpose_a=True) / config.training.num_samples

  def cross_block():
    shift = tf.constant(0.5)
    scale = tf.constant(np.pi / 2)
    circuit_values = tf.identity(modular_hamiltonian.circuit.trainable_variables[0])

    def grad(indices, updates):
      modular_hamiltonian.circuit.trainable_variables[0].assign(
          tf.tensor_scatter_nd_add(
              circuit_values, indices=indices, updates=updates))

      with tf.GradientTape() as tape:
        tape.watch(modular_hamiltonian_copy.energy.trainable_variables[0])
        expectation = qhbm.expectation(modular_hamiltonian_copy)

      return tape.gradient(expectation,
                           modular_hamiltonian_copy.energy.trainable_variables[0])

    def row(i):
      return scale * (
          grad([[i]], [-shift]) - grad([[i]], [shift]))

    indices = tf.range(tf.shape(modular_hamiltonian.circuit.trainable_variables[0])[0])
    block = tf.map_fn(fn=row, elems=indices, fn_output_signature=tf.float32)
    modular_hamiltonian.circuit.trainable_variables[0].assign(circuit_values)
    return block

  def qnn_block():
    shift = tf.constant(0.5)
    scale = tf.constant(np.pi / 2)
    circuit_values = tf.identity(modular_hamiltonian.circuit.trainable_variables[0])

    def grad(indices, updates):
      modular_hamiltonian.circuit.trainable_variables[0].assign(
          tf.tensor_scatter_nd_add(
              circuit_values, indices=indices, updates=updates))

      with tf.GradientTape() as tape:
        tape.watch(modular_hamiltonian_copy.circuit.trainable_variables[0])
        expectation = qhbm.expectation(modular_hamiltonian_copy)

      return tape.jacobian(expectation,
                           modular_hamiltonian_copy.circuit.trainable_variables[0])

    def row(i):
      return scale * (
          grad([[i]], [-shift]) - grad([[i]], [shift]))

    indices = tf.range(tf.shape(modular_hamiltonian.circuit.trainable_variables[0])[0])
    block = tf.map_fn(fn=row, elems=indices, fn_output_signature=tf.float32)
    modular_hamiltonian.circuit.trainable_variables[0].assign(circuit_values)
    return block

  block_ebm = ebm_block()
  block_cross = tf.squeeze(cross_block())
  block_qnn = tf.squeeze(qnn_block())

  block_upper = tf.concat([block_ebm, tf.transpose(block_cross)], 1)
  block_lower = tf.concat([block_cross, block_qnn], 1)
  im = tf.concat([block_upper, block_lower], 0)
  return (im + tf.transpose(im)) / 2.


def conditional_decorator(dec, condition):

  def decorator(func):
    if condition:
      return dec(func)
    return func

  return decorator


def train_model(qhbm: inference.QHBM,
                modular_hamiltonian: models.Hamiltonian,
                optimizer,
                num_steps,
                target_hamiltonian_shards,
                target_density_matrix,
                metrics_dir,
                metrics_writer,
                config,
                target_hamiltonian: models.Hamiltonian = None,
                beta = None,
                prev_modular_hamiltonian: models.Hamiltonian = None,
                channel = None):
  """Train given model and write metrics on progress."""

  modular_hamiltonian_copy, qhbm_copy = get_initial_qhbm(
      target_hamiltonian_shards, config, "qhbm_copy")

  if prev_modular_hamiltonian is not None and channel is not None:
    modular_hamiltonian_copy_2, qhbm_copy_2 = get_initial_qhbm(
        target_hamiltonian_shards, config, "qhbm_copy_2")
    for c, v in zip(modular_hamiltonian_copy_2.variables, prev_modular_hamiltonian.variables):
      c.assign(v)
    channel_circuit = models.DirectQuantumCircuit(channel)
    evolved_circuit = modular_hamiltonian_copy_2.circuit + channel_circuit
    if config.model.qnn == "analytic":
      evolved_qnn = inference.AnalyticQuantumInference(evolved_circuit, name=qhbm_copy.name)
    elif config.model.qnn == "sampled":
      evolved_qnn = inference.SampledQuantumInference(evolved_circuit, config.training.num_samples, name=qhbm_copy.name)
    evolved_qhbm = inference.QHBM(qhbm_copy_2.e_inference, evolved_qnn)
    evolved_qhbm_data = data.QHBMData(evolved_qhbm)

  grad_idx = 1 if config.model.circuit == "qaia" else -1

  @conditional_decorator(tf.function, config.training.method in ["vanilla", "natural"])
  def train_step(step):
    with tf.GradientTape() as tape:
      if prev_modular_hamiltonian is not None and channel is not None:
        loss = inference.qmhl(evolved_qhbm_data, qhbm)
      elif target_hamiltonian is not None and beta is not None:
        loss = inference.vqt(qhbm, target_hamiltonian, beta)
        
    grads = tape.gradient(loss, modular_hamiltonian.trainable_variables)
    
    if config.training.method == "vanilla":
      optimizer.apply_gradients(zip(grads, modular_hamiltonian.trainable_variables))

    if config.training.method == "natural":
      for c, v in zip(modular_hamiltonian_copy.variables, modular_hamiltonian.variables):
        c.assign(v)

      info_matrix = information_matrix(qhbm, modular_hamiltonian, modular_hamiltonian_copy, config)
      if config.training.info_matrix_eigval_reg:
        eigvals = tf.cast(tf.linalg.eigvalsh(info_matrix), tf.float32)
        min_eigval = tf.reduce_min(eigvals)
        if min_eigval <= config.training.info_matrix_reg:
          reg = config.training.info_matrix_reg + tf.math.abs(tf.math.minimum(min_eigval, 0.0))
        else:
          reg = 0.0
      else:
        reg = config.training.info_matrix_reg
      reg_info_matrix = info_matrix + reg * tf.eye(tf.shape(info_matrix)[0])

      flat_grads = tf.concat([tf.reshape(g, [-1]) for g in grads], 0)
      flat_natural_grads = tf.squeeze(
          tf.linalg.lstsq(
              reg_info_matrix,
              tf.expand_dims(flat_grads, 1),
              l2_regularizer=config.training.lstsq_l2_regularizer,
              fast=config.training.lstsq_fast))
      variable_shapes = [tf.shape(v) for v in modular_hamiltonian.trainable_variables]
      variable_sizes = [tf.size(v) for v in modular_hamiltonian.trainable_variables]
      natural_grads = []
      i = 0
      for size, shape in zip(variable_sizes, variable_shapes):
        natural_grads.append(tf.reshape(flat_natural_grads[i:i + size], shape))
        i += size
      optimizer.apply_gradients(zip(natural_grads, modular_hamiltonian.trainable_variables))

      with metrics_writer.as_default():
        if config.logging.info_matrix:
          tf.summary.histogram("info_matrix", info_matrix, step=step)
          tf.summary.scalar("info_matrix_norm", tf.norm(info_matrix, ord=config.logging.norm_ord), step=step)
          eigvals = tf.cast(tf.linalg.eigvalsh(info_matrix), tf.float32)
          min_eigval = tf.reduce_min(eigvals)
          max_eigval = tf.reduce_max(eigvals)
          tf.summary.histogram("info_matrix_eigvals", eigvals, step=step)
          tf.summary.scalar("info_matrix_min_eigval", min_eigval, step=step)
          tf.summary.scalar("info_matrix_max_eigval", max_eigval, step=step)
          tf.summary.scalar("info_matrix_cond_number", tf.math.abs(max_eigval) / tf.math.abs(min_eigval), step=step)

        if config.logging.reg_info_matrix:
          tf.summary.scalar("reg", reg, step=step)
          tf.summary.histogram("reg_info_matrix", reg_info_matrix, step=step)
          tf.summary.scalar("reg_info_matrix_norm", tf.norm(reg_info_matrix, ord=config.logging.norm_ord), step=step)
          eigvals = tf.cast(tf.linalg.eigvalsh(reg_info_matrix), tf.float32)
          min_eigval = tf.reduce_min(eigvals)
          max_eigval = tf.reduce_max(eigvals)
          tf.summary.histogram("reg_info_matrix_eigvals", eigvals, step=step)
          tf.summary.scalar("reg_info_matrix_min_eigval", min_eigval, step=step)
          tf.summary.scalar("reg_info_matrix_max_eigval", max_eigval, step=step)
          tf.summary.scalar("reg_info_matrix_cond_number", tf.math.abs(max_eigval) / tf.math.abs(min_eigval), step=step)

        if config.logging.natural_grads:
          flat_natural_energy_grads = tf.concat([tf.reshape(g, [-1]) for g in natural_grads[:grad_idx]], 0)
          flat_natural_circuit_grads = tf.concat([tf.reshape(g, [-1]) for g in natural_grads[grad_idx:]], 0)
          tf.summary.histogram("natural_grads", flat_natural_grads, step=step)
          tf.summary.histogram("natural_energy_grads", flat_natural_energy_grads, step=step)
          tf.summary.histogram("natural_circuit_grads", flat_natural_circuit_grads, step=step)
          tf.summary.scalar("natural_grad_norm", tf.norm(flat_natural_grads, ord=config.logging.norm_ord), step=step)
          tf.summary.scalar("natural_energy_grad_norm", tf.norm(flat_natural_energy_grads, ord=config.logging.norm_ord), step=step)
          tf.summary.scalar("natural_circuit_grad_norm", tf.norm(flat_natural_circuit_grads, ord=config.logging.norm_ord), step=step)

    if config.training.method == "mirror":
      step_metrics_dir = os.path.join(metrics_dir, f"train_step_{step}")
      step_metrics_writer = tf.summary.create_file_writer(step_metrics_dir)

      for c, v in zip(modular_hamiltonian_copy.trainable_variables,
                      modular_hamiltonian.trainable_variables):
        c.assign(v)

      @tf.function
      def train_inner_step(inner_step):
        with tf.GradientTape(persistent=True) as tape:
          energy_inner_prod = tf.reduce_sum([
              tf.reduce_sum(v * g)
              for v, g in zip(modular_hamiltonian.energy.trainable_variables, grads[:grad_idx])
          ])
          circuit_inner_prod = tf.reduce_sum([
              tf.reduce_sum(v * g)
              for v, g in zip(modular_hamiltonian.circuit.trainable_variables, grads[grad_idx:])
          ])
          inner_prod = energy_inner_prod + circuit_inner_prod

          div = inference.vqt(qhbm, modular_hamiltonian_copy, 1.0)
          euclidean_div = 0.5 * tf.reduce_sum([
              tf.reduce_sum((v - c)**2) for v, c in zip(
                  modular_hamiltonian.trainable_variables, modular_hamiltonian_copy.trainable_variables)
          ])

          inner_loss = inner_prod + 1. / config.training.learning_rate * (
              (1. - config.training.euclidean_div_factor) * div +
              config.training.euclidean_div_factor * euclidean_div)

        inner_loss_grads = tape.gradient(inner_loss,
                                         modular_hamiltonian.trainable_variables)
        optimizer.apply_gradients(
            zip(inner_loss_grads, modular_hamiltonian.trainable_variables))
    
        with step_metrics_writer.as_default():
          if config.logging.inner_loss:
            tf.summary.scalar("inner_loss", inner_loss, step=inner_step)

          if config.logging.inner_prod:
            tf.summary.scalar("inner_prod", inner_prod, step=inner_step)
            tf.summary.scalar(
                "energy_inner_prod", energy_inner_prod, step=inner_step)
            tf.summary.scalar(
                "circuit_inner_prod", circuit_inner_prod, step=inner_step)

          if config.logging.div:
            tf.summary.scalar("div", div, step=inner_step)
            tf.summary.scalar("euclidean_div", euclidean_div, step=inner_step)

          if config.logging.inner_loss_grads:
            flat_inner_loss_grads = tf.concat([tf.reshape(g, [-1]) for g in inner_loss_grads], 0)
            flat_inner_loss_energy_grads = tf.concat([tf.reshape(g, [-1]) for g in inner_loss_grads[:grad_idx]], 0)
            flat_inner_loss_circuit_grads = tf.concat([tf.reshape(g, [-1]) for g in inner_loss_grads[grad_idx:]], 0)
            tf.summary.histogram("inner_loss_grads", flat_inner_loss_grads, step=inner_step)
            tf.summary.histogram("inner_loss_energy_grads", flat_inner_loss_energy_grads, step=inner_step)
            tf.summary.histogram("inner_loss_circuit_grads", flat_inner_loss_circuit_grads, step=inner_step)
            tf.summary.scalar(
            "inner_loss_grad_norm", tf.norm(flat_inner_loss_grads, ord=config.logging.norm_ord), step=step)
            tf.summary.scalar(
            "inner_loss_energy_grad_norm", tf.norm(flat_inner_loss_energy_grads, ord=config.logging.norm_ord), step=step)
            tf.summary.scalar(
            "inner_loss_circuit_grad_norm", tf.norm(flat_inner_loss_circuit_grads, ord=config.logging.norm_ord), step=step)          

          if config.logging.variables:
            flat_variables = tf.concat([tf.reshape(v, [-1]) for v in modular_hamiltonian.trainable_variables], 0)
            flat_energy_variables = tf.concat([tf.reshape(v, [-1]) for v in modular_hamiltonian.energy.trainable_variables], 0)
            flat_circuit_variables = tf.concat([tf.reshape(v, [-1]) for v in modular_hamiltonian.circuit.trainable_variables], 0)
            tf.summary.histogram("variables", flat_variables, step=inner_step)
            tf.summary.histogram(
                "energy_variables", flat_energy_variables, step=inner_step)
            tf.summary.histogram(
                "circuit_variables", flat_circuit_variables, step=inner_step)
  
      for inner_step in tf.range(config.training.num_inner_steps, dtype=tf.int64):
        train_inner_step(inner_step)
    
    with metrics_writer.as_default():
      if config.logging.loss:
        tf.summary.scalar("loss", loss, step=step)

      if config.logging.variables:
        flat_variables = tf.concat([tf.reshape(v, [-1]) for v in modular_hamiltonian.trainable_variables], 0)
        flat_energy_variables = tf.concat([tf.reshape(v, [-1]) for v in modular_hamiltonian.energy.trainable_variables], 0)
        flat_circuit_variables = tf.concat([tf.reshape(v, [-1]) for v in modular_hamiltonian.circuit.trainable_variables], 0)
        tf.summary.histogram(
              "variables", flat_variables, step=step)
        tf.summary.histogram(
            "energy_variables", flat_energy_variables, step=step)
        tf.summary.histogram(
            "circuit_variables", flat_circuit_variables, step=step)

      if config.logging.grads:
        flat_grads = tf.concat([tf.reshape(g, [-1]) for g in grads], 0)
        flat_energy_grads = tf.concat([tf.reshape(g, [-1]) for g in grads[:grad_idx]], 0)
        flat_circuit_grads = tf.concat([tf.reshape(g, [-1]) for g in grads[grad_idx:]], 0)
        tf.summary.histogram("grads", flat_grads, step=step)
        tf.summary.histogram("energy_grads", flat_energy_grads, step=step)
        tf.summary.histogram("circuit_grads", flat_circuit_grads, step=step)
        tf.summary.scalar(
            "grad_norm", tf.norm(flat_grads, ord=config.logging.norm_ord), step=step)
        tf.summary.scalar(
        "energy_grad_norm", tf.norm(flat_energy_grads, ord=config.logging.norm_ord), step=step)
        tf.summary.scalar(
        "circuit_grad_norm", tf.norm(flat_circuit_grads, ord=config.logging.norm_ord), step=step)
      
      if step % config.logging.expensive_downsample == 0 or step == num_steps - 1:
        if config.logging.fidelity:
            fidelity = inference.fidelity(modular_hamiltonian,
                                          tf.cast(target_density_matrix, tf.complex64))
            tf.summary.scalar("fidelity", fidelity, step=step)
        
        if config.logging.relative_entropy:
          density_matrix = inference.density_matrix(modular_hamiltonian)
          if prev_modular_hamiltonian is not None and channel is not None:
            relative_entropy = baselines_utils.relative_entropy(target_density_matrix, density_matrix)
          elif target_hamiltonian is not None and beta is not None:
            relative_entropy = baselines_utils.relative_entropy(density_matrix, target_density_matrix)
          tf.summary.scalar(
                "relative_entropy", relative_entropy, step=step)

        if config.logging.density_matrix:
          density_matrix_image = baselines_utils.density_matrix_to_image(
              inference.density_matrix(modular_hamiltonian))
          tf.summary.image("density_matrix", density_matrix_image, step=step)

  for step in tf.range(num_steps, dtype=tf.int64):
    train_step(step)


def main(argv):
  del argv  # unused arg

  seed = FLAGS.seed
  logging.info(f"seed: {seed}")
  tf.random.set_seed(seed)

  output_dir = os.environ.get('AIP_TENSORBOARD_LOG_DIR', FLAGS.output_dir)
  results_dir = os.path.join(output_dir, "results")
  logging.info("\n\n\n\n\n\n\n\nBEGIN MAIN")
  logging.info(f"\n\nSaving results to: {results_dir}\n\n")
  tf.io.gfile.makedirs(results_dir)

  config = FLAGS.config
  with tf.io.gfile.GFile(os.path.join(results_dir, "config.json"), "w") as outfile:
    json.dump(config.to_dict(), outfile)

  bias = round(config.dataset.bias, config.dataset.digits)
  x_hamiltonian, z_hamiltonian = get_tfim_hamiltonian(bias, config)
  target_hamiltonian_shards = [x_hamiltonian, z_hamiltonian]
  target_hamiltonian = x_hamiltonian + z_hamiltonian
  target_hamiltonian_matrix = tf.constant(target_hamiltonian.matrix(), dtype=tf.complex128)
  target_hamiltonian = tfq.convert_to_tensor([target_hamiltonian])

  if config.training.loss == "vqt":
    betas = np.linspace(config.dataset.beta_min, config.dataset.beta_max,
                        config.dataset.beta_steps)
    betas = [round(beta, config.dataset.digits) for beta in betas]
    iterates = betas
  elif config.training.loss == "qvartz":
    unitaries = [get_tfim_unitary(x_hamiltonian, z_hamiltonian, config)] * config.dataset.time_steps
    iterates = [round(config.dataset.beta, config.dataset.digits)] + unitaries

  for trial in range(config.training.num_trials):
    modular_hamiltonian, qhbm = get_initial_qhbm(
        target_hamiltonian_shards, config, "qhbm")

    for sequence_step in range(len(iterates)):
      if sequence_step == 0 or (sequence_step == 1 and config.training.loss == "qvartz"):
        if config.training.method == "mirror":
          optimizer = get_optimizer(config.training.optimizer,
                                    config.training.inner_learning_rate)
        else:
          optimizer = get_optimizer(config.training.optimizer,
                                    config.training.learning_rate)

      vqt = config.training.loss == "vqt" or (
          config.training.loss == "qvartz" and sequence_step == 0)
      if vqt:
        beta = iterates[sequence_step]
        beta_tensor = tf.constant(beta, dtype=tf.float32)
        if config.training.loss == "qvartz":
          evolution_time = 0.0
          evolution_time = round(evolution_time, config.dataset.digits)
        target_density_matrix, target_entropy, target_log_partition_function = compute_data_point_metrics(
            beta=beta_tensor,
            target_hamiltonian_matrix=target_hamiltonian_matrix)
      else:
        evolution_time = sequence_step * config.dataset.total_time / config.dataset.time_steps
        evolution_time = round(evolution_time, config.dataset.digits)
        unitary = iterates[sequence_step]
        unitary_matrix = tf.constant(unitary.unitary(), dtype=tf.complex128)
        target_density_matrix, target_entropy = compute_data_point_metrics(
            prev_target_density_matrix=target_density_matrix,
            channel_matrix=unitary_matrix)

      if config.training.loss == "vqt":
        data_point_label = (
            f"beta_{str(beta).replace('.','p')}"
        )
        logging.info(f"Starting experiment: beta = {beta}")
      else:
        data_point_label = (
            f"time_{str(evolution_time).replace('.','p')}"
        )
        logging.info(f"Starting experiment: time = {evolution_time}")
      if trial == 0:
        data_point_dir = os.path.join(results_dir, "metrics", data_point_label,
                                      "data_point")
        data_point_metrics_writer = tf.summary.create_file_writer(
            data_point_dir)
        with data_point_metrics_writer.as_default():
          step_0 = tf.constant(0, dtype=tf.int64)
          tf.summary.scalar("target_entropy", target_entropy, step=step_0)
          if vqt:
            tf.summary.scalar(
                "target_log_partition_function",
                target_log_partition_function,
                step=step_0)
          if config.logging.density_matrix:
            target_density_matrix_image = baselines_utils.density_matrix_to_image(target_density_matrix)
            tf.summary.image("target_density_matrix", target_density_matrix_image, step=step_0)

      # Training loop
      if config.training.train:
        if not vqt:
          prev_modular_hamiltonian, _ = get_initial_qhbm(
              target_hamiltonian_shards, config, "prev_modular_hamiltonian")
          for c, v in zip(prev_modular_hamiltonian.variables, modular_hamiltonian.variables):
            c.assign(v)

        if sequence_step > 0 and config.training.seq_init == "random":
          modular_hamiltonian, qhbm = get_initial_qhbm(
              target_hamiltonian_shards, config, "qhbm")
          if config.training.method == "mirror":
            optimizer = get_optimizer(config.training.optimizer,
                                      config.training.inner_learning_rate)
          else:
            optimizer = get_optimizer(config.training.optimizer,
                                      config.training.learning_rate)

        model_label = f"train_model_trial_{trial}"
        model_dir = os.path.join(results_dir, "metrics", data_point_label,
                                 model_label)
        model_metrics_writer = tf.summary.create_file_writer(model_dir)
        initial_time = time.time()

        num_steps = config.training.init_steps if sequence_step == 0 else config.training.num_steps
        if vqt:
          train_model(
              qhbm,
              modular_hamiltonian,
              optimizer,
              num_steps,
              target_hamiltonian_shards,
              target_density_matrix,
              model_dir,
              model_metrics_writer,
              config,
              target_hamiltonian=target_hamiltonian,
              beta=beta_tensor)
        else:
          train_model(
              qhbm,
              modular_hamiltonian,
              optimizer,
              num_steps,
              target_hamiltonian_shards,
              target_density_matrix,
              model_dir,
              model_metrics_writer,
              config,
              prev_modular_hamiltonian=prev_modular_hamiltonian,
              channel=unitary)

        with model_metrics_writer.as_default():
          if vqt:
            target_loss = -target_log_partition_function
          else:
            target_loss = target_entropy
          tf.summary.scalar("target_loss", target_loss, step=num_steps-1)

        total_wall_time = time.time() - initial_time
        logging.info("Finished training. Total min: %.2f",
                     total_wall_time / 60.0)

      hparams_writer = tf.summary.create_file_writer(
          os.path.join(results_dir, "hparams"))
      with hparams_writer.as_default():
        hp.hparams({
            "loss":
                config.training.loss,
            "method":
                config.training.method,
            "optimizer":
                config.training.optimizer,
            "seq_init":
                config.training.seq_init
        })


if __name__ == "__main__":
  app.run(main)
