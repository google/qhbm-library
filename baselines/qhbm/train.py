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
from baselines import architectures
from baselines import utils as baselines_utils
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_quantum as tfq

from tensorboard.plugins.hparams import api as hp

tfb = tfp.bijectors
tfd = tfp.distributions

# Output logging
flags.DEFINE_string("experiment_name", "qhbm_experiment", "")
flags.DEFINE_string("output_dir", "/tmp/qhbm_logs/", "")
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

  if config.dataset.lattice_dimension == 1:
    num_sites = num_rows * num_cols
    qubits = cirq.GridQubit.rect(1, num_sites)
    for i in range(num_sites):
      x_hamiltonian -= bias * cirq.X(qubits[i])
      z_hamiltonian -= cirq.Z(qubits[i]) * cirq.Z(qubits[(i + 1) % num_sites])
    return x_hamiltonian, z_hamiltonian

  elif config.dataset.lattice_dimension == 2:
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


def get_average_correlation_op(config):
  num_rows = config.dataset.num_rows
  num_cols = config.dataset.num_cols

  if config.dataset.lattice_dimension == 1:
    num_sites = num_rows * num_cols
    qubits = cirq.GridQubit.rect(1, num_sites)
    op = cirq.PauliSum()
    for q0, q1 in zip(qubits, qubits[1:]):
      op += (1.0 / (num_sites - 1)) * cirq.Z(q0) * cirq.Z(q1)
    return op
    
  elif config.dataset.lattice_dimension == 2:
    hamiltonian = cirq.PauliSum()
    qubits = get_qubit_grid(num_rows, num_cols)
    extended_qubits = get_qubit_grid(num_rows, num_cols)
    for r, row in enumerate(qubits):
      extended_qubits[r].append(row[0])
    extended_qubits.append(qubits[0])
    count = 0
    # Horizontal interactions.
    for row in extended_qubits[:-1]:
      for q0, q1 in zip(row, row[1:]):
        hamiltonian += cirq.Z(q0) * cirq.Z(q1)
        count += 1
    # Vertical interactions.
    for row_0, row_1 in zip(extended_qubits, extended_qubits[1:]):
      for q0, q1 in zip(row_0[:-1], row_1):
        hamiltonian += cirq.Z(q0) * cirq.Z(q1)
        count += 1
    hamiltonian = hamiltonian / float(count)
    return hamiltonian


def compute_data_point_metrics(average_correlation_op_matrix,
                               beta=None,
                               target_hamiltonian_matrix=None,
                               prev_target_density_matrix=None,
                               channel_matrix=None):
  if beta is not None and target_hamiltonian_matrix is not None:
    target_density_matrix = baselines_utils.np_get_thermal_state(beta, target_hamiltonian_matrix)
    target_log_partition_function = baselines_utils.log_partition_function(
        beta, target_hamiltonian_matrix)
  elif prev_target_density_matrix is not None and channel_matrix is not None:
    target_density_matrix = channel_matrix @ prev_target_density_matrix @ tf.linalg.adjoint(channel_matrix)
  target_correlation = tf.linalg.trace(
      tf.matmul(average_correlation_op_matrix, target_density_matrix))
  target_state_eigvals = tf.linalg.eigvalsh(target_density_matrix)
  target_entropy = -tf.math.reduce_sum(tf.math.multiply_no_nan(tf.math.log(target_state_eigvals), target_state_eigvals))
  if beta is not None and target_hamiltonian_matrix is not None:
    return target_density_matrix, target_correlation, target_entropy, target_log_partition_function
  return target_density_matrix, target_correlation, target_entropy


def get_initial_qhbm(hamiltonian_shards, config, name):
  """Gets initial untrained QHBM."""
  num_sites = config.dataset.num_rows * config.dataset.num_cols
  num_layers = config.hparams.num_layers

  # energy
  energy_initializer = tf.keras.initializers.RandomNormal(
      mean=config.hparams.energy_mean, stddev=config.hparams.energy_stddev)
  energy = models.KOBE(list(range(num_sites)), 2, energy_initializer)
  ebm = inference.AnalyticEnergyInference(
      energy, config.training.samples, name=name)

  # circuit
  if config.dataset.lattice_dimension == 1:
    raw_qubits = cirq.GridQubit.rect(1, num_sites)
  elif config.dataset.lattice_dimension == 2:
    raw_qubits = cirq.GridQubit.rect(config.dataset.num_rows,
                                     config.dataset.num_cols)
  if num_layers == 0:
    pqc = cirq.Circuit(cirq.I(q) for q in raw_qubits)
  else:
    pqc = architectures.get_hardware_efficient_ansatz(
        raw_qubits, num_layers, name)
  circuit_initializer = tf.keras.initializers.RandomNormal(
      mean=config.hparams.circuit_mean, stddev=config.hparams.circuit_stddev)
  if config.training.ansatz == "qhea":
    circuit = models.DirectQuantumCircuit(pqc, circuit_initializer)
  elif config.training.ansatz == "qaia":
    circuit = models.QAIA(hamiltonian_shards,
                          energy.operator_shards(raw_qubits),
                          num_layers, circuit_initializer)
    circuit.value_layers_inputs[0][1].assign(energy.post_process[0].kernel)
  if config.training.qnn == "analytic":
    qnn = inference.AnalyticQuantumInference(circuit, name=name)
  elif config.training.qnn == "sampled":
    qnn = inference.SampledQuantumInference(
        circuit, config.training.samples, name=name)

  qhbm = inference.QHBM(ebm, qnn)
  return qhbm.modular_hamiltonian, qhbm


def get_optimizer(optimizer, learning_rate):
  if optimizer == "SGD":
    return tf.keras.optimizers.SGD(learning_rate=learning_rate)
  elif optimizer == "Adam":
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    

def train_model(qhbm: inference.QHBM,
                modular_hamiltonian: models.Hamiltonian,
                optimizer,
                num_steps,
                target_hamiltonian_shards,
                target_density_matrix,
                metrics_writer,
                config,
                prev_modular_hamiltonian: models.Hamiltonian = None,
                target_hamiltonian: models.Hamiltonian = None,
                beta = None,
                channel = None):
  """Train given model and write metrics on progress."""

  modular_hamiltonian_copy, qhbm_copy = get_initial_qhbm(
      target_hamiltonian_shards, config, "qhbm_copy")

  if channel:
    modular_hamiltonian_copy_2, qhbm_copy_2 = get_initial_qhbm(
        target_hamiltonian_shards, config, "qhbm_copy_2")
    for c, v in zip(modular_hamiltonian_copy_2.variables, prev_modular_hamiltonian.variables):
      c.assign(v)
    channel_circuit = models.DirectQuantumCircuit(channel)
    evolved_circuit = modular_hamiltonian_copy_2.circuit + channel_circuit
    if config.training.qnn == "analytic":
      evolved_qnn = inference.AnalyticQuantumInference(evolved_circuit, name=qhbm_copy.name)
    elif config.training.qnn == "sampled":
      evolved_qnn = inference.SampledQuantumInference(evolved_circuit, config.training.samples, name=qhbm_copy.name)
    evolved_qhbm = inference.QHBM(qhbm_copy_2.e_inference, evolved_qnn)
    evolved_qhbm_data = data.QHBMData(evolved_qhbm)

  def train_step(step):
    with tf.GradientTape() as tape:
      if target_hamiltonian is not None and beta is not None:
        loss = inference.vqt(qhbm, target_hamiltonian, beta)
      else:
        loss = inference.qmhl(evolved_qhbm_data, qhbm)
    grads = tape.gradient(loss, modular_hamiltonian.trainable_variables)
    optimizer.apply_gradients(zip(grads, modular_hamiltonian.trainable_variables))

    with metrics_writer.as_default():
      if config.logging.loss:
        tf.summary.scalar("loss", loss, step=step)

      if config.training.ansatz == "qhea":
        if config.logging.energy_variables:
          tf.summary.histogram(
              "energy_variables", modular_hamiltonian.energy.trainable_variables, step=step)
        if config.logging.circuit_variables:
          tf.summary.histogram(
              "circuit_variables", modular_hamiltonian.circuit.trainable_variables, step=step)
        if config.logging.energy_grads:
          tf.summary.histogram("energy_grads", grads[:-1], step=step)
          tf.summary.scalar(
            "energy_grad_size", tf.reduce_max(tf.abs(grads[:-1])), step=step)
        if config.logging.circuit_grads:
          tf.summary.histogram("circuit_grads", grads[-1:], step=step)
          tf.summary.scalar(
            "circuit_grad_size", tf.reduce_max(tf.abs(grads[-1:])), step=step)

      elif config.training.ansatz == "qaia":
        tf.summary.histogram(
            "thetas_classical", modular_hamiltonian.energy.post_process[0].kernel, step=step)
        tf.summary.histogram(
            "etas", modular_hamiltonian.circuit.value_layers_inputs[0][0], step=step)
        if not config.hparams.tied:
          tf.summary.histogram(
              "thetas_quantum",
              modular_hamiltonian.circuit.value_layers_inputs[0][1],
              step=step)
        tf.summary.histogram(
            "gammas", modular_hamiltonian.circuit.value_layers_inputs[0][2], step=step)

      if step % config.logging.expensive_downsample == 0 or step == num_steps - 1:
        if config.logging.fidelity:
            fidelity = inference.fidelity(modular_hamiltonian,
                                          tf.cast(target_density_matrix, tf.complex64))
            tf.summary.scalar("fidelity", fidelity, step=step)
        
        if config.logging.relative_entropy:
          density_matrix = inference.density_matrix(modular_hamiltonian)
          if channel:
            relative_entropy = baselines_utils.relative_entropy(target_density_matrix, density_matrix)
          else:
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

  n_cores = len(tf.config.list_logical_devices("GPU"))
  logging.info(f"num GPU cores: {n_cores}")

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

  average_correlation_op = get_average_correlation_op(config)
  average_correlation_op_matrix = tf.constant(
      average_correlation_op.matrix(), dtype=tf.complex128)
  average_correlation_op = tfq.convert_to_tensor([average_correlation_op])

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

  for iteration in range(config.hparams.num_iterations):
    modular_hamiltonian, qhbm = get_initial_qhbm(
        target_hamiltonian_shards, config, "qhbm")

    for sequence_step in range(len(iterates)):
      if sequence_step == 0 or (sequence_step == 1 and config.training.loss == "qvartz"):
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
        target_density_matrix, target_correlation, target_entropy, target_log_partition_function = compute_data_point_metrics(
            average_correlation_op_matrix,
            beta=beta_tensor,
            target_hamiltonian_matrix=target_hamiltonian_matrix)
      else:
        evolution_time = sequence_step * config.dataset.total_time / config.dataset.time_steps
        evolution_time = round(evolution_time, config.dataset.digits)
        unitary = iterates[sequence_step]
        unitary_matrix = tf.constant(unitary.unitary(), dtype=tf.complex128)
        target_density_matrix, target_correlation, target_entropy = compute_data_point_metrics(
            average_correlation_op_matrix,
            prev_target_density_matrix=target_density_matrix,
            channel_matrix=unitary_matrix)

      if config.training.loss == "vqt":
        data_point_label = (
            f"beta_{str(beta).replace('.','p')}"
        )
        logging.info(f"Starting experiment: beta: {beta}")
      else:
        data_point_label = (
            f"time_{str(evolution_time).replace('.','p')}"
        )
        logging.info(f"Starting experiment: time: {evolution_time}")
      if iteration == 0:
        data_point_dir = os.path.join(results_dir, "metrics", data_point_label,
                                      "data_point")
        data_point_metrics_writer = tf.summary.create_file_writer(
            data_point_dir)
        with data_point_metrics_writer.as_default():
          step_0 = tf.constant(0, dtype=tf.int64)
          tf.summary.scalar(
              "target_correlation", target_correlation, step=step_0)
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

        if sequence_step > 0 and config.training.initialization == "random":
          modular_hamiltonian, qhbm = get_initial_qhbm(
              target_hamiltonian_shards, config, "qhbm")
          optimizer = get_optimizer(config.training.optimizer,
                                    config.training.learning_rate)

        model_label = f"train_model_iteration_{iteration}"
        model_dir = os.path.join(results_dir, "metrics", data_point_label,
                                 model_label)
        model_metrics_writer = tf.summary.create_file_writer(model_dir)
        initial_t = time.time()

        num_steps = config.training.init_steps if sequence_step == 0 else config.training.num_steps
        if vqt:
          train_model(
              qhbm,
              modular_hamiltonian,
              optimizer,
              num_steps,
              target_hamiltonian_shards,
              target_density_matrix,
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

        total_wall_time = time.time() - initial_t
        logging.info("Finished training. Total min: %.2f",
                     total_wall_time / 60.0)

      summary_writer = tf.summary.create_file_writer(
          os.path.join(results_dir, "hparams"))
      with summary_writer.as_default():
        hp.hparams({
            "qnn_layers":
                config.hparams.num_layers,
            "learning_rate":
                config.training.learning_rate
        })


if __name__ == "__main__":
  app.run(main)
