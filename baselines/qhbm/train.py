"""Train using geometric regularization."""

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
flags.DEFINE_string("experiment_name", "ising_ring", "")
flags.DEFINE_string("output_dir", "/tmp/qhbm_logs/", "")
config_flags.DEFINE_config_file("config", "path/to/config",
                                "File containing the training configuration.")
flags.DEFINE_bool("debug", True, "")

# General model flags.
flags.DEFINE_integer("seed", 42, "Random seed.")

# Accelerator flags.
flags.DEFINE_bool("force_use_cpu", False, "If True, force usage of CPU")
flags.DEFINE_bool("use_gpu", False, "Whether to run on GPU or otherwise TPU.")
flags.DEFINE_bool("use_bfloat16", False, "Whether to use mixed precision.")
flags.DEFINE_integer("num_cores", 8, "Number of TPU cores or number of GPUs.")
flags.DEFINE_string(
    "tpu", None,
    "Name of the TPU. Only used if force_use_cpu and use_gpu are both False.")

FLAGS = flags.FLAGS


def _qubit_grid(rows, cols):
  """Rectangle of qubits returned as a nested list."""
  qubits = []
  for r in range(rows):
    qubits.append([cirq.GridQubit(r, c) for c in range(cols)])
  return qubits


def get_tfim_hamiltonian(bias, config):
  z_hamiltonian = cirq.PauliSum()
  x_hamiltonian = cirq.PauliSum()
  if config.dataset.lattice_dimension == 1:
    num_sites = config.dataset.num_rows * config.dataset.num_cols
    qubits = cirq.GridQubit.rect(1, num_sites)
    for i in range(num_sites):
      x_hamiltonian -= bias * cirq.X(qubits[i])
      z_hamiltonian -= cirq.Z(qubits[i]) * cirq.Z(qubits[(i + 1) % num_sites])
    return x_hamiltonian, z_hamiltonian
  elif config.dataset.lattice_dimension == 2:
    num_rows = config.dataset.num_rows
    num_cols = config.dataset.num_cols
    qubits = _qubit_grid(num_rows, num_cols)
    extended_qubits = _qubit_grid(num_rows, num_cols)
    for r, row in enumerate(qubits):
      extended_qubits[r].append(row[0])
    extended_qubits.append(qubits[0])
    # Horizontal interactions.
    for row in extended_qubits[:-1]:
      for q0, q1 in zip(row, row[1:]):
        z_hamiltonian -= cirq.Z(q0) * cirq.Z(q1)
    # Vertical interactions.
    for row_0, row_1 in zip(extended_qubits, extended_qubits[1:]):
      for q0, q1 in zip(row_0[:-1], row_1):
        z_hamiltonian -= cirq.Z(q0) * cirq.Z(q1)
    for row in qubits:
      for q in row:
        x_hamiltonian -= bias * cirq.X(q)
    return x_hamiltonian, z_hamiltonian


def get_tfim_unitaries(h_x, h_z, config):
  if config.dataset.coefficents == "gp":
    time_points = tf.linspace(0, config.dataset.total_time,
                              config.dataset.time_steps + 1)[:-1]
    x_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        config.dataset.x_amp, config.dataset.x_len)
    z_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        config.dataset.z_amp, config.dataset.z_len)
    x_gp = tfp.distributions.GaussianProcess(x_kernel,
                                             tf.expand_dims(time_points, -1))
    z_gp = tfp.distributions.GaussianProcess(z_kernel,
                                             tf.expand_dims(time_points, -1))
    x_noise = x_gp.sample() + 1.0
    z_noise = z_gp.sample() + 1.0
    x_noise = x_noise.numpy()
    z_noise = z_noise.numpy()
  else:
    x_noise = np.ones(config.dataset.time_steps)
    z_noise = np.ones(config.dataset.time_steps)
  h_shards = [h_x, h_z]
  unitaries = []
  for i in range(config.dataset.time_steps):
    coefficients = [
        config.dataset.total_time /
        (config.dataset.time_steps * config.dataset.trotter_steps) * x_noise[i],
        config.dataset.total_time /
        (config.dataset.time_steps * config.dataset.trotter_steps) * z_noise[i]
    ]
    unitaries.append(
        tfq.util.exponential(h_shards * config.dataset.trotter_steps,
                             coefficients * config.dataset.trotter_steps))
  return unitaries


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
    qubits = _qubit_grid(num_rows, num_cols)
    extended_qubits = _qubit_grid(num_rows, num_cols)
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


def compute_data_point_metrics(average_correlation_op_m_t,
                               beta_t=None,
                               target_h_m_t=None,
                               init_state=None,
                               channel_m_t=None):
  if beta_t is not None and target_h_m_t is not None:
    target_state = baselines_utils.np_get_thermal_state(beta_t, target_h_m_t)
    true_log_partition_function = baselines_utils.log_partition_function(
        beta_t, target_h_m_t)
  elif init_state is not None and channel_m_t is not None:
    target_state = channel_m_t @ init_state @ tf.linalg.adjoint(channel_m_t)
  true_correlation = tf.linalg.trace(
      tf.matmul(average_correlation_op_m_t, target_state))
  rho_eigs = tf.linalg.eigvalsh(target_state)
  rho_prod = tf.math.multiply_no_nan(tf.math.log(rho_eigs), rho_eigs)
  true_entropy = -tf.math.reduce_sum(rho_prod)
  if beta_t is not None and target_h_m_t is not None:
    return target_state, true_correlation, true_entropy, true_log_partition_function
  return target_state, true_correlation, true_entropy


@tf.custom_gradient
def straight_through_sigmoid(x):
  y = tf.sigmoid(x)

  def grad(upstream):
    return upstream

  return y, grad


@tf.custom_gradient
def straight_through_sin(x):
  y = tf.math.sin(x)

  def grad(upstream):
    return upstream

  return y, grad


def get_initial_hamiltonian_and_inference(hamiltonian_shards, config, name):
  """Gets initial untrained QHBM."""
  num_sites = config.dataset.num_rows * config.dataset.num_cols
  num_model_layers = config.hparams.p

  # energy
  ebm_initializer = tf.keras.initializers.RandomNormal(
      mean=config.hparams.ebm_mean, stddev=config.hparams.ebm_stddev)
  energy = models.KOBE(list(range(num_sites)), 2, ebm_initializer)
  e_infer = inference.AnalyticEnergyInference(
      energy, config.training.samples, name=name)

  # circuit
  if config.dataset.lattice_dimension == 1:
    raw_qubits = cirq.GridQubit.rect(1, num_sites)
  elif config.dataset.lattice_dimension == 2:
    raw_qubits = cirq.GridQubit.rect(config.dataset.num_rows,
                                     config.dataset.num_cols)
  else:
    raise ValueError(
        f"lattice_dimension {config.dataset.lattice_dimension} is not supported"
    )

  if num_model_layers == 0:
    u = cirq.Circuit(cirq.I(q) for q in raw_qubits)
  else:
    u = architectures.get_hardware_efficient_model_unitary(
        raw_qubits, num_model_layers, name)
  qnn_initializer = tf.keras.initializers.RandomNormal(
      mean=config.hparams.qnn_mean, stddev=config.hparams.qnn_stddev)
  if config.training.ansatz == "qhea":
    circuit = models.DirectQuantumCircuit(u, qnn_initializer)
  elif config.training.ansatz == "qaia":
    circuit = models.QAIA(hamiltonian_shards,
                          energy.operator_shards(raw_qubits),
                          num_model_layers, qnn_initializer)
    circuit.value_layers_inputs[0][1].assign(energy.post_process[0].kernel)

  if config.training.quantum_inference == "analytic":
    q_infer = inference.AnalyticQuantumInference(circuit, name=name)
  elif config.training.quantum_inference == "sampled":
    q_infer = inference.SampledQuantumInference(
        circuit, config.training.samples, name=name)
  qhbm = inference.QHBM(e_infer, q_infer)
  return qhbm.modular_hamiltonian, qhbm


def get_optimizer(optimizer, learning_rate):
  if optimizer == "SGD":
    return tf.keras.optimizers.SGD(learning_rate=learning_rate)
  elif optimizer == "Adam":
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)
  else:
    raise ValueError(f"optimizer {optimizer} is not supported")


def conditional_decorator(dec, condition):

  def decorator(func):
    if condition:
      return dec(func)
    return func

  return decorator


def train_model(model_inf: inference.QHBM,
                model_h: models.Hamiltonian,
                target_h_shards,
                target_dm,
                metrics_dir,
                metrics_writer,
                config,
                optimizer,
                num_steps,
                method,
                model_h_to_propagate: models.Hamiltonian = None,
                target_h: models.Hamiltonian = None,
                beta=None,
                channel=None):
  """Train given model and write metrics on progress."""

  model_h_copy, model_inf_copy = get_initial_hamiltonian_and_inference(
      target_h_shards, config, "model_h_copy")

  if channel:
    model_h_copy_2, model_inf_copy_2 = get_initial_hamiltonian_and_inference(
        target_h_shards, config, "model_h_copy_2")
    for c, m in zip(model_h_copy_2.variables, model_h_to_propagate.variables):
      c.assign(m)
    channel_circuit = models.DirectQuantumCircuit(channel)
    new_circuit = model_h_copy_2.circuit + channel_circuit
    if config.training.quantum_inference == "analytic":
      new_q_infer = inference.AnalyticQuantumInference(new_circuit, name=model_inf_copy.name)
    elif config.training.quantum_inference == "sampled":
      new_q_infer = inference.SampledQuantumInference(new_circuit, config.training.samples, name=model_inf_copy.name)
    data_qhbm = inference.QHBM(model_inf_copy_2.e_inference, new_q_infer)
    q_data = data.QHBMData(data_qhbm)

  @conditional_decorator(tf.function, method in ["vanilla"])
  def train_step(s):
    with tf.GradientTape() as tape:
      if target_h is not None and beta is not None:
        loss = inference.vqt(model_inf, target_h, beta)
      else:
        loss = inference.qmhl(q_data, model_inf)
    grads = tape.gradient(loss, model_h.trainable_variables)

    with metrics_writer.as_default():
      if config.logging.loss:
        tf.summary.scalar("loss", loss, step=s)
      if config.logging.fidelity:
        if s % config.logging.expensive_downsample == 0 or s == num_steps - 1:
          fidelity = inference.fidelity(model_h,
                                        tf.cast(target_dm, tf.complex64))
          tf.summary.scalar("fidelity", fidelity, step=s)

    if method == "vanilla":
      optimizer.apply_gradients(zip(grads, model_h.trainable_variables))

    with metrics_writer.as_default():
      if config.training.ansatz == "qhea":
        if config.logging.thetas:
          tf.summary.histogram(
              "thetas", model_h.energy.trainable_variables, step=s)
        if config.logging.phis:
          tf.summary.histogram(
              "phis", model_h.circuit.trainable_variables, step=s)
        if config.logging.thetas_grads:
          tf.summary.histogram("thetas_grads", grads[:-1], step=s)
        if config.logging.phis_grads:
          tf.summary.histogram("phis_grads", grads[-1:], step=s)
        tf.summary.histogram(
            "theta_loss_grad_size", tf.reduce_max(tf.abs(grads[:-1])), step=s)
        tf.summary.histogram(
            "phi_loss_grad_size", tf.reduce_max(tf.abs(grads[-1:])), step=s)
      elif config.training.ansatz == "qaia":
        tf.summary.histogram(
            "thetas_classical", model_h.energy.post_process[0].kernel, step=s)
        tf.summary.histogram(
            "etas", model_h.circuit.value_layers_inputs[0][0], step=s)
        if not config.hparams.tied:
          tf.summary.histogram(
              "thetas_quantum",
              model_h.circuit.value_layers_inputs[0][1],
              step=s)
        tf.summary.histogram(
            "gammas", model_h.circuit.value_layers_inputs[0][2], step=s)
      if config.logging.density_matrix:
        if s % config.logging.expensive_downsample == 0 or s == num_steps - 1:
          density_matrix = baselines_utils.density_matrix_to_image(
              inference.density_matrix(model_h))
          tf.summary.image("density_matrix", density_matrix, step=s)
          if channel:
            density_matrix = baselines_utils.density_matrix_to_image(
                inference.density_matrix(data_qhbm.modular_hamiltonian))
            tf.summary.image("true_density_matrix", density_matrix, step=s)

    return loss

  for s in tf.range(num_steps, dtype=tf.int64):
    loss = train_step(s)

  return s


def main(argv):
  del argv  # unused arg

  # seed = random.randint(0, 1e10)
  seed = FLAGS.seed
  logging.info(f"using seed: {seed}")
  tf.random.set_seed(seed)

  # ========================================================================== #
  # Get config.
  # ========================================================================== #
  output_dir = os.environ.get('AIP_TENSORBOARD_LOG_DIR', FLAGS.output_dir)
  print(f"output_dir: {output_dir}")
  results_dir = os.path.join(output_dir, "results")
  n_cores = len(tf.config.list_logical_devices("GPU"))
  logging.info(f"num GPU cores: {n_cores}")

  logging.info("\n\n\n\n\n\n\n\nBEGIN MAIN")
  config = FLAGS.config
  logging.info(f"\n\nSaving results to: {results_dir}\n\n")
  tf.io.gfile.makedirs(results_dir)

  with tf.io.gfile.GFile(os.path.join(results_dir, "config.json"), "w") as outfile:
    json.dump(config.to_dict(), outfile)

  average_correlation_op = get_average_correlation_op(config)
  average_correlation_op_m = average_correlation_op.matrix()
  average_correlation_op_m_t = tf.constant(
      average_correlation_op_m, dtype=tf.complex128)
  average_correlation_op_t = tfq.convert_to_tensor([average_correlation_op])

  finished_profile = False
  float_bias = config.dataset.bias
  bias = round(float_bias, config.dataset.digits)
  x_ham, z_ham = get_tfim_hamiltonian(bias, config)
  target_h_shards = [x_ham, z_ham]
  target_h = x_ham + z_ham
  target_h_m = target_h.matrix()
  target_h_m_t = tf.constant(target_h_m, dtype=tf.complex128)
  target_h_t = tfq.convert_to_tensor([target_h])
  if config.training.loss == "vqt":
    betas = np.linspace(config.dataset.beta_min, config.dataset.beta_max,
                        config.dataset.beta_steps)
    betas = [round(beta, config.dataset.digits) for beta in betas]
    iterates = betas
  elif config.training.loss == "qvartz":
    unitaries = get_tfim_unitaries(x_ham, z_ham, config)
    iterates = [round(config.dataset.beta, config.dataset.digits)] + unitaries
  for iteration in range(config.hparams.max_iterations):
    model_h, model_inf = get_initial_hamiltonian_and_inference(
        target_h_shards, config, "model_h")
    for sequence_step in range(len(iterates)):
      if sequence_step == 0:
        optimizer = get_optimizer(config.training.optimizer,
                                  config.training.learning_rate)
      elif sequence_step == 1 and config.training.loss == "qvartz":
        optimizer = get_optimizer(config.training.optimizer,
                                  config.training.learning_rate)
      is_vqt = config.training.loss == "vqt" or (
          config.training.loss == "qvartz" and sequence_step == 0)
      if is_vqt:
        beta = iterates[sequence_step]
        beta_t = tf.constant(beta, dtype=tf.float32)
        if config.training.loss == "qvartz":
          evolution_time = 0.0
          evolution_time = round(evolution_time, config.dataset.digits)
      else:
        evolution_time = sequence_step * config.dataset.total_time / config.dataset.time_steps
        evolution_time = round(evolution_time, config.dataset.digits)
        unitary = iterates[sequence_step]
        unitary_m_t = tf.constant(unitary.unitary(), dtype=tf.complex128)
      if config.profile_dataset and not finished_profile:
        finished_profile = True
        with tf.profiler.experimental.Profile(results_dir):
          with tf.profiler.experimental.Trace(
              "compute_data_point_metrics", step_num=0, _r=1):
            if is_vqt:
              target_state, true_correlation, true_entropy, true_log_partition_function = compute_data_point_metrics(
                  average_correlation_op_m_t,
                  beta_t=beta_t,
                  target_h_m_t=target_h_m_t)
            else:
              target_state, true_correlation, true_entropy = compute_data_point_metrics(
                  average_correlation_op_m_t,
                  init_state=target_state,
                  channel_m_t=unitary_m_t)
      else:
        if is_vqt:
          target_state, true_correlation, true_entropy, true_log_partition_function = compute_data_point_metrics(
              average_correlation_op_m_t,
              beta_t=beta_t,
              target_h_m_t=target_h_m_t)
        else:
          target_state, true_correlation, true_entropy = compute_data_point_metrics(
              average_correlation_op_m_t,
              init_state=target_state,
              channel_m_t=unitary_m_t)
      if config.training.loss == "vqt":
        data_point_label = (
            f"bias_{str(bias).replace('.','p')}_beta_{str(beta).replace('.','p')}"
        )
        logging.info(f"Starting experiment: bias {bias}, beta {beta}")
      else:
        data_point_label = (
            f"bias_{str(bias).replace('.','p')}_time_{str(evolution_time).replace('.','p')}"
        )
        logging.info(f"Starting experiment: bias {bias}, time {evolution_time}")
      if iteration == 0:
        data_point_dir = os.path.join(results_dir, "metrics", data_point_label,
                                      "data_point")
        data_point_metrics_writer = tf.summary.create_file_writer(
            data_point_dir)
        with data_point_metrics_writer.as_default():
          step_zero = tf.constant(0, dtype=tf.int64)
          tf.summary.scalar(
              "true_correlation", true_correlation, step=step_zero)
          tf.summary.scalar("true_entropy", true_entropy, step=step_zero)
          if is_vqt:
            tf.summary.scalar(
                "true_log_partition_function",
                true_log_partition_function,
                step=step_zero)

      # Training loop
      if config.training.train:
        if not is_vqt:
          model_h_to_propagate, _ = get_initial_hamiltonian_and_inference(
              target_h_shards, config, "model_h_to_propagate")
          for c, m in zip(model_h_to_propagate.variables, model_h.variables):
            c.assign(m)
        if sequence_step > 0 and config.training.param_init == "random":
          model_h, model_inf = get_initial_hamiltonian_and_inference(
              target_h_shards, config, "model_h")
          optimizer = get_optimizer(config.training.optimizer,
                                    config.training.learning_rate)

        model_label = f"p_{config.hparams.p}_iteration_{iteration}"
        model_dir = os.path.join(results_dir, "metrics", data_point_label,
                                 model_label)
        model_metrics_writer = tf.summary.create_file_writer(model_dir)
        logging.info(f"Number of layers: {config.hparams.p}")
        initial_t = time.time()

        num_steps = config.training.head_of_snake_steps if sequence_step == 0 else config.training.max_steps
        method = config.training.method
        if is_vqt:
          final_step = train_model(
              model_inf,
              model_h,
              target_h_shards,
              target_state,
              model_dir,
              model_metrics_writer,
              config,
              optimizer,
              num_steps,
              method,
              target_h=target_h_t,
              beta=beta_t)
        else:
          final_step = train_model(
              model_inf,
              model_h,
              target_h_shards,
              target_state,
              model_dir,
              model_metrics_writer,
              config,
              optimizer,
              num_steps,
              method,
              model_h_to_propagate=model_h_to_propagate,
              channel=unitary)
        with model_metrics_writer.as_default():
          if config.logging.relative_entropy:
            # VQT direction of relative entropy
            relative_entropy = baselines_utils.relative_entropy(
                inference.density_matrix(model_h), target_state)
            tf.summary.scalar(
                "relative_entropy", relative_entropy, step=final_step)
          model_correlation = model_inf.expectation(average_correlation_op_t)[0]
          tf.summary.scalar(
              "model_correlation", model_correlation, step=final_step)
          if is_vqt:
            target_loss = -true_log_partition_function
          else:
            target_loss = true_entropy
          tf.summary.scalar("target_loss", target_loss, step=final_step)

        total_wall_time = time.time() - initial_t
        logging.info("Finished training. total %.2f min",
                     total_wall_time / 60.0)

      summary_writer = tf.summary.create_file_writer(
          os.path.join(results_dir, "hparams"))
      with summary_writer.as_default():
        hp.hparams({
            "qnn_layers":
                config.hparams.p,
            "training_method":
                config.training.method,
            "learning_rate":
                config.training.learning_rate
        })


if __name__ == "__main__":
  app.run(main)
