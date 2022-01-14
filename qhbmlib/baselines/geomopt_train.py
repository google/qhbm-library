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
from qhbmlib import architectures
from qhbmlib import circuit_infer
from qhbmlib import circuit_model
from qhbmlib import energy_infer
from qhbmlib import energy_model
from qhbmlib import hamiltonian_infer
from qhbmlib import hamiltonian_model
from qhbmlib import util
from qhbmlib import vqt_new
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_quantum as tfq

google_internal = True
if google_internal:
  from google3.learning.deepmind.xmanager2.client import xmanager_api
  import google3.learning.deepmind.xmanager2.client.google as xm
  from google3.third_party.tensorboard.plugins.hparams import api as hp
  from google3.pyglib import gfile

tfb = tfp.bijectors
tfd = tfp.distributions

# Output logging
flags.DEFINE_string("experiment_label", "geomopt_ngd_ising_ring", "")
flags.DEFINE_string("output_dir", "/tmp/geomopt_logs/", "")
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
  if config.dataset.lattice_dimension == 1:
    num_sites = config.dataset.num_rows * config.dataset.num_cols
    qubits = cirq.GridQubit.rect(1, num_sites)
    hamiltonian = cirq.PauliSum()
    for i in range(num_sites):
      hamiltonian -= bias * cirq.X(qubits[i])
      hamiltonian -= cirq.Z(qubits[i]) * cirq.Z(qubits[(i + 1) % num_sites])
    return hamiltonian
  elif config.dataset.lattice_dimension == 2:
    num_rows = config.dataset.num_rows
    num_cols = config.dataset.num_cols
    z_hamiltonian = cirq.PauliSum()
    x_hamiltonian = cirq.PauliSum()
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


# def get_long_correlation_op(num_qubits):
#   qubits = cirq.GridQubit.rect(1, num_qubits)
#   op = cirq.PauliSum.from_pauli_strings(cirq.PauliString({qubits[0]: cirq.I}))
#   for q in qubits:
#     op *= cirq.Z(q)
#   return op


def compute_data_point_metrics(beta_t, target_h_m_t,
                               average_correlation_op_m_t):
  target_thermal_state = util.np_get_thermal_state(beta_t, target_h_m_t)
  true_correlation = tf.linalg.trace(
      tf.matmul(average_correlation_op_m_t, target_thermal_state))
  rho_eigs = tf.linalg.eigvalsh(target_thermal_state)
  rho_prod = tf.math.multiply_no_nan(tf.math.log(rho_eigs), rho_eigs)
  true_entropy = -tf.math.reduce_sum(rho_prod)
  return target_thermal_state, true_correlation, true_entropy


def get_initial_qhbm(config, name):
  """Gets initial untrained QHBM."""
  num_sites = config.dataset.num_rows * config.dataset.num_cols
  num_model_layers = config.hparams.p

  # energy
  ebm_initializer = tf.keras.initializers.RandomUniform(
      minval=-config.hparams.ebm_param_lim, maxval=config.hparams.ebm_param_lim)
  energy = energy_model.KOBE(list(range(num_sites)), 2, ebm_initializer)
  energy.build([None, num_sites])

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
    u, _ = architectures.get_hardware_efficient_model_unitary(
        raw_qubits, num_model_layers, name)
  qnn_initializer = tf.keras.initializers.RandomUniform(
      minval=-config.hparams.qnn_param_lim, maxval=config.hparams.qnn_param_lim)
  if config.training.method == 'mirror':
    symbol_names = tf.constant([str(x) for x in list(sorted(tfq.util.get_circuit_symbols(u)))], dtype=tf.string)
    values = [tf.Variable(qnn_initializer(shape=[len(symbol_names)]))]
    if config.geometry.activation == 'tanh':
      activation = tf.keras.layers.Activation(lambda x: np.pi * tf.tanh(config.geometry.activation_scale * x))
    elif config.geometry.activation == 'sigmoid':
      activation = tf.keras.layers.Activation(lambda x: 2 * np.pi * (tf.sigmoid(config.geometry.activation_scale * x) - 0.5))
    elif config.geometry.activation == 'clip':
      activation = tf.keras.layers.Activation(lambda x: tf.clip_by_value(x, -np.pi, np.pi))
    else:
      raise ValueError()
    value_layers = [[activation]]
    test_qnn = circuit_model.QuantumCircuit(u, u.all_qubits(), symbol_names, values, value_layers)
  else:
    test_qnn = circuit_model.DirectQuantumCircuit(u, qnn_initializer)
  test_qnn.build([])
  return hamiltonian_model.Hamiltonian(energy, test_qnn, name)


def information_matrix(h_inf, model, model_copy, config):
  """Estimates the Bogoliubov-Kubo-Mori information matrix.

  Args:
    h_inf: Hamiltonian inference.
    model: qhbm model. exp(-model)/Z(model) = rho.
    model_copy: copy of model.
    config: config dict.

  Returns:
    The BKM information matrix. This is tr[d_j rho d_k model] element-wise i.e.
    the Hilbert-Schmidt inner product of a mixture coords tangent vector and
    an exponential coords tangent vector.
  """

  def ebm_block():
    h_inf.e_inference.infer(model.energy)
    samples = h_inf.e_inference.sample(config.geometry.num_samples)

    with tf.GradientTape() as tape:
      tape.watch(model.energy.trainable_variables[0])
      energies = model.energy(samples)
    energy_jac = tape.jacobian(energies, model.energy.trainable_variables[0])
    avg_energy_grad = tf.reduce_mean(energy_jac, axis=0)
    centered_energy_jac = energy_jac - avg_energy_grad
    # sample-based approximation of covariance of energy grads.
    return tf.matmul(
        centered_energy_jac, centered_energy_jac,
        transpose_a=True) / config.geometry.num_samples

  def cross_block():
    with tf.GradientTape() as t2:
      t2.watch(model.circuit.trainable_variables[0])
      with tf.GradientTape() as t1:
        t1.watch(model_copy.energy.trainable_variables[0])
        expectation = h_inf.expectation(model, model_copy,
                                        config.geometry.num_samples)

      g = t1.gradient(expectation, model_copy.energy.trainable_variables[0])

    return t2.jacobian(g, model.circuit.trainable_variables[0])

  def qnn_block():
    shift = tf.constant(0.5)
    scale = tf.constant(np.pi / 2)

    qnn_values = tf.identity(model.circuit.value_layers_inputs[0])

    def adj_diff_grad(indices, updates):
      model.circuit.value_layers_inputs[0].assign(
          tf.tensor_scatter_nd_add(
              qnn_values, indices=indices, updates=updates))

      with tf.GradientTape() as tape:
        tape.watch(model_copy.circuit.trainable_variables[0])
        expectation = h_inf.expectation(model, model_copy,
                                        config.geometry.num_samples)

      return tape.jacobian(expectation,
                           model_copy.circuit.trainable_variables[0])

    def row(i):
      return scale * (
          adj_diff_grad([[i]], [-shift]) - adj_diff_grad([[i]], [shift]))

    indices = tf.range(tf.shape(model.circuit.trainable_variables[0])[0])
    model.circuit.value_layers_inputs[0].assign(qnn_values)
    return tf.map_fn(fn=row, elems=indices, fn_output_signature=tf.float32)

  block_ebm = ebm_block()
  block_cross = cross_block()
  block_qnn = qnn_block()

  block_upper = tf.concat([block_ebm, block_cross], 1)
  block_lower = tf.concat([tf.transpose(block_cross), block_qnn], 1)
  im = tf.concat([block_upper, block_lower], 0)
  return (im + tf.transpose(im)) / 2.

def conditional_decorator(dec, condition):
    def decorator(func):
        if condition:
            return dec(func)
        return func
    return decorator

def train_model(h_inf, model, model_copy, target_h, beta, target_loss, target_dm, avg_corr_op,
                metrics_dir, config, work_unit):
  """Train given model and write metrics on progress."""
  metrics_writer = tf.summary.create_file_writer(metrics_dir)
  optimizer = tf.keras.optimizers.get(dict(config.training.optimizer))
  
  @conditional_decorator(tf.function, config.training.method == 'vanilla' or config.training.method == 'natural')
  def train_step(step):
    with tf.GradientTape() as tape:
      loss = vqt_new.vqt(h_inf, model, config.training.num_samples, target_h, beta)
    loss_grads = tape.gradient(loss, model.trainable_variables)

    if config.training.method == 'vanilla':
      optimizer.apply_gradients(zip(loss_grads, model.trainable_variables))

    if config.training.method == "natural":
      for c, m in zip(model_copy.trainable_variables, model.trainable_variables):
        c.assign(m)
      im = information_matrix(h_inf, model, model_copy, config)
      if config.geometry.eigval_eps:
        e = tf.linalg.eigvalsh(im)
        e = tf.sort(tf.cast(e, tf.float32))
        if e[0] <= config.geometry.eps:
          reg = config.geometry.eps + tf.math.abs(tf.math.minimum(e[0], 0.0))
        else:
          reg = 0.0
      else:
        reg = config.geometry.eps
      reg_im = im + reg * tf.eye(tf.shape(im)[0])

      flat_grads = tf.concat([tf.reshape(g, [-1]) for g in loss_grads], 0)
      flat_natural_grads = tf.squeeze(
          tf.linalg.lstsq(
              reg_im,
              tf.expand_dims(flat_grads, 1),
              l2_regularizer=config.geometry.l2_regularizer,
              fast=config.geometry.fast))
      vars_shapes = [tf.shape(var) for var in model.trainable_variables]
      vars_sizes = [tf.size(var) for var in model.trainable_variables]
      natural_grads = []
      i = 0
      for size, shape in zip(vars_sizes, vars_shapes):
        natural_grads.append(tf.reshape(flat_natural_grads[i:i + size], shape))
        i += size
      optimizer.apply_gradients(zip(natural_grads, model.trainable_variables))

      with metrics_writer.as_default():
        e = tf.linalg.eigvalsh(im)
        e = tf.sort(tf.cast(e, tf.float32))
        tf.summary.histogram("im_eigvals", e, step=step)
        tf.summary.scalar("min_im_eigval", e[0], step=step)
        tf.summary.scalar("max_im_eigval", e[-1], step=step)
        tf.summary.scalar("avg_im_eigval", tf.reduce_mean(e), step=step)
        tf.summary.scalar("im_norm", tf.linalg.norm(im), step=step)

        tf.summary.scalar("reg", reg, step=step)
        e = tf.linalg.eigvalsh(reg_im)
        e = tf.sort(tf.cast(e, tf.float32))
        tf.summary.histogram("reg_im_eigvals", e, step=step)
        tf.summary.scalar("min_reg_im_eigval", e[0], step=step)
        tf.summary.scalar("max_reg_im_eigval", e[-1], step=step)
        tf.summary.scalar("avg_reg_im_eigval", tf.reduce_mean(e), step=step)
        tf.summary.scalar("reg_im_norm", tf.linalg.norm(reg_im), step=step)

        tf.summary.histogram(f'energy_natural_grads', natural_grads[:-1], step=step)
        tf.summary.histogram(f'circuit_natural_grads', natural_grads[-1:], step=step)
        tf.summary.scalar(f'energy_natural_grad_size', tf.reduce_max(tf.abs(natural_grads[:-1])), step=step)
        tf.summary.scalar(f'circuit_natural_grad_size', tf.reduce_max(tf.abs(natural_grads[-1:])), step=step)

    if config.training.method == 'mirror':
      step_metrics_dir = os.path.join(metrics_dir, f'train_step_{step}')
      step_metrics_writer = tf.summary.create_file_writer(step_metrics_dir)

      learning_rate = optimizer.learning_rate
      if isinstance(learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
        learning_rate = learning_rate(step)
      inner_optimizer = tf.keras.optimizers.get(dict(config.training.inner_optimizers[step]))

      for c, m in zip(model_copy.trainable_variables, model.trainable_variables):
        c.assign(m)

      @tf.function
      def train_inner_step(inner_step):
        with tf.GradientTape(persistent=True) as tape:
          energy_inner_prod = tf.reduce_sum([tf.reduce_sum(v * g) for v, g in zip(model.energy.trainable_variables, loss_grads[:-1])])
          circuit_inner_prod = tf.reduce_sum([tf.reduce_sum(v * g) for v, g in zip(model.circuit.symbol_values, loss_grads[-1:])])
          inner_prod = energy_inner_prod + circuit_inner_prod
          reg = vqt_new.vqt(model, config.training.num_samples, model_copy, 1.0)
          inner_loss = inner_prod + 1.0 / learning_rate * reg
        inner_loss_grads = tape.gradient(inner_loss, model.trainable_variables)
        inner_prod_grads = tape.gradient(inner_prod, model.trainable_variables)
        reg_grads = tape.gradient(reg, model.trainable_variables)
        inner_optimizer.apply_gradients(zip(inner_loss_grads, model.trainable_variables))

        with step_metrics_writer.as_default():
          tf.summary.scalar(f'energy_inner_prod', energy_inner_prod, step=inner_step)
          tf.summary.scalar(f'circuit_inner_prod', circuit_inner_prod, step=inner_step)
          tf.summary.scalar(f'inner_prod', inner_prod, step=inner_step)
          tf.summary.histogram(f'energy_inner_prod_grads', inner_prod_grads[:-1], step=inner_step)
          tf.summary.histogram(f'circuit_inner_prod_grads', inner_prod_grads[-1:], step=inner_step)
          tf.summary.scalar(f'energy_inner_prod_grad_size', tf.reduce_max(tf.abs(inner_prod_grads[:-1])), step=inner_step)
          tf.summary.scalar(f'circuit_inner_prod_grad_size', tf.reduce_max(tf.abs(inner_prod_grads[-1:])), step=inner_step)

          tf.summary.scalar(f'reg', reg, step=inner_step)
          tf.summary.histogram(f'energy_reg_grads', reg_grads[:-1], step=inner_step)
          tf.summary.histogram(f'circuit_reg_grads', reg_grads[-1:], step=inner_step)
          tf.summary.scalar(f'energy_reg_grad_size', tf.reduce_max(tf.abs(reg_grads[:-1])), step=inner_step)
          tf.summary.scalar(f'circuit_reg_grad_size', tf.reduce_max(tf.abs(reg_grads[-1:])), step=inner_step)

          tf.summary.scalar(f'inner_loss', inner_loss, step=inner_step)
          tf.summary.histogram(f'energy_inner_loss_grads', inner_loss_grads[:-1], step=inner_step)
          tf.summary.histogram(f'circuit_inner_loss_grads', inner_loss_grads[-1:], step=inner_step)
          tf.summary.scalar(f'energy_inner_loss_grad_size', tf.reduce_max(tf.abs(inner_loss_grads[:-1])), step=inner_step)
          tf.summary.scalar(f'circuit_inner_loss_grad_size', tf.reduce_max(tf.abs(inner_loss_grads[-1:])), step=inner_step)
        
          running_loss = vqt_new.vqt(model, config.training.num_samples, target_h, beta)
          tf.summary.scalar(f'running_loss', running_loss, step=inner_step)

          tf.summary.histogram(f'energy_variables', model.energy.trainable_variables, step=inner_step)
          tf.summary.histogram(f'circuit_variables', model.circuit.trainable_variables, step=inner_step)
          tf.summary.histogram(f'activation_circuit_variables', model.circuit.symbol_values, step=inner_step)

      for inner_step in tf.range(config.geometry.num_inner_steps[step], dtype=tf.int64):
        train_inner_step(inner_step)
    
    with metrics_writer.as_default():
      tf.summary.scalar("loss", loss, step=step)
      tf.summary.histogram("energy_loss_grads", loss_grads[:-1], step=step)
      tf.summary.histogram("circuit_loss_grads", loss_grads[-1:], step=step)
      tf.summary.histogram("energy_loss_grad_size", tf.reduce_max(tf.abs(loss_grads[:-1])), step=step)
      tf.summary.histogram("circuit_loss_grad_size", tf.reduce_max(tf.abs(loss_grads[-1:])), step=step)
      tf.summary.histogram("energy_variables", model.energy.trainable_variables, step=step)
      tf.summary.histogram("circuit_variables", model.circuit.trainable_variables, step=step)
      if step == 0:
        tf.summary.scalar("target_loss", target_loss, step=step)
      if config.logging.fidelity:
        if step % config.logging.expensive_downsample == 0 or step == config.training.num_steps - 1:
          fidelity = hamiltonian_infer.fidelity(model, target_dm)
          tf.summary.scalar("fidelity", fidelity, step=step)
      if config.logging.relative_entropy:
        if step % config.logging.expensive_downsample == 0 or step == config.training.num_steps - 1:
          relative_entropy = util.relative_entropy(hamiltonian_infer.density_matrix(model), target_dm)
          tf.summary.scalar("relative_entropy", relative_entropy, step=step)
      if config.logging.model_correlation:
        if step % config.logging.expensive_downsample == 0 or step == config.training.num_steps - 1:
          model_correlation = h_inf.expectation(model, avg_corr_op, config.training.num_samples)[0]
          tf.summary.scalar("model_correlation", model_correlation, step=step)
      if config.logging.density_matrix:
        if step % config.logging.expensive_downsample == 0 or step == config.training.num_steps - 1:
          density_matrix = util.density_matrix_to_image(hamiltonian_infer.density_matrix(model))
          tf.summary.image("density_matrix", density_matrix, step=step)

    return loss

  for step in tf.range(config.training.num_steps, dtype=tf.int64):
    loss = train_step(step)

    objective = work_unit.get_measurement_series("train/loss")
    objective.create_measurement(loss, step + 1)


def main(argv):
  del argv  # unused arg
  if google_internal:
    xm.setup_work_unit()
    xm_client = xmanager_api.XManagerApi()
    work_unit = xm_client.get_current_work_unit()
    work_unit_str = "run_{}_{}".format(work_unit.experiment_id, work_unit.id)

  tf.random.set_seed(FLAGS.seed)

  # ========================================================================== #
  # Get config.
  # ========================================================================== #

  results_dir = os.path.join(FLAGS.output_dir, work_unit_str, "results",
                             FLAGS.experiment_label)
  n_cores = len(tf.config.list_logical_devices("GPU"))
  logging.info(f"num GPU cores: {n_cores}")

  logging.info("\n\n\n\n\n\n\n\nBEGIN MAIN")
  config = FLAGS.config
  logging.info(f"\n\nSaving results to: {results_dir}\n\n")
  tf.io.gfile.makedirs(results_dir)
  with gfile.Open(os.path.join(results_dir, "config.json"), "w") as outfile:
    json.dump(config.to_dict(), outfile)

  average_correlation_op = get_average_correlation_op(config)
  average_correlation_op_m = average_correlation_op.matrix()
  average_correlation_op_m_t = tf.constant(
      average_correlation_op_m, dtype=tf.complex128)
  average_correlation_op_t = tfq.convert_to_tensor([average_correlation_op])

  # long_correlation_op = get_long_correlation_op(num_sites)
  # long_correlation_op_m = long_correlation_op.matrix()
  # long_correlation_op_m_t = tf.constant(
  #     long_correlation_op_m, dtype=tf.complex128)
  # long_correlation_op_t = tfq.convert_to_tensor([long_correlation_op])

  # if config.training.optimizer == "ADAM":
  #   optimizer_initializer = tf.keras.optimizers.Adam
  # elif config.training.optimizer == "SGD":
  #   optimizer_initializer = tf.keras.optimizers.SGD
  # else:
  #   raise ValueError(
  #       f"config.training.optimizer {config.training.optimizer} is not supported"
  #   )

  finished_profile = False
  float_bias = config.dataset.bias
  bias = round(float_bias, config.dataset.digits)
  if config.dataset.lattice_dimension == 1:
    target_h = get_tfim_hamiltonian(bias, config)
  elif config.dataset.lattice_dimension == 2:
    x_ham, z_ham = get_tfim_hamiltonian(bias, config)
    target_h = x_ham + z_ham
  target_h_m = target_h.matrix()
  target_h_m_t = tf.constant(target_h_m, dtype=tf.complex128)
  target_h_t = tfq.convert_to_tensor([target_h])
  for float_beta in tf.linspace(config.dataset.beta_min,
                                config.dataset.beta_max,
                                config.dataset.beta_steps).numpy().tolist():
    beta = round(float_beta, config.dataset.digits)
    beta_t = tf.constant(beta, dtype=tf.float64)
    if config.profile_dataset and not finished_profile:
      finished_profile = True
      with tf.profiler.experimental.Profile(results_dir):
        with tf.profiler.experimental.Trace(
            "compute_data_point_metrics", step_num=0, _r=1):
          target_thermal_state, true_correlation, true_entropy = compute_data_point_metrics(
              beta_t, target_h_m_t, average_correlation_op_m_t)
    else:
      target_thermal_state, true_correlation, true_entropy = compute_data_point_metrics(
          beta_t, target_h_m_t, average_correlation_op_m_t)
    data_point_label = (
        f"bias_{str(bias).replace('.','p')}_beta_{str(beta).replace('.','p')}")
    data_point_dir = os.path.join(results_dir, "metrics", data_point_label,
                                  "data_point")
    data_point_metrics_writer = tf.summary.create_file_writer(data_point_dir)
    logging.info(f"Starting experiment: bias {bias}, beta {beta}")
    with data_point_metrics_writer.as_default():
      step_zero = tf.constant(0, dtype=tf.int64)
      tf.summary.scalar("true_correlation", true_correlation, step=step_zero)
      # tf.summary.scalar(
      #     "true_long_correlation", true_long_correlation, step=step_zero)
      tf.summary.scalar("true_entropy", true_entropy, step=step_zero)

    # Training loop
    if config.training.train:
      for iteration in range(config.hparams.num_iterations):
        model = get_initial_qhbm(config, "vqt_model")
        e_inf = energy_infer.AnalyticEnergyInference(model.energy.num_bits)
        q_inf = circuit_infer.QuantumInference()
        h_inf = hamiltonian_infer.QHBM(e_inf, q_inf)

        model_label = f"p_{config.hparams.p}_iteration_{iteration}"
        model_dir = os.path.join(results_dir, "metrics", data_point_label,
                                 model_label)
        logging.info(f"Number of layers: {config.hparams.p}")
        initial_t = time.time()

        model_copy = get_initial_qhbm(config, "vqt_model_copy")
        train_model(h_inf, model, model_copy, target_h_t, beta, true_entropy, target_thermal_state, average_correlation_op_t, model_dir, config, work_unit)
        # with model_metrics_writer.as_default():
        #   if config.logging.relative_entropy:
        #     # VQT direction of relative entropy
        #     relative_entropy = util.relative_entropy(
        #         hamiltonian_infer.density_matrix(model), target_thermal_state)
        #     tf.summary.scalar(
        #         "relative_entropy", relative_entropy, step=final_step)
        #   model_correlation = h_inf.expectation(model, average_correlation_op_t,
        #                                         config.training.samples)[0]
        #   tf.summary.scalar(
        #       "model_correlation", model_correlation, step=final_step)
          # model_long_correlation = h_inf.expectation(model,
          #                                            long_correlation_op_t,
          #                                            config.training.samples)[0]
          # tf.summary.scalar(
          #     "model_long_correlation", model_long_correlation,
          #     step=final_step)
        total_wall_time = time.time() - initial_t
        logging.info("Finished training. total %.2f min",
                     total_wall_time / 60.0)

  if google_internal:
    summary_writer = tf.summary.create_file_writer(
        os.path.join(results_dir, "hparams"))
    with summary_writer.as_default():
      hp.hparams({
          "qnn_layers": config.hparams.p,
          "training_method": config.training.method,
          "learning_rate": config.training.learning_rate,
      })


if __name__ == "__main__":
  app.run(main)
