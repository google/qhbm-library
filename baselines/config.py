"""Configuration for geometric regularization experiments."""

import datetime
import getpass
import os.path

import itertools
import ml_collections


def get_config():
  """Returns the configuration for this experiment."""
  config = ml_collections.ConfigDict()
  config.experiment_name = ('qhbm_experiment' + '_' +
      datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))

  # dataset generation settings
  dataset = ml_collections.ConfigDict()
  dataset.num_rows = 2
  dataset.num_cols = 2
  dataset.lattice_dim = 1
  dataset.bias = 1.0
  dataset.beta = 1.0
  dataset.beta_min = 0.5
  dataset.beta_max = 2.25
  dataset.beta_steps = 4
  dataset.total_time = 3.0
  dataset.time_steps = 3
  dataset.trotter_steps = 1
  dataset.digits = 3
  config.dataset = dataset

  model = ml_collections.ConfigDict()
  model.energy = 'kobe'
  model.kobe_order = 2
  model.ebm = 'analytic'
  model.energy_init_mean = 0.0
  model.energy_init_stddev = 0.1
  model.circuit = 'qhea'
  model.circuit_layers = 7
  model.circuit_init_mean = 0.0
  model.circuit_init_stddev = 0.1
  model.qnn = 'analytic'
  config.model = model

  # training settings
  training = ml_collections.ConfigDict()
  # If False, only simulate the dataset (no model training)
  training.train = True
  training.num_trials = 1
  training.loss = 'qvartz'
  training.method = 'mirror'
  training.optimizer = 'Adam'
  training.learning_rate = 0.1
  training.inner_learning_rate = 2.5e-3
  training.init_steps = 1000
  training.num_steps = 100
  training.num_inner_steps = 100
  training.num_samples = 500
  training.seq_init = 'prev'
  training.info_matrix_reg = 1.0
  training.info_matrix_eigval_reg = True
  training.lstsq_fast = False
  training.lstsq_l2_regularizer = 1e-2
  training.euclidean_div_factor = 0.5
  config.training = training

  # logging settings
  logging = ml_collections.ConfigDict()
  logging.loss = True
  logging.variables = True
  logging.grads = True
  logging.norm_ord = 2
  logging.fidelity = True
  logging.relative_entropy = True
  logging.density_matrix = False
  logging.info_matrix = True
  logging.reg_info_matrix = True
  logging.natural_grads = True
  logging.inner_loss = True
  logging.inner_prod = True
  logging.div = True
  logging.inner_loss_grads = True
  logging.expensive_downsample = 1
  config.logging = logging

  config.args = {
      'experiment_name': config.experiment_name,
      'output_dir': '/tmp/qhbm_logs/{}'.format(config.experiment_name),
      'config': os.path.basename(__file__),
      'seed': 42,
  }
  return config


def get_sweep():
  loss = ['vqt', 'qvartz']
  method = ['vanilla', 'natural', 'mirror']
  optimizer = ['SGD', 'Adam']
  seq_init = ['random', 'optimal']
  return list(dict([('config.training.loss', l), ('config.training.method', m), ('config.training.optimizer', o), ('config.training.seq_init', i)]) for (l, m, o, i) in itertools.product(loss, method, optimizer, seq_init))
