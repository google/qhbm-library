"""Configuration for geometric regularization experiments."""

import datetime
import getpass
import os.path

import itertools
import ml_collections


def get_config():
  """Returns the configuration for this experiment."""
  config = ml_collections.ConfigDict()
  config.platform = 'cpu'
  # config.gpu_type = 'v100'
  config.num_gpus = 1
  config.experiment_name = ('qhbm_test' + '_' +
      datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
  output_dir = '/tmp/qhbm_logs/{}'.format(
      config.experiment_name)

  # dataset generation settings
  dataset = ml_collections.ConfigDict()
  dataset.num_rows = 2
  dataset.num_cols = 2
  dataset.lattice_dimension = 1
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

  # training settings
  training = ml_collections.ConfigDict()
  # If False, only simulate the dataset (no model training)
  training.train = True
  training.ansatz = 'qhea'
  training.qnn = 'analytic'
  training.loss = 'qvartz'
  training.optimizer = 'Adam'
  training.learning_rate = 0.1
  training.init_steps = 1000 + 1
  training.num_steps = 100 + 1
  training.samples = 500
  training.initialization = 'optimal'
  config.training = training

  # logging settings
  logging = ml_collections.ConfigDict()
  logging.loss = True
  logging.energy_variables = True
  logging.circuit_variables = True
  logging.energy_grads = True
  logging.circuit_grads = True
  logging.fidelity = True
  logging.relative_entropy = True
  logging.density_matrix = False
  logging.expensive_downsample = 1
  config.logging = logging

  # hyperparameters
  hparams = ml_collections.ConfigDict()
  hparams.energy_mean = 0.0
  hparams.energy_stddev = 0.1
  hparams.circuit_mean = 0.0
  hparams.circuit_stddev = 0.1
  hparams.num_iterations = 1
  hparams.num_layers = 7
  hparams.tied = False
  config.hparams = hparams

  config.args = {
      'experiment_name': config.experiment_name,
      'output_dir': output_dir,
      'config': os.path.basename(__file__),
  }
  return config


def get_sweep():
  optimizer = ['Adam']
  initialization = ['random']
  num_layers = [7]
  bias = [1.0]
  seed = [1]
  return list(dict([('config.training.optimizer', o), ('config.training.param_init', pi), ('config.hparams.p', pj), ('config.dataset.bias', b), ('seed', s)]) for (o, m, pi, pj, b, s) in itertools.product(optimizer, method, initialization, num_layers, bias, seed))
