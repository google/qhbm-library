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
  output_dir = 'qhbm/{}'.format(
      config.experiment_name)

  config.profile_dataset = False

  # dataset generation settings.
  dataset = ml_collections.ConfigDict()
  dataset.num_rows = 2
  dataset.num_cols = 2
  dataset.bias = 1.0
  dataset.beta_min = 0.5
  dataset.beta_max = 2.25
  dataset.beta_steps = 4
  dataset.digits = 3
  dataset.lattice_dimension = 1
  # QVARTZ
  dataset.total_time = 3.0
  dataset.time_steps = 3 #10
  dataset.trotter_steps = 1
  dataset.beta = 1.0
  dataset.coefficents = 'noiseless'
  dataset.x_amp = 1.0
  dataset.x_len = 1.0
  dataset.z_amp = 1.0
  dataset.z_len = 1.0
  config.dataset = dataset

  # training settings
  training = ml_collections.ConfigDict()
  # If False, only simulate the dataset (no model training)
  training.loss = 'qvartz'
  training.train = True
  training.samples = 500
  training.head_of_snake_steps = 10 + 1 #500
  training.num_steps = 10 + 1 #100
  training.method = 'vanilla'
  training.optimizer = 'Adam'
  training.learning_rate = 0.1
  training.param_init = 'optimal'
  training.quantum_inference = 'analytic'
  training.ansatz = 'qhea'
  config.training = training

  # logging settings
  logging = ml_collections.ConfigDict()
  logging.loss = True
  logging.fidelity = True
  logging.thetas = True
  logging.phis = True
  logging.thetas_grads = True
  logging.phis_grads = True
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
  # Fraction of loss within which to converge
  hparams.num_iterations = 1
  hparams.num_layers = 7  # default
  hparams.tied = False
  config.hparams = hparams

  config.args = {
      'experiment_name': config.experiment_name,
      'output_dir': output_dir,
      'config': os.path.basename(__file__),
      'debug': False,
  }
  return config


def get_sweep():
  optimizer = ['Adam']
  method = ['vanilla']
  param_init = ['random']
  p = [7]
  bias = [1.0]
  seed = [1]
  return list(dict([('config.training.optimizer', o), ('config.training.method', m), ('config.training.param_init', pi), ('config.hparams.p', pj), ('config.dataset.bias', b), ('seed', s)]) for (o, m, pi, pj, b, s) in itertools.product(optimizer, method, param_init, p, bias, seed))
