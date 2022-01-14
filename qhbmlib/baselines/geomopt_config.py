"""Configuration for geometric regularization experiments."""

import datetime
import getpass
import os.path

import tensorflow as tf
import numpy as np
import ml_collections


def get_config():
  """Returns the configuration for this experiment."""
  config = ml_collections.ConfigDict()
  config.user = getpass.getuser()
  config.priority = 'prod'
  config.platform = 'cpu'
  # config.gpu_type = 'v100'
  config.num_gpus = 1
  config.experiment_name = (
      os.path.splitext(os.path.basename(__file__))[0] + '_' +
      datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
  output_dir = 'gs://launcher-beta-test-bucket/geomopt/{}'.format(
      config.experiment_name)
  # BEGIN GOOGLE-INTERNAL
  config.gfs_user = 'x-quantum'
  config.cell = 'lu'
  output_dir = ('/cns/{cell}-d/home/{group}/{user}/rs=6.3/geomopt/'
                '{experiment}/'.format(
                    cell=config.cell,
                    group='x-quantum',
                    user=config.user,
                    experiment=config.experiment_name))
  # END GOOGLE-INTERNAL

  config.profile_dataset = False

  # dataset generation settings.
  dataset = ml_collections.ConfigDict()
  dataset.num_rows = 3
  dataset.num_cols = 3
  dataset.bias = 1.0
  dataset.beta_min = 1.0
  dataset.beta_max = 3.0
  dataset.beta_steps = 1
  dataset.digits = 3
  dataset.lattice_dimension = 2
  config.dataset = dataset

  # training settings
  training = ml_collections.ConfigDict()
  # If False, only simulate the dataset (no model training)
  training.train = True
  training.method = 'natural'
  training.optimizer = tf.keras.optimizers.serialize(tf.keras.optimizers.SGD(learning_rate=0.01))
  training.num_steps = 1000
  training.num_samples = 500
  # training.lmbda (use 1/lr instead)
  config.training = training

  geometry = ml_collections.ConfigDict()
  # NGD
  geometry.eps = 1e-2
  geometry.eigval_eps = True
  geometry.fast = False
  geometry.l2_regularizer = 0
  # MD
  geometry.activation = 'sigmoid'
  geometry.activation_scale = 6.0 / np.pi
  geometry.inner_optimizers = [tf.keras.optimizers.serialize(tf.keras.optimizers.SGD(learning_rate=0.001))] * 3 + [tf.keras.optimizers.serialize(tf.keras.optimizers.Adam(learning_rate=0.01))] * (training.num_steps - 3)
  geometry.num_inner_steps = [20] * training.num_steps
  # geometry.num_samples = 1e3
  config.geometry = geometry

  # logging settings
  logging = ml_collections.ConfigDict()
  logging.fidelity = True
  logging.relative_entropy = True
  logging.model_correlation = True
  logging.density_matrix = False
  logging.expensive_downsample = 10
  # logging.loss = True
  # logging.thetas = True
  # logging.phis = True
  # logging.thetas_grads = True
  # logging.phis_grads = True
  config.logging = logging

  # hyperparameters
  hparams = ml_collections.ConfigDict()
  hparams.ebm_param_lim = 0.25
  hparams.qnn_param_lim = 0.5
  # Fraction of loss within which to converge
  hparams.num_iterations = 1
  hparams.p = 6  # default
  config.hparams = hparams

  config.args = {
      'output_dir': output_dir,
      'debug': False,
  }
  return config


def get_sweep(hyper):
  num_random_trials = 2**4
  return hyper.product([
      hyper.sweep('config.training.method', ['natural', 'vanilla']),
      hyper.sweep('config.hparams.p', [6]),
      hyper.sweep(
          'config.dataset.bias',
          [
              2.5  # ferromagnetic regime (vanilla QHEA struggles)
          ]),
      hyper.zipit([
          hyper.loguniform('config.training.learning_rate',
                           hyper.interval(1e-3, 1.0)),
          hyper.uniform('seed', hyper.discrete(range(0, int(1e6))))
      ],
                  length=num_random_trials),
  ])
