"""Configuration for geometric regularization experiments."""

import datetime
import getpass
import os.path

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
  config.cell = 'yo'
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
  dataset.num_sites = 4
  dataset.bias = 1.0
  dataset.beta_min = 1.0
  dataset.beta_max = 3.0
  dataset.beta_steps = 1
  dataset.digits = 3
  config.dataset = dataset

  # training settings
  training = ml_collections.ConfigDict()
  # If False, only simulate the dataset (no model training)
  training.train = True
  training.samples = 10000
  training.max_steps = 1000
  training.regularizer_order = 2
  training.regularizer_initial_strength = 0.0
  training.regularizer_decay_steps = 500
  # Fraction of loss within which to converge
  training.loss_stop_threshold = 0.04
  training.loss_stop_ignore_steps = 0
  training.method = 'natural'
  training.activation = 'sigmoid'
  training.lmbda = 100
  training.num_steps = 500
  training.num_start_steps = 3
  # start_opt = tf.keras.optimizers.SGD(learning_rate=0.001)
  # training.start_optimizer = tf.keras.optimizers.serialize(start_opt)
  training.num_start_inner_steps = 3000
  training.optimizer = 'ADAM'
  training.learning_rate = 0.01
  # training.learning_rate_end = 0.001
  # training.learning_decay_start = 500
  # training.learning_decay_end = 1000
  # training.learning_decay_step = 25
  training.num_inner_steps = 20
  training.num_samples = 1e5
  training.eps = 1e-2
  training.eigval_eps = False
  training.fast = False
  training.l2_regularizer = 0
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
  # logging.analytic = False
  # logging.image = False
  config.logging = logging

  # hyperparameters
  hparams = ml_collections.ConfigDict()
  hparams.ebm_param_lim = 0.25
  hparams.qnn_param_lim = 0.25
  # hparams.loss_p_threshold = 0.005  # Fraction of loss within which to converge
  hparams.max_iterations = 1
  # hparams.kobe_order = 2
  hparams.p = 5  # default
  config.hparams = hparams

  config.args = {
      'output_dir': output_dir,
      'debug': False,
  }
  return config


def get_sweep(hyper):
  num_trials = 1  # 2**10
  return hyper.product([
      hyper.sweep('seed', hyper.discrete(range(num_trials))),
      hyper.sweep('config.hparams.p', [2]),
      hyper.sweep('config.dataset.bias', [1.0]),
  ])
