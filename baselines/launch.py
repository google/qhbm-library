"""XManager/XCloud launcher for both GPU and TPU jobs.

The launcher works with any Python binary with the following flags:

* `output_dir` is the directory for saving summaries and logs;
* `use_gpu` determines whether to run on GPU or otherwise TPU;
* `num_cores` is the number of TPU cores or GPUs;
* `tpu` is the TPU main address (flag not required if launching with GPU);
* `seed` is the experiment's random seed.

For binaries that support only certain accelerator settings, we recommend still
using these flags. Raise errors outside its support or rely on runtime errors.
"""

import collections
import functools
import importlib.util
import inspect
import json
import operator
import os
import random
import time
import asyncio
from typing import Any, Dict, List, Optional, Text

from absl import app
from absl import flags
from absl import logging
from ml_collections.config_dict import config_dict
from xmanager import xm
from xmanager import xm_local
from xmanager.cloud import vertex

# Binary flags
flags.DEFINE_string(
    'binary', None,
    'Filepath to Python script to run. For external GCS experiments, it can be '
    'an absolute path to the binary, or a relative one with respect to the '
    'current folder.')
flags.mark_flag_as_required('binary')
flags.DEFINE_list(
    'args', [], 'Flag arguments to pass to binary. Follow the format '
    '--args=batch_size=64,train_epochs=300.')
flags.DEFINE_string(
    'config', None, 'Filepath to Python file with a function '
    'get_sweep(hyper) returning a hyperparameter sweep and/or '
    'a function get_config() returning a ConfigDict.')
flags.DEFINE_bool('launch_on_gcp', True, 'Whether or not to launch on GCS.')

# Accelerator flags
flags.DEFINE_string('platform', None, 'Platform (e.g., tpu-v2, tpu-v3, gpu).')
flags.DEFINE_string(
    'tpu_topology', '2x2',
    'TPU topology. Only used if platform is TPU. {x}x{y} means x*x **chips**, '
    'and because the number of devices is the number of cores, we further '
    'multiply by 2 because there are 2 cores per chip. For example, 2x2 is '
    'equivalent to an 8 core TPU slice, 8x8 = 128 cores, etc.')
flags.DEFINE_string('gpu_type', 'p100',
                    'GPU type. Only used if platform is GPU.')
flags.DEFINE_integer('num_gpus', None,
                     'Number of GPUs. Only used if platform is GPU.')
flags.DEFINE_integer('num_cpus', None, 'Number of CPUs.')
flags.DEFINE_integer('num_workers', 1, 'Number of workers (including chief)'
                     'in cluster.')
flags.DEFINE_integer(
    'memory', None, 'Amount of CPU memory in GB. Only used if launching on '
    'GCP.')
flags.DEFINE_string('experiment_name', None,
                    'Experiment name; defaults to timestamp.')
flags.DEFINE_integer('num_runs', 1,
                     'Number of runs each with a different seed.')

FLAGS = flags.FLAGS

_JobMetadata = collections.namedtuple('_JobMetadata', [
    'platform_str',
    'num_workers',
    'gpu_type',
    'num_gpus',
    'tpu_topology',
    'num_cpus',
    'experiment_name',
    'memory',
])


def _get_attr(config, name: str) -> Optional[Any]:
  """Get a given attribute from the passed FLAGS or ConfigDict."""
  # Note that if a flag is passed with its default value, this will not override
  # a conflicting config value.
  has_flag_value = name in FLAGS and FLAGS[name].value != FLAGS[name].default
  if has_flag_value:
    return FLAGS[name].value
  elif config and name in config:
    return config[name]
  elif name in FLAGS:
    return FLAGS[name].default
  return None


def _build_binary_metadata(config):
  """Extracts job metadata and args from the given ConfigDict and/or FLAGS."""

  if config:
    flag_args = config.args
    experiment_name = _get_attr(config, 'experiment_name')
  else:
    flag_args = dict(arg.split('=', 1) for arg in FLAGS.args)
    experiment_name = FLAGS.experiment_name

  if not experiment_name:  # default experiment name
    experiment_name = time.strftime('%m%d_%H%M%S')

  metadata = _JobMetadata(
      platform_str=_get_attr(config, 'platform'),
      num_workers=_get_attr(config, 'num_workers'),
      gpu_type=_get_attr(config, 'gpu_type'),
      num_gpus=_get_attr(config, 'num_gpus'),
      tpu_topology=_get_attr(config, 'tpu_topology'),
      num_cpus=_get_attr(config, 'num_cpus'),
      memory=_get_attr(config, 'memory'),
      experiment_name=experiment_name,
  )

  use_gpu = 'gpu' in metadata.platform_str or metadata.platform_str == 'cpu'

  if metadata.platform_str == 'cpu':
    num_cores = 1
  elif 'gpu' in metadata.platform_str:
    num_cores = metadata.num_gpus
  else:
    num_cores = 2 * functools.reduce(
        operator.mul, [int(i) for i in metadata.tpu_topology.split('x')])
  if 'num_cores' in flag_args and flag_args['num_cores'] != num_cores:
    raise ValueError(
        '"num_cores" requested in binary incompatible with inferred number of '
        'cores based on tpu_topology and platform_str ({}!={} respectively)'
        .format(flag_args['num_cores'], num_cores))
  args = dict(num_cores=num_cores, use_gpu=use_gpu)
  args.update(flag_args)
  return args, metadata


def _split_path_to_ub(filepath):
  """For a path '/a/b/c/baselines/...', return '/a/b/c', 'baselines/...'."""
  filepath = os.path.abspath(filepath)
  pieces = filepath.split('/')
  library_index = None
  for pi, piece in enumerate(pieces):
    if piece == 'qhbm-library':
      library_index = pi + 1
      break
  if library_index is None:
    raise ValueError(
        'Unable to parse FLAGS.binary ({}) to find the location of the qhbm-library.'.format(filepath))
  library_dir = '/'.join(pieces[:library_index])
  project_dir = '/'.join(pieces[library_index:-1])
  binary_path = '/'.join(pieces[-1:])
  return library_dir, project_dir, binary_path

def _launch_gcp_experiment(library_dir, project_dir, binary_path, sweep, args, metadata):
  """Launch a job on GCP using the Cloud AI Platform."""

  with xm_local.create_experiment(metadata.experiment_name) as experiment:
    # Note that we normally would need to append a "$@" in order to properly
    # forward the args passed to the job into the python command, but the XM
    # library already does this for us.
    run_cmd = f'python {binary_path} $@'
    # These images are necessary to get tf-nightly pre-installed.
    # Our lazy loading `__getattr__ = _lazy_import` in `__init__.py` requires
    # at least Python 3.7, so we use a base image that has Python 3.7.
    if metadata.platform_str == 'gpu':
      # base_image = 'tensorflow/tensorflow:nightly-gpu'
      base_image = 'gcr.io/deeplearning-platform-release/tf2-gpu.2-7'
    else:
      # base_image = 'tensorflow/tensorflow:nightly'
      base_image = 'gcr.io/deeplearning-platform-release/tf2-cpu.2-7'
    pip_cmd = 'pip --no-cache-dir install'
    # spec = xm.Dockerfile(path=library_dir)
    spec = xm.PythonContainer(
        path=library_dir,
        base_image=base_image,
        entrypoint=xm.CommandList([run_cmd]),
        docker_instructions=[
            f'COPY {os.path.basename(library_dir)} qhbm-library',
            f'WORKDIR qhbm-library',
            'RUN curl -sSL https://install.python-poetry.org | python - --preview',
            'RUN export PATH="/root/.local/bin:$PATH"',
            'RUN /root/.local/bin/poetry config virtualenvs.create false && /root/.local/bin/poetry install --no-interaction --no-ansi',
            f'RUN {pip_cmd} ml_collections',
            f'WORKDIR {project_dir}',
        ],
    )
    [executable] = experiment.package([
        xm.Packageable(
            executable_spec=spec,
            executor_spec=xm_local.Vertex.Spec(),
        ),
    ])

    platform = {}
    if 'tpu' in metadata.platform_str:
      # To run on a tpu-v2-8, tpu_topology should be 2x2.
      pieces = map(int, metadata.tpu_topology.split('x'))
      num_tpus = pieces[0] * pieces[1] * 2  # 2 cores per TPU chip.
      platform = {metadata.platform_str.split('-')[-1]: num_tpus}
    elif metadata.platform_str == 'gpu':
      platform = {metadata.gpu_type: metadata.num_gpus}

    if metadata.num_cpus is not None:
      platform['cpu'] = metadata.num_cpus * xm.vCPU
    if metadata.memory is not None:
      platform['memory'] = metadata.memory * xm.GiB

    bucket = os.environ.get('GOOGLE_CLOUD_BUCKET_NAME')

    tensorboard = vertex.get_default_client().get_or_create_tensorboard(metadata.experiment_name)
    tensorboard = asyncio.get_event_loop().run_until_complete(tensorboard)
    
    # Create one job per setting in the hyperparameter sweep. The default case
    # is a length 1 sweep with a single argument name "seed".
    for ji, sweep_args in enumerate(sweep):
      job_args = args.copy()
      if 'output_dir' in job_args:
        job_args['output_dir'] = os.path.join(bucket, job_args['output_dir'], str(experiment.experiment_id), str(ji))
      if 'data_dir' in job_args and job_args.get('download_data', False):
        job_args['data_dir'] = os.path.join(bucket, job_args['data_dir'], str(experiment.experiment_id), str(ji))
      # Overwrite any values in `args` with the `sweep_args`.
      job_args.update(sweep_args)
      tensorboard_capability = xm_local.TensorboardCapability(name=tensorboard, base_output_directory=job_args['output_dir'])
      job_requirements = xm.JobRequirements(**platform)
      executor = xm_local.Vertex(requirements=job_requirements, tensorboard=tensorboard_capability)
      logging.info('Launching job %d/%d with args %s.\n', ji + 1, len(sweep),
                  json.dumps(job_args, indent=4, sort_keys=True))
      job = xm.Job(
          executable=executable,
          executor=executor,
          args=job_args,
      )
      experiment.add(job)


def _generate_hyperparameter_sweep(
    config_module) -> List[Dict[Text, Any]]:
  """Generate the hyperparameter sweep."""
  if FLAGS.config and 'get_sweep' in dir(config_module):
    if FLAGS.num_runs != 1:
      raise ValueError('FLAGS.num_runs not supported with config.get_sweep().')
    sweep = config_module.get_sweep()
  else:
    sweep = [{
        'seed': seed + random.randint(0, 1e10)
    } for seed in range(FLAGS.num_runs)]
  return sweep


def _load_config_helper(config_path, launch_on_gcp):
  """Get the ConfigDict from config_path:get_config()."""
  config_module_spec = importlib.util.spec_from_file_location(
      '', os.path.abspath(config_path))
  config_module = importlib.util.module_from_spec(config_module_spec)
  config_module_spec.loader.exec_module(config_module)
  config = None
  if 'get_config' in dir(config_module):
    # Check if get_config takes a parameter called launch_on_gcp, and if so then
    # pass in FLAGS.launch_on_gcp.
    get_config_inspect = inspect.getfullargspec(config_module.get_config)
    get_config_params = get_config_inspect.args
    if 'launch_on_gcp' in get_config_params:
      config = config_module.get_config(launch_on_gcp=launch_on_gcp)
    else:
      config = config_module.get_config()
  return config_module, config


def _load_config(config_path, launch_on_gcp):
  """Load the ConfigDict if one was passed in as FLAGS.config."""
  if config_path:
    config_module = None
    if not config_module:
      config_module, config = _load_config_helper(config_path, launch_on_gcp)
  else:
    config_module = None
    config = None
  return config_module, config

def main(argv):
  del argv  # unused arg
  config_module, config = _load_config(FLAGS.config, FLAGS.launch_on_gcp)
  args, metadata = _build_binary_metadata(config)
  print(args, metadata)
  sweep = _generate_hyperparameter_sweep(config_module)
  if FLAGS.launch_on_gcp:
    library_dir, project_dir, binary_path = _split_path_to_ub(FLAGS.binary)
    print(library_dir, project_dir, binary_path)
    _launch_gcp_experiment(library_dir, project_dir, binary_path, sweep, args,
                                  metadata)


if __name__ == '__main__':
  app.run(main)
