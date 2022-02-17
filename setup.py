# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['qhbmlib']

package_data = \
{'': ['*']}

install_requires = \
['tensorflow-probability==0.15.0',
 'tensorflow-quantum==0.6.1',
 'tensorflow==2.7.0']

setup_kwargs = {
    'name': 'qhbmlib',
    'version': '0.3.0',
    'description': 'Quantum Hamiltonian-Based Models built on TensorFlow Quantum',
    'long_description': '# QHBM Library\n\nThis repository is a collection of tools for building and training\nQuantum Hamiltonian-Based Models.  These tools depend on\n[TensorFlow Quantum](https://www.tensorflow.org/quantum),\nand are thus compatible with both real and simulated quantum computers.\n\nThis is not an officially supported Google product.\n',
    'author': 'The QHBM Library Authors',
    'author_email': 'no-reply@google.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/google/qhbm-library',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7,<3.10',
}


setup(**setup_kwargs)
