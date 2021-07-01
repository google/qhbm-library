# Install the QHBM Library

There are three ways to start developing with the QHBM Library:
* Choose any example notebook at TODO, make a copy, and modify to suit your needs.
* Install the package in a local environment.
* Work from the source code.

## Pip package

### Requirements

## Build from source

The following steps were tested on Ubuntu.

### 1. Set up a Python 3 development environment

To manage python versions, you will use [pyenv](https://realpython.com/intro-to-pyenv/#installing-pyenv).  To start, install the [recommended dependencies](https://github.com/pyenv/pyenv/wiki#suggested-build-environment):
```
sudo apt-get update; sudo apt-get install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```
Install `pyenv` with the official installer:
```
curl https://pyenv.run | bash
```
Follow the directions in the output to finish adding `pyenv` to your system.  Once you finish appropriately updating your `~/.profile` file, `~/.bashrc` file, etc., restart your terminal:
```
exec "$SHELL"
```




### 2. Create a virtual environment
