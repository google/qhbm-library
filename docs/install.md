# Install the QHBM Library

There are three ways to start developing with the QHBM Library:
* Choose any example notebook at TODO, make a copy, and modify to suit your needs.
* Install the package in a local environment.
* Work from the source code.

## Pip package

### Requirements

## Install from source

### 1. Fork and clone the repository

In the top right of the GitHub project, under your profile picture, there is a button labelled "Fork". Click this button. You now have a personal repository with a copy of the library code.

Open a terminal. From your working directory, clone your forked copy of the library:
```
git clone https://github.com/USERNAME/qhbm-library.git
cd qhbm-library
```

### 2. Install Python development tools

To manage python versions, you will use [pyenv](https://realpython.com/intro-to-pyenv/#installing-pyenv). To start, install the [recommended dependencies](https://github.com/pyenv/pyenv/wiki#suggested-build-environment):
```
sudo apt update
sudo apt install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
sudo apt upgrade
```
Install `pyenv` with the official installer:
```
curl https://pyenv.run | bash
```
Follow the directions in the output to finish adding `pyenv` to your system. Once you finish appropriately updating your `~/.profile` file, `~/.bashrc` file, etc., restart your terminal:
```
exec $SHELL
```
Now you can install a specific version of python and make it the default for your current working directory:
```
pyenv install 3.8.11
pyenv local 3.8.11
```
You should verify that the python version is the one you just localized:
```
python --version
```

### 3. Set up a virtual environment

To manage virtual environments, you will use [poetry](https://python-poetry.org/). Install from source:
```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
```
After installation, follow the on screen instructions to ensure the `poetry` command can be found by your shell. Restart the shell and confirm successful installation:
```
poetry --version
```
Poetry automatically manages your environment using the specifications in the `pyproject.toml` file. To initiate your virtual environment and install all dependencies, run:
```
poetry install
```
Note that `poetry` uses the Python version you specified in the previous section. To confirm that the QHBM Library has been successfully installed from source, you can run the unit tests:
```
poetry run pytest
```
