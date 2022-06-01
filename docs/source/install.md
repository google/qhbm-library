# Install the QHBM Library

There are two ways to start developing with the QHBM Library:
* Install the package in a local environment.
* Work from the source code.


## Pip package

To install, simply run:
```
pip install qhbmlib
```

## Install from source

### 1. Fork and clone the repository

In the top right of the GitHub project, under your profile picture, there is a button labelled "Fork". Click this button. You now have a personal repository with a copy of the library code.

Open a terminal. From your working directory, clone your forked copy of the library:
```
git clone https://github.com/USERNAME/qhbm-library.git
cd qhbm-library
```

Now you need to tell your local git client about the parent repo of your fork:
```
git remote add upstream https://github.com/google/qhbm-library.git
```

### 2. Install dependency manager

We use a Python dependency manager called [poetry](https://python-poetry.org/). Install it from source:
```
curl -sSL https://install.python-poetry.org | python<X> -
```
where `<X>` is your desired version of Python; we currently support 3.7, 3.8, or 3.9.  You may be prompted to install additional development packages; you may also need to add poetry to your PATH variable, see the [poetry documentation](https://python-poetry.org/docs/master/#installing-with-the-official-installer) for details.  Restart the shell and confirm successful installation:
```
poetry --version
```

### 3. Install QHBM Library

Poetry automatically manages your environment using the specifications in the `pyproject.toml` file. To initiate your `poetry` managed virtual environment and install all dependencies, simply run:
```
poetry install
```
To confirm that the QHBM Library has been successfully installed from source, you can run the unit tests:
```
poetry run pytest
```
