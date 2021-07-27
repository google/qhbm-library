# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement (CLA). You (or your employer) retain the copyright to your
contribution; this simply gives us permission to use and redistribute your
contributions as part of the project. Head over to
<https://cla.developers.google.com/> to see your current agreements on file or
to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code Reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Developer Workflow

We follow a standard open source pull request workflow. Before starting, be sure to follow the instructions to [install from source](https://github.com/google/qhbm-library/blob/main/docs/install.md#install-from-source).

### 2. Set up development tools

We use a Python dependency manager called [poetry](https://python-poetry.org/). Install it from source:
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