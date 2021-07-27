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

### 1. Update your fork

Inside your local copy of your fork, checkout the mainline development branch:
```
git checkout main
```
Update with any remote changes:
```
git fetch upstream
git merge upstream/main
```

### 2. Create a development branch

Choose an [open issue](https://github.com/google/qhbm-library/issues) or [create one](https://github.com/google/qhbm-library/issues/new) describing what you will do. Start a discussion and request to be assigned the issue, ensuring work does not accidentally get duplicated.

Now you can start a new branch for the work, with a name of your choosing:
```
git checkout main
git checkout -b BRANCH_NAME
```

### 3. Develop the feature

Implement your solution to the issue!


### 4. Submit code for review

Request a review from one of the project maintainers:
-zaqqwerty
-jaeyoo

What we are looking for during our review:
Test coverage. Any new functionality should be unit tested. Testing serves many functions:
- tests are an additional source of information beyond the documentation on how to use your features
- allows code to be refactored or updated with confidence, since good tests will break if there are mistakes after an update

Things to keep in mind:
- Maximum PR size of around 300 additional lines is a good general limit. Pull requests (PRs) much larger than this should be broken up into smaller PRs. This makes review easier, and modularity is a sign of a good solution (the ability for it to be understood, thus submitted, in small chunks).
- Respond to all reviewer comments. If you did exactly what was suggested, you can simply say "Done". Otherwise, provide a brief description of how you addressed their concern. Feel free to respond with clarifying questions or open a discussion.