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

We follow a standard open source pull request workflow.

### 1. Get the source code

Before starting, be sure to follow the instructions to [install from source](https://github.com/google/qhbm-library/blob/main/docs/install.md#install-from-source).

### 2. Update your fork

Inside your local copy of your fork, checkout the mainline development branch:
```
git checkout main
```
Update with any remote changes:
```
git fetch upstream
git merge upstream/main
git push
```

### 3. Create a development branch

Choose an [open issue](https://github.com/google/qhbm-library/issues) or [create one](https://github.com/google/qhbm-library/issues/new) describing what you will do. Start a discussion and request to be assigned the issue, ensuring work does not accidentally get duplicated.

Now you can start a new branch for the work, with a name of your choosing:
```
git checkout main
git checkout -b BRANCH_NAME
```

Let GitHub know about your new branch:
```
git push --set-upstream origin BRANCH_NAME
```

### 3. Develop the feature

Implement your solution to the issue! Make frequent commits and pushes to ensure your work is saved:
```
git add .
git commit -m '<short, informative commit message>'
git push
```

### 4. Prepare code for submission

When you feel your code is ready to submit, first run the preparation script:
```
./scripts/prepare_pr.sh
```
This will do three things.
#### Format
First, it will format your code with `yapf`. This modifies files, so you will need to commit these changes before proceding.
#### Lint
Then, it will run `pylint` to find possible code defects and further style issues. Any issues will need to be corrected before submitting, since our Continuous Integration system runs the same set of checks.
#### Test
Finally, the script runs all the library tests and determines coverage percentages. All tests need to pass before submission.

Next, get any upstream changes:
```
git checkout main
git fetch upstream
git merge upstream/main
git push
git checkout BRANCH_NAME
git merge main
git push
```
There may be conflicts upon merging `main` into your branch. If this occurs, open the conflicted files and choose the correct code to keep. Then, run `git commit` to finish the merge.

### 5. Pull request

Pull requests (PRs) are where code gets reviewed and approved.  Navigate to the [pull request](https://github.com/google/qhbm-library/pulls) page of the library.  If you have recently pushed to a branch of your fork, there should be a yellow banner, with a button labelled ????. Click this button to open the PR interface.

Request a review from one of the project maintainers:
- zaqqwerty

What we are looking for during our review:

#### Documentation
Good 

#### Test coverage
Any new functionality should be unit tested. Testing serves many functions:
- additional source of information beyond the documentation on how to use your feature
- code can be updated in the future with confidence, since good tests will break if code is broken

#### Lint


#### General
Other things to keep in mind:
- Maximum PR size of around 300 additional lines is a good general limit. Pull requests (PRs) much larger than this should be broken up into smaller PRs. This makes review easier, and modularity is a sign of a good solution (the ability for it to be understood, thus submitted, in small chunks).
- Respond to all reviewer comments. If you did exactly what was suggested, you can simply say "Done". Otherwise, provide a brief description of how you addressed their concern. Feel free to respond with clarifying questions or open a discussion.
