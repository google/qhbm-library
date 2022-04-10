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

Before starting, be sure to follow the instructions to [install from source](https://github.com/google/qhbm-library/blob/main/docs/INSTALL.md#install-from-source).


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


### 4. Develop the feature

Implement your solution to the issue! Make frequent commits and pushes to ensure your work is saved:
```
git add .
git commit -m '<short, informative commit message>'
git push
```


### 5. Prepare code for review

When you feel your code is ready to submit, first run the preparation script:
```
./scripts/prepare_pr.sh
```
This will do three things.
#### Format
First, it will format your code with `yapf`. This modifies files, so you will need to commit these changes before proceding.
#### Lint
Then, it will run `pylint` to find possible code defects and further style issues. More information on our linter can be found [here](https://google.github.io/styleguide/pyguide.html#21-lint). Any issues will need to be corrected before submitting, since our Continuous Integration system runs the same set of checks.
#### Test
Finally, the script runs all the library tests and determines coverage percentages. All tests need to pass before submission.


### 6. Merge any upstream changes

Multiple developers may be updating the same code in parallel. Thus, you will need to fetch any upstream changes:
```
git checkout main
git fetch upstream
git merge upstream/main
```
If you get the message "Already up to date.", you can continue to the next section.

Otherwise, you will need to merge changes into your development branch:
```
git checkout BRANCH_NAME
git merge main
```
There may be conflicts upon merging `main` into your branch. If this occurs, open the conflicted files and choose the correct code to keep. Then, run `git commit` to finish the merge.


### 7. Pull request

Pull requests (PRs) are how code gets reviewed and approved. To start a PR, first update GitHub with your branch changes:
```
git checkout BRANCH_NAME
git push
```
Then, navigate to the [pull request](https://github.com/google/qhbm-library/pulls) page of the library. Since you recently pushed changes, there should be a yellow banner on the webpage, with a button labelled "Compare & pull request". Click this button to open the PR creation interface.

Edit the title of the PR to describe what the PR accomplishes. In the larger "comment" field below, go into more depth on what the PR changes or adds. Be sure to tag the issue associated with this PR, using the #<issue number> syntax.

Request a review from one of the project maintainers. Click the gear on the right next to the "Reviewers" tab. Scroll down the list and select any of the following current project maintainers:
- zaqqwerty
  
Then you can click "Create pull request". 

It might be that during the creation of the pull request, the "Reviewers" tab is not visible. If this is the case, click "Create pull request" and add the reviewer in the following screen. This can be done in the panel on the right-hand side. 

Beyond the checks, here is what we are looking for during our review:

#### Code correctness
Does your code resolve the assigned issue? If not, is it clear how the code forms a part of such a solution? Performance is another aspect we consider when evaluating correctness.

#### Code structure
Is your code easy to read? To support this, we try to follow the standard [Google open source guidelines](https://google.github.io/styleguide/pyguide.html) for Python code. Even within the bounds of the style guide, there can still be more or less clear ways of coding.

#### Comments and Documentation
Good documentation helps future programmers understand how to use your feature. See [this tutorial](https://realpython.com/documenting-python-code/) for an overview, and see the [style guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for the specific structure we want in our docstrings.

#### Test new features

Any new functionality should be unit tested. The main idea is to demonstrate that your new code does what it claims to do.

Beyond demonstrating correctness:
- Tests are additional sources of information beyond the documentation about how to use your feature.
- Tests act as guardrails for future code updates. Tested code can be updated with confidence, since good tests will break if the tested code gets broken.

#### General
Other things to keep in mind:
- Maximum PR size of around 300 additional lines is a good general limit. Pull requests (PRs) much larger than this should likely be broken up into smaller PRs. This makes review easier, and modularity is a sign of a good solution (the ability for it to be understood, thus submitted, in small chunks).
- Respond to all reviewer comments. If you did exactly what was suggested, you can simply say "Done". Otherwise, provide a brief description of how you addressed their concern. Feel free to respond with clarifying questions or to open a discussion instead. The purpose of these responses is to ensure you and the reviewer are in agreement about the code before it gets accepted.
