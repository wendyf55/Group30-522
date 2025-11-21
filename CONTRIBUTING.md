# Contributing to Group3-522

This outlines how to propose a change to Group3-522.

### Fixing typos

Small typos or grammatical errors in documentation may be edited directly using
the GitHub web interface, so long as the changes are made in the _source_ file.

* YES: you edit a docstring or comment in a `.py` file in the main package directory (e.g. `src/` or `<project_name>/`).
* NO: you edit generated documentation files (for example HTML or files in `docs/_build/`).

### Prerequisites

Before you make a substantial pull request, you should always open an issue
and make sure someone from the team agrees that it's a problem or a useful
feature. If you've found a bug, create an associated issue and illustrate
the bug with a minimal reproducible example (small code snippet and any
necessary data).

### Pull request process

* We recommend that you create a Git branch for each pull request (PR).
* Check the automated test / CI status (e.g. GitHub Actions) before and after
  making changes. The `README` should contain badges for any continuous
  integration services used by the project.
* New code should follow our Python style guide (PEP 8). You can use tools
  like `black` and/or `ruff`/`flake8` to apply these styles, but please
  don't reformat unrelated code in your PR.
* We use Python docstrings and documentation in the `docs/` folder (Markdown
  or Sphinx). Please update or add docs when you change user-facing behaviour.
* We use `pytest` for tests. Contributions with appropriate test cases are
  much easier to review and accept.
* For user-facing changes, add an entry near the top of `CHANGELOG.md`
  describing the changes made, followed by your GitHub username and links
  to relevant issue(s)/PR(s).

### Code of Conduct

Please note that this project is released with a [Contributor Code of
Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to
abide by its terms.

### Further details

For more information on how to contribute, please read the sections above
or open an issue in the repository if anything is unclear.
