# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Pull request 97

Link: <https://github.com/wendyf55/Group30-522/pull/97>

GITHUB ISSUE FOR PEER REVIEW: <https://github.com/UBC-MDS/data-analysis-review-2025/issues/51#issuecomment-3638951656>

### Added

- `make` to `environment.yaml` to enable `make analysis` command in Docker container (fixes "make: not found" error)
- `pytest` to `environment.yaml` for running unit tests
- Prerequisites section to README.md with explicit requirements (Docker Desktop, Git, Terminal access, optional Conda)


## Pull request 92

Link: <https://github.com/wendyf55/Group30-522/pull/92>

GITHUB ISSUE FOR PEER REVIEW: <https://github.com/UBC-MDS/data-analysis-review-2025/issues/55#issuecomment-3639184190>

### Added

- Interpretations in non-linear model section linking multicollinearity to our non-linear models results


## Pull request 91

Link: <https://github.com/wendyf55/Group30-522/pull/91>

GITHUB ISSUE FOR PEER REVIEW: <https://github.com/UBC-MDS/data-analysis-review-2025/issues/41#issuecomment-3638437456>

### Added

- Sentence linking multicollinearity in EDA to our usage of non-linear models


## Pull request 89

Link: <https://github.com/wendyf55/Group30-522/pull/89>

GITHUB ISSUE FOR PEER REVIEW: <https://github.com/UBC-MDS/data-analysis-review-2025/issues/41#issuecomment-3638437456>

### Added

- Description for data validation

### Fixed

- In the generated abalone_rings.html, some figures do not display. In particular, Figure 4 shows a 403 Forbidden error: "Blocking request from unknown origin." One way to fix this is to add the following to the YAML header: embed-resources: true
