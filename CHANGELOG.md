GITHUB ISSUE FOR PEER REVIEW: <https://github.com/UBC-MDS/data-analysis-review-2025/issues/41#issuecomment-3638437456>

Added:

- description for data validation
- In the generated abalone_rings.html, some figures do not display. In particular, Figure 4 shows a 403 Forbidden error: "Blocking request from unknown origin." One way to fix this is to add the following to the YAML header: embed-resources: true

PR for evidence of change: <https://github.com/wendyf55/Group30-522/pull/89>


GITHUB ISSUE FOR PEER REVIEW: <https://github.com/UBC-MDS/data-analysis-review-2025/issues/41#issuecomment-3638437456>

Added:
- sentence linking multicollinearity to our usage of non-linear models


GITHUB ISSUE FOR PEER REVIEW: <https://github.com/UBC-MDS/data-analysis-review-2025/issues/55#issuecomment-3639184190>

Added:
- interpretations linking multicollinearity to our non-linear models results


GITHUB ISSUE FOR PEER REVIEW: <https://github.com/UBC-MDS/data-analysis-review-2025/issues/51#issuecomment-3638951656>

Added:
- `make` to `environment.yaml` to enable `make analysis` command in Docker container (fixes "make: not found" error)
- `pytest` to `environment.yaml` for running unit tests
- Prerequisites section to README.md with explicit requirements (Docker Desktop, Git, Terminal access, optional Conda)
