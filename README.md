# Precinct Election Analysis

An exploratory, reproducible system for validating precinct election returns and examining
patterns that may merit contextual review. It implements the three public Election Truth
Alliance (ETA) analysis views—down-ballot difference, vote share by vote count, and turnout
analysis—and adds explicit validation, spatial statistics, digit diagnostics, unsupervised ML,
complete exports, and optional LLM-assisted explanation.

> An anomaly is unusual under a stated model. It is not proof of fraud, misconduct, or an
> incorrect outcome. Aggregate-result analysis is not a risk-limiting audit of ballot evidence.

Copyright © 2026 Rich Hamilton. Licensed under the [GNU General Public License v3.0 only](LICENSE).

## What is implemented

- Schema-driven CSV ingestion that preserves source columns, records a SHA-256 provenance hash,
  reports validation issues, and exports excluded rows rather than silently repairing them.
- ETA-compatible descriptive views:
  - down-ballot difference = `100 × (presidential − same-party down-ballot) / presidential`;
  - candidate share versus that candidate's precinct vote count, with a descriptive trend;
  - turnout/share scatterplots and candidate-vote totals grouped into turnout ranges.
- Optional isolation by mapped vote type (Mail, Early, Election Day, Provisional, and so on).
- Leave-one-out turnout/share residuals, last-digit diagnostics with Holm correction, optional
  Benford diagnostics with preconditions, permutation Moran statistics with local FDR control,
  Isolation Forest, and an opt-in DBSCAN method.
- A Streamlit dashboard and a bounded MCP server suitable for an LLM tool client.
- Optional OpenAI Responses API narratives that may explain and compare computed diagnostics but
  cannot create measurements, override method status, infer causes, or prioritize ballot audits.

The exact assumptions and limits are documented in [docs/methodology.md](docs/methodology.md).
Stepwise operating and architecture instructions are available in
[docs/user-guide.md](docs/user-guide.md), with a standalone rendered edition at
[output/pdf/precinct-election-analysis-user-guide.pdf](output/pdf/precinct-election-analysis-user-guide.pdf).

## Install and run

Python 3.11–3.13 is supported.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dashboard,mcp,dev]"
streamlit run src/dashboard.py
```

Load the fictional sample in the dashboard or upload a CSV matching `data.schema` in
`config.yaml`. Change that mapping for real contests; candidate names and columns are not
hard-coded in the analysis layer.

Useful checks:

```powershell
python -m src.data_ingestion
python -m src.statistical_models
python -m src.ml_models
python -m src.visualization
pytest --cov=src --cov-branch --cov-report=term-missing
ruff check src test
mypy src
```

## Input contract

The default fictional schema expects jurisdiction and precinct identifiers, registered voters,
ballots cast, valid contest votes, and configured candidate vote columns. Coordinates, reported
turnout, vote type, and down-ballot pairs are optional mappings. A same precinct may appear once
per vote type; the internal stable key includes vote type when mapped.

Counts must be numeric, integral, and nonnegative. Candidate totals cannot exceed the mapped
valid-contest total. Ballots cannot exceed registration and contest votes cannot exceed ballots
unless the corresponding jurisdiction-specific override is explicit. Missing critical values,
duplicate stable keys, impossible count relationships, and invalid coordinates are excluded and
reported. No election count, registration value, or coordinate is imputed.

## MCP and LLM use

Start the custom server over standard input/output:

```powershell
election-analysis-mcp
# or
python -m src.mcp_server
```

It exposes health, fictional sample generation, CSV validation, analysis, and bounded narrative
context. Install the LLM adapter separately with `python -m pip install -e ".[llm]"`, set
`OPENAI_API_KEY`, and enable `llm.enabled` in `config.yaml` to generate dashboard summaries.
Raw precinct rows are not sent by the narrative component: it receives method statuses,
diagnostics, aggregate flag counts, and the mandatory interpretation warning.

## Project layout

```text
election-analysis/
├── src/                 # all Python application and MCP source
├── test/                # automated tests
├── docs/                # methodology and limitations
├── .github/workflows/   # continuous integration
├── config.yaml          # explicit schema and method settings
└── pyproject.toml       # package, dependencies, and tool configuration
```

All examples and bundled sample records are fictional. Preserve the exported configuration,
validation report, source hash, exclusions, random seed, and per-method statuses when sharing a
result.
