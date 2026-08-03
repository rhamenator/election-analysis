# Methodology and interpretation limits

## Relationship to ETA

The public [Election Truth Alliance methodology](https://electiontruthalliance.org/our-methodology/)
describes three views. This project implements their observable definitions while applying a more
conservative interpretation boundary.

| View | Implementation | Interpretation |
|---|---|---|
| Down-ballot difference | For each configured same-party pair: `100 × (presidential − down-ballot) / presidential`. Negative values mean the down-ballot candidate received more votes. | Descriptive only. No universal 2–3% expectation is encoded and no flag is created without a documented comparison design. |
| Vote share by vote count | Precinct candidate share is plotted vertically against that candidate's vote count horizontally. An OLS line reports slope, p-value, and R². | The line is descriptive. Precinct size, geography, demographics, and vote-type composition can induce slopes. |
| Turnout | Candidate share is plotted against turnout and candidate vote totals are aggregated into turnout bins. | The turnout axis is not time. A non-bell-shaped distribution is not intrinsically anomalous; discrete precinct populations are mixtures, not necessarily normal samples. |

When a vote-type column is mapped, filters constrain analysis, display, and export to exactly the
same selected types. Comparisons across types remain contextual because voting populations and
administration differ. Historical comparison is not inferred from a single election file.

## Additional diagnostics

### Turnout/share residual model

The system fits an ordinary least-squares polynomial to the configured lower-turnout reference
range and computes leave-one-out expected shares for reference observations. It reports prediction
intervals, studentized residuals, leverage, and a configurable exploratory flag. This is not a
causal model. Aggregation, omitted demographics, geography, precinct design, and jurisdiction
heterogeneity can create large residuals.

### Digit diagnostics

Last digits are tested against a uniform distribution at dataset level and successful tests are
Holm-adjusted. Round-number columns are descriptive. First-digit Benford analysis is disabled by
default and only runs with a configured minimum sample spanning multiple orders of magnitude.
Election counts need not satisfy Benford assumptions; digit tests cannot identify a responsible
mechanism. See Deckert, Myagkov, and Ordeshook, “Benford's Law and the Detection of Election
Fraud,” *Political Analysis* 19(3), 2011, DOI 10.1093/pan/mpr014.

### Spatial diagnostics

Global and local Moran statistics use permutation p-values. Local p-values receive
Benjamini–Hochberg false-discovery-rate adjustment. K-nearest-neighbor weights are explicitly
labeled as a fallback; queen or rook adjacency requires actual polygon geometry and is never
silently approximated. Political behavior is ordinarily spatially clustered, so spatial
association is not evidence of error. Moran's original statistic is described in P. A. P. Moran,
“Notes on Continuous Stochastic Phenomena,” *Biometrika* 37, 1950,
DOI 10.1093/biomet/37.1-2.17.

### Machine learning

Isolation Forest and optional DBSCAN operate on documented, leakage-aware numeric features.
Isolation scores are normalized against the fitted training-score range. DBSCAN rejects degenerate
all-noise outcomes. Neither model estimates the probability of fraud, and their outputs are never
combined into an uncalibrated composite score.

## LLM-assisted interpretation

The LLM is a constrained explanation layer over deterministic output. It receives method status,
diagnostics, aggregate flag counts, and limitations. It can:

- summarize what ran and what was unavailable;
- compare concordant or conflicting computed results;
- explain assumptions in plain language;
- suggest contextual questions or additional data needed for human review.

It cannot calculate missing statistics, invent significance, infer malicious intent, confirm an
outcome, or designate an audit priority. Provider failure does not affect numeric results.

## What aggregate analysis cannot establish

Precinct totals cannot demonstrate that ballots were interpreted or tabulated correctly. A true
risk-limiting audit manually examines a probability sample of trustworthy voter-verifiable paper
records and has a pre-specified chance of escalating to a full hand count when the reported outcome
is wrong. See Lindeman and Stark, “A Gentle Introduction to Risk-Limiting Audits,” *IEEE Security &
Privacy* 10(5), 2012, DOI 10.1109/MSP.2012.56.

Any flagged record should first prompt source verification, reconciliation with official reporting
rules, and jurisdiction-specific context—not a claim about fraud or the certified outcome.
