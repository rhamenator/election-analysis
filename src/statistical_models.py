"""Transparent exploratory statistical and spatial diagnostics."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors

from .config import load_config
from .models import AnalysisRun, DownBallotDefinition, MethodState, MethodStatus

logger = logging.getLogger(__name__)


class AnalysisUnavailable(ValueError):
    """Raised when documented preconditions for a method are not met."""


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Return Holm step-down family-wise-error adjusted p-values."""
    values = np.asarray(list(p_values), dtype=float)
    if values.size == 0:
        return values
    if np.any((values < 0) | (values > 1) | ~np.isfinite(values)):
        raise ValueError("p-values must be finite values in [0, 1]")
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    adjusted_ordered = np.maximum.accumulate(
        np.minimum(1.0, ordered * np.arange(len(values), 0, -1))
    )
    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = adjusted_ordered
    return adjusted


def benjamini_hochberg_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Return Benjamini-Hochberg false-discovery-rate adjusted p-values."""
    values = np.asarray(list(p_values), dtype=float)
    if values.size == 0:
        return values
    if np.any((values < 0) | (values > 1) | ~np.isfinite(values)):
        raise ValueError("p-values must be finite values in [0, 1]")
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.arange(1, len(values) + 1)
    adjusted_ordered = np.minimum.accumulate((ordered * len(values) / ranks)[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = np.minimum(1.0, adjusted_ordered)
    return adjusted


class TurnoutShareAnalyzer:
    """Exploratory leave-one-out polynomial turnout/share diagnostic."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        defaults = load_config(None)["statistics"]["turnout_share"]
        self.config = {**defaults, **(config or {})}

    @staticmethod
    def _turnout_column(frame: pd.DataFrame) -> str:
        if "Calculated_Turnout_Percent" in frame:
            return "Calculated_Turnout_Percent"
        if "Reported_Turnout_Percent" in frame:
            return "Reported_Turnout_Percent"
        raise AnalysisUnavailable(
            "Turnout/share analysis requires calculated turnout from ballots and registration, "
            "or an explicitly mapped reported-turnout field"
        )

    @staticmethod
    def _validate_candidate(frame: pd.DataFrame, candidate_column: str) -> None:
        allowed = set(frame.attrs.get("candidate_share_columns", []))
        if not allowed:
            allowed = {
                str(column)
                for column in frame.columns
                if str(column).startswith("Candidate_Share__")
            }
        if candidate_column not in allowed:
            raise AnalysisUnavailable(
                f"{candidate_column!r} is not a configured candidate vote-share field"
            )

    def analyze(
        self, frame: pd.DataFrame, candidate_column: str
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        self._validate_candidate(frame, candidate_column)
        turnout_column = self._turnout_column(frame)
        valid = (
            pd.to_numeric(frame[turnout_column], errors="coerce").between(0, 100)
            & pd.to_numeric(frame[candidate_column], errors="coerce").between(0, 1)
            & (pd.to_numeric(frame["Valid_Contest_Votes"], errors="coerce") > 0)
        )
        work = frame.loc[valid, ["Precinct_ID", turnout_column, candidate_column]].copy()
        minimum = int(self.config["minimum_observations"])
        degree = int(self.config["polynomial_degree"])
        if len(work) < max(minimum, degree + 3):
            raise AnalysisUnavailable(
                f"Need at least {max(minimum, degree + 3)} valid precincts; found {len(work)}"
            )
        turnout = work[turnout_column].to_numpy(float)
        share = work[candidate_column].to_numpy(float)
        if np.unique(turnout).size <= degree:
            raise AnalysisUnavailable(
                "Turnout values do not vary enough to fit the configured model"
            )

        center = float(turnout.mean())
        scale = float(turnout.std())
        if scale == 0:
            raise AnalysisUnavailable("All valid turnout values are identical")
        x = np.vander((turnout - center) / scale, N=degree + 1, increasing=True)
        baseline_quantile = float(self.config["baseline_turnout_quantile"])
        if not 0.5 <= baseline_quantile <= 1:
            raise ValueError("baseline_turnout_quantile must be in [0.5, 1]")
        baseline_limit = float(np.quantile(turnout, baseline_quantile))
        reference = turnout <= baseline_limit
        x_reference = x[reference]
        share_reference = share[reference]
        if len(x_reference) < max(minimum, degree + 3):
            raise AnalysisUnavailable("Too few observations remain in the baseline turnout range")
        if np.linalg.matrix_rank(x_reference) < x_reference.shape[1]:
            raise AnalysisUnavailable("The turnout design matrix is rank deficient")
        xtx_inv = np.linalg.inv(x_reference.T @ x_reference)
        coefficients = xtx_inv @ x_reference.T @ share_reference
        fitted = x @ coefficients
        residual = share - fitted
        leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
        reference_residual = residual[reference]
        residual_df = len(x_reference) - x_reference.shape[1]
        mse = float(np.dot(reference_residual, reference_residual) / residual_df)
        if not np.isfinite(mse) or mse <= np.finfo(float).eps:
            raise AnalysisUnavailable("Residual variance is zero; anomaly residuals are undefined")

        one_minus_h = np.clip(1 - leverage, np.finfo(float).eps, None)
        loo_residual = residual.copy()
        loo_residual[reference] = residual[reference] / one_minus_h[reference]
        loo_expected = share - loo_residual
        prediction_multiplier = 1 + leverage
        prediction_multiplier[reference] = 1 + leverage[reference] / one_minus_h[reference]
        prediction_se = np.sqrt(mse * prediction_multiplier)
        standardized = loo_residual / prediction_se
        confidence = float(self.config["confidence_level"])
        critical = float(stats.t.ppf((1 + confidence) / 2, residual_df))
        threshold = float(self.config["studentized_residual_threshold"])
        leverage_limit = float(self.config["high_leverage_multiplier"]) * x.shape[1] / len(work)

        diagnostics = pd.DataFrame(
            {
                "Precinct_ID": work["Precinct_ID"].to_numpy(),
                "Turnout_Baseline_Expected_Share": loo_expected,
                "Turnout_Share_Residual": loo_residual,
                "Turnout_Share_Studentized_Residual": standardized,
                "Turnout_Share_Prediction_Lower": loo_expected - critical * prediction_se,
                "Turnout_Share_Prediction_Upper": loo_expected + critical * prediction_se,
                "Turnout_Share_High_Leverage": leverage > leverage_limit,
                "Turnout_Share_Flag": np.abs(standardized) > threshold,
            }
        )
        result = frame.merge(diagnostics, on="Precinct_ID", how="left", validate="one_to_one")
        metadata = {
            "model": "leave-one-out ordinary least-squares polynomial regression",
            "turnout_denominator": turnout_column,
            "candidate_share_denominator": "Valid_Contest_Votes",
            "polynomial_degree": degree,
            "baseline_turnout_quantile": baseline_quantile,
            "baseline_turnout_upper_limit": baseline_limit,
            "baseline_observations": int(reference.sum()),
            "confidence_level": confidence,
            "residual_degrees_of_freedom": residual_df,
            "studentized_residual_threshold": threshold,
            "valid_observations": len(work),
            "flagged_observations": int(diagnostics["Turnout_Share_Flag"].sum()),
            "limitation": (
                "Exploratory model only. Jurisdiction structure and demographic/geographic "
                "heterogeneity can produce large residuals without data error or misconduct."
            ),
        }
        return result, metadata


class DigitAnalyzer:
    """Dataset-level last-digit and optional Benford goodness-of-fit diagnostics."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        defaults = load_config(None)["statistics"]["digits"]
        self.config = {**defaults, **(config or {})}

    @staticmethod
    def _integer_values(frame: pd.DataFrame, column: str) -> np.ndarray:
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(float)
        return np.round(numeric[np.isclose(numeric, np.round(numeric))]).astype(int)

    def analyze(
        self, frame: pd.DataFrame, vote_columns: Iterable[str]
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        columns = [column for column in vote_columns if column in frame]
        if not columns:
            raise AnalysisUnavailable("No configured candidate vote-count columns are present")
        minimum = int(self.config["minimum_observations"])
        alpha = float(self.config["alpha"])
        tests: list[dict[str, Any]] = []
        output = frame.copy()
        for column in columns:
            values = self._integer_values(frame, column)
            if len(values) < minimum:
                tests.append(
                    {
                        "column": column,
                        "test": "last_digit_uniformity",
                        "status": "unavailable",
                        "reason": f"Need {minimum} integer observations; found {len(values)}",
                    }
                )
                continue
            counts = np.bincount(np.abs(values) % 10, minlength=10)
            statistic, p_value = stats.chisquare(counts)
            tests.append(
                {
                    "column": column,
                    "test": "last_digit_uniformity",
                    "status": "successful",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "sample_size": len(values),
                    "expected": "uniform digits 0-9",
                }
            )
            output[f"Round_Number_Multiple_10__{column}"] = (
                pd.to_numeric(output[column], errors="coerce") % 10 == 0
            )

            if self.config["benford_enabled"]:
                positive = values[values > 0]
                orders = np.log10(positive.max()) - np.log10(positive.min()) if len(positive) else 0
                benford_minimum = int(self.config["benford_minimum_observations"])
                required_orders = float(self.config["benford_minimum_orders"])
                if len(positive) < benford_minimum or orders < required_orders:
                    tests.append(
                        {
                            "column": column,
                            "test": "benford_first_digit",
                            "status": "unavailable",
                            "reason": (
                                f"Preconditions not met: need {benford_minimum} positive values "
                                f"spanning {required_orders:g} orders of magnitude"
                            ),
                        }
                    )
                else:
                    first = np.floor(positive / 10 ** np.floor(np.log10(positive))).astype(int)
                    observed = np.bincount(first, minlength=10)[1:10]
                    expected_probability = np.log10(1 + 1 / np.arange(1, 10))
                    statistic, p_value = stats.chisquare(
                        observed, f_exp=len(first) * expected_probability
                    )
                    tests.append(
                        {
                            "column": column,
                            "test": "benford_first_digit",
                            "status": "successful",
                            "statistic": float(statistic),
                            "p_value": float(p_value),
                            "sample_size": len(first),
                        }
                    )

        successful = [test for test in tests if test["status"] == "successful"]
        if successful:
            adjusted = holm_adjust(test["p_value"] for test in successful)
            for test, adjusted_p in zip(successful, adjusted, strict=True):
                test["adjusted_p_value"] = float(adjusted_p)
                test["significant"] = bool(adjusted_p < alpha)
        if not successful:
            raise AnalysisUnavailable("No digit diagnostic met its sample-size preconditions")
        return output, {
            "tests": tests,
            "correction": "Holm family-wise error correction across successful digit tests",
            "alpha": alpha,
            "scope": "dataset-level diagnostics; values are not precinct-level anomaly scores",
            "limitation": (
                "Digit patterns can reflect reporting rules, precinct sizes, and administrative "
                "processes. They are not evidence of manipulation."
            ),
        }


class VoteShareByCountAnalyzer:
    """Describe candidate share as precinct candidate vote count changes."""

    def analyze(
        self, frame: pd.DataFrame, candidate_share: str, candidate_votes: str
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        required = ["Precinct_ID", candidate_share, candidate_votes]
        missing = [column for column in required if column not in frame]
        if missing:
            raise AnalysisUnavailable(f"Vote-share-by-count fields are missing: {missing}")
        work = frame[required].copy()
        work[candidate_share] = pd.to_numeric(work[candidate_share], errors="coerce")
        work[candidate_votes] = pd.to_numeric(work[candidate_votes], errors="coerce")
        work = work.dropna()
        if len(work) < 3:
            raise AnalysisUnavailable(
                "At least three valid vote-count/share observations are required"
            )
        if work[candidate_votes].nunique() < 2:
            raise AnalysisUnavailable("Candidate vote counts have no variation")

        regression = stats.linregress(work[candidate_votes], work[candidate_share])
        expected = regression.intercept + regression.slope * work[candidate_votes]
        key = candidate_share.removeprefix("Candidate_Share__")
        expected_column = f"Vote_Count_Trend_Expected_Share__{key}"
        residual_column = f"Vote_Count_Trend_Residual__{key}"
        additions = pd.DataFrame(
            {
                "Precinct_ID": work["Precinct_ID"],
                expected_column: expected,
                residual_column: work[candidate_share] - expected,
            }
        )
        output = frame.merge(additions, on="Precinct_ID", how="left", validate="one_to_one")
        return output, {
            "model": "ordinary least-squares linear descriptive trend",
            "candidate_vote_column": candidate_votes,
            "candidate_share_column": candidate_share,
            "valid_observations": len(work),
            "slope_share_per_100_votes": float(regression.slope * 100),
            "intercept": float(regression.intercept),
            "r_squared": float(regression.rvalue**2),
            "slope_p_value": float(regression.pvalue),
            "expected_share_column": expected_column,
            "residual_column": residual_column,
            "scope": "exploratory descriptive relationship; this method creates no anomaly flag",
            "limitation": (
                "A slope can result from geography, demographics, precinct design, vote type, "
                "or other ordinary heterogeneity. It is not evidence of manipulation."
            ),
        }


class DownBallotDifferenceAnalyzer:
    """Compute ETA-compatible presidential versus same-party down-ballot differences."""

    def analyze(
        self,
        frame: pd.DataFrame,
        presidential_column: str,
        pairs: Iterable[DownBallotDefinition],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        selected = [pair for pair in pairs if pair.down_ballot_column in frame]
        if not selected:
            raise AnalysisUnavailable("No configured down-ballot comparison is available")
        if presidential_column not in frame:
            raise AnalysisUnavailable(f"Presidential vote column is missing: {presidential_column}")
        output = frame.copy()
        summaries: list[dict[str, Any]] = []
        presidential = pd.to_numeric(output[presidential_column], errors="coerce")
        for pair in selected:
            down_ballot = pd.to_numeric(output[pair.down_ballot_column], errors="coerce")
            valid = presidential.gt(0) & down_ballot.notna() & down_ballot.ge(0)
            vote_column = f"Down_Ballot_Difference_Votes__{pair.key}"
            percent_column = f"Down_Ballot_Difference_Percent__{pair.key}"
            output[vote_column] = (presidential - down_ballot).where(valid)
            output[percent_column] = (100 * output[vote_column] / presidential).where(valid)
            values = output.loc[valid, percent_column]
            if values.empty:
                summaries.append({"key": pair.key, "label": pair.label, "status": "unavailable"})
                continue
            total_presidential = float(presidential[valid].sum())
            total_down_ballot = float(down_ballot[valid].sum())
            summaries.append(
                {
                    "key": pair.key,
                    "label": pair.label,
                    "status": "successful",
                    "presidential_vote_column": presidential_column,
                    "down_ballot_vote_column": pair.down_ballot_column,
                    "valid_observations": int(valid.sum()),
                    "aggregate_difference_votes": total_presidential - total_down_ballot,
                    "aggregate_difference_percent_of_presidential": (
                        100 * (total_presidential - total_down_ballot) / total_presidential
                    ),
                    "median_precinct_difference_percent": float(values.median()),
                    "negative_difference_precincts": int((values < 0).sum()),
                    "difference_vote_column": vote_column,
                    "difference_percent_column": percent_column,
                }
            )
        if not any(item["status"] == "successful" for item in summaries):
            raise AnalysisUnavailable("No valid positive presidential counts support a comparison")
        return output, {
            "definition": (
                "100 * (presidential votes - same-party down-ballot votes) / presidential votes"
            ),
            "comparisons": summaries,
            "scope": "descriptive comparison; this method creates no anomaly flag",
            "limitation": (
                "Ballot roll-off, split-ticket voting, candidate effects, contest eligibility, and "
                "vote-type composition can all produce differences. Historical or contextual "
                "comparisons require separately documented data."
            ),
        }


class SpatialAnalyzer:
    """Permutation-based Moran diagnostics with explicit weights semantics."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        defaults = load_config(None)["statistics"]["spatial"]
        self.config = {**defaults, **(config or {})}

    def _knn_weights(self, coordinates: np.ndarray) -> tuple[np.ndarray, int]:
        n = len(coordinates)
        k = min(int(self.config["knn_neighbors"]), n - 1)
        if k < 1:
            raise AnalysisUnavailable("At least two valid coordinate pairs are required")
        # With X=None, scikit-learn excludes each fitted observation from its own
        # neighbor list. Request exactly k rather than manually dropping a column.
        neighbors = NearestNeighbors(n_neighbors=k).fit(coordinates)
        indices = neighbors.kneighbors(return_distance=False)
        weights = np.zeros((n, n), dtype=float)
        rows = np.repeat(np.arange(n), k)
        weights[rows, indices.ravel()] = 1.0
        weights /= weights.sum(axis=1, keepdims=True)
        return weights, k

    def _polygon_weights(self, frame: pd.DataFrame, kind: str) -> tuple[np.ndarray, int]:
        if "Geometry" not in frame:
            raise AnalysisUnavailable(
                f"{kind} adjacency requires a mapped Geometry polygon column; KNN is not "
                "silently substituted"
            )
        try:
            from libpysal.weights import Queen, Rook
        except ImportError as exc:
            raise AnalysisUnavailable(
                "Polygon adjacency requires the optional spatial dependency group"
            ) from exc
        builder = Queen if kind == "queen" else Rook
        weights_object = builder.from_iterable(frame["Geometry"])
        weights_object.transform = "r"
        return np.asarray(weights_object.full()[0], dtype=float), -1

    def analyze(self, frame: pd.DataFrame, variable: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        if variable not in frame:
            raise AnalysisUnavailable(f"Spatial variable is missing: {variable}")
        coordinate_columns = ["Latitude", "Longitude"]
        if not all(column in frame for column in coordinate_columns):
            raise AnalysisUnavailable("Spatial analysis requires mapped latitude and longitude")
        valid = (
            frame[coordinate_columns].notna().all(axis=1)
            & pd.to_numeric(frame[variable], errors="coerce").notna()
        )
        work = frame.loc[valid, ["Precinct_ID", *coordinate_columns, variable]].copy()
        if len(work) < 3:
            raise AnalysisUnavailable(
                "At least three records with coordinates and values are required; "
                f"found {len(work)}"
            )

        kind = str(self.config["weights_type"])
        if kind == "knn":
            weights, effective_k = self._knn_weights(
                work[["Longitude", "Latitude"]].to_numpy(float)
            )
            weights_label = f"K-nearest-neighbor fallback (k={effective_k})"
        else:
            weights, effective_k = self._polygon_weights(frame.loc[valid], kind)
            weights_label = f"{kind} polygon adjacency"

        values = pd.to_numeric(work[variable], errors="coerce").to_numpy(float)
        centered = values - values.mean()
        denominator = float(centered @ centered)
        if denominator <= np.finfo(float).eps:
            raise AnalysisUnavailable("Spatial variable has zero variance")
        n = len(values)
        lag = weights @ centered
        global_i = float(centered @ lag / denominator)
        local_i = centered * lag / (denominator / n)
        permutations = int(self.config["permutations"])
        rng = np.random.default_rng(int(self.config["random_state"]))
        perm_global = np.empty(permutations)
        perm_local = np.empty((permutations, n))
        for index in range(permutations):
            permuted = rng.permutation(centered)
            permuted_lag = weights @ permuted
            perm_global[index] = float(permuted @ permuted_lag / denominator)
            # Conditional randomization: keep each focal observation fixed while
            # permuting the values available to its spatial lag.
            perm_local[index] = centered * permuted_lag / (denominator / n)
        global_p = float(
            (np.count_nonzero(np.abs(perm_global) >= abs(global_i)) + 1) / (permutations + 1)
        )
        local_p = (np.count_nonzero(np.abs(perm_local) >= np.abs(local_i), axis=0) + 1) / (
            permutations + 1
        )
        adjusted = benjamini_hochberg_adjust(local_p)
        alpha = float(self.config["alpha"])
        cluster_type = np.select(
            [
                (centered >= 0) & (lag >= 0),
                (centered < 0) & (lag < 0),
                (centered >= 0) & (lag < 0),
                (centered < 0) & (lag >= 0),
            ],
            [
                "high-high cluster",
                "low-low cluster",
                "high-low spatial outlier",
                "low-high spatial outlier",
            ],
            default="unclassified",
        )
        diagnostics = pd.DataFrame(
            {
                "Precinct_ID": work["Precinct_ID"].to_numpy(),
                "Local_Moran_I": local_i,
                "Local_Moran_Raw_P": local_p,
                "Local_Moran_Adjusted_P": adjusted,
                "Spatial_Pattern": cluster_type,
                "Spatial_Significant": adjusted < alpha,
            }
        )
        output = frame.merge(diagnostics, on="Precinct_ID", how="left", validate="one_to_one")
        return output, {
            "weights": weights_label,
            "valid_observations": n,
            "excluded_for_missing_coordinates_or_value": int((~valid).sum()),
            "global_moran_i": global_i,
            "global_expected_i": -1 / (n - 1),
            "global_permutation_p": global_p,
            "permutations": permutations,
            "local_correction": "Benjamini-Hochberg false-discovery-rate correction",
            "alpha": alpha,
            "significant_local_patterns": int((adjusted < alpha).sum()),
            "limitation": (
                "Spatial association may reflect ordinary geographic political clustering and "
                "does not by itself identify error or misconduct."
            ),
        }


# Backward-compatible names for callers that imported the proof-of-concept classes.
ShpilkinAnalyzer = TurnoutShareAnalyzer
EntropyAnalyzer = DigitAnalyzer


class StatisticalAnomalyDetector:
    """Run selected diagnostics without creating an unjustified composite score."""

    def __init__(self, config_path: str | None = "config.yaml") -> None:
        self.full_config = load_config(config_path)
        statistical = self.full_config["statistics"]
        self.turnout_share = TurnoutShareAnalyzer(statistical["turnout_share"])
        self.shpilkin = self.turnout_share
        self.digits = DigitAnalyzer(statistical["digits"])
        self.entropy = self.digits
        self.spatial = SpatialAnalyzer(statistical["spatial"])
        self.vote_share_by_count = VoteShareByCountAnalyzer()
        self.down_ballot_difference = DownBallotDifferenceAnalyzer()
        self.last_run: AnalysisRun | None = None

    def run(
        self,
        frame: pd.DataFrame,
        candidate_column: str,
        *,
        candidate_vote_columns: Iterable[str],
        candidate_vote_column: str | None = None,
        down_ballot_pairs: Iterable[DownBallotDefinition] = (),
        methods: Iterable[str] = (
            "turnout_share",
            "vote_share_by_count",
            "digit_diagnostics",
            "spatial",
        ),
        input_schema: str = "configured",
        excluded: pd.DataFrame | None = None,
    ) -> AnalysisRun:
        requested = list(dict.fromkeys(methods))
        unknown = set(requested) - {
            "turnout_share",
            "vote_share_by_count",
            "down_ballot_difference",
            "digit_diagnostics",
            "spatial",
        }
        if unknown:
            raise ValueError(f"Unknown statistical methods: {sorted(unknown)}")
        seed = int(self.full_config["statistics"]["spatial"]["random_state"])
        run = AnalysisRun.new(
            frame,
            methods=requested,
            candidate=candidate_column,
            seed=seed,
            config=self.full_config["statistics"],
            excluded=excluded,
            input_schema=input_schema,
        )
        run.data.attrs["candidate_share_columns"] = [
            str(column) for column in frame.columns if str(column).startswith("Candidate_Share__")
        ]
        for method in requested:
            try:
                if method == "turnout_share":
                    run.data, diagnostic = self.turnout_share.analyze(run.data, candidate_column)
                elif method == "vote_share_by_count":
                    if candidate_vote_column is None:
                        raise AnalysisUnavailable(
                            "The selected candidate vote column is unavailable"
                        )
                    run.data, diagnostic = self.vote_share_by_count.analyze(
                        run.data, candidate_column, candidate_vote_column
                    )
                elif method == "down_ballot_difference":
                    if candidate_vote_column is None:
                        raise AnalysisUnavailable(
                            "The selected presidential vote column is unavailable"
                        )
                    run.data, diagnostic = self.down_ballot_difference.analyze(
                        run.data, candidate_vote_column, down_ballot_pairs
                    )
                elif method == "digit_diagnostics":
                    run.data, diagnostic = self.digits.analyze(run.data, candidate_vote_columns)
                else:
                    run.data, diagnostic = self.spatial.analyze(run.data, candidate_column)
                run.diagnostics[method] = diagnostic
                run.statuses[method] = MethodStatus(
                    method, MethodState.SUCCESS, "Completed", diagnostic
                )
            except AnalysisUnavailable as exc:
                run.statuses[method] = MethodStatus(method, MethodState.UNAVAILABLE, str(exc))
            except Exception as exc:  # method isolation is intentional and surfaced to callers
                logger.exception("%s analysis failed", method)
                run.statuses[method] = MethodStatus(method, MethodState.FAILED, str(exc))
        self.last_run = run
        return run

    def run_full_analysis(
        self, frame: pd.DataFrame, candidate_column: str = "Candidate_Share__candidate_a"
    ) -> pd.DataFrame:
        """Compatibility wrapper returning records; statuses remain available on ``last_run``."""
        vote_columns = [str(column) for column in frame.columns if str(column).startswith("Votes_")]
        self.last_run = self.run(
            frame,
            candidate_column,
            candidate_vote_columns=vote_columns,
            candidate_vote_column=vote_columns[0] if vote_columns else None,
        )
        return self.last_run.data


def main() -> None:
    from .data_ingestion import ElectionDataIngester
    from .sample_data import generalized_sample_data

    ingestion = ElectionDataIngester().process(
        generalized_sample_data().to_csv(index=False).encode()
    )
    detector = StatisticalAnomalyDetector()
    run = detector.run(
        ingestion.data,
        ingestion.schema.candidates[0].share_column,
        candidate_vote_columns=ingestion.schema.candidate_columns,
        candidate_vote_column=ingestion.schema.candidates[0].column,
        input_schema=ingestion.schema.source_schema,
    )
    print({name: asdict(status) for name, status in run.statuses.items()})


if __name__ == "__main__":
    main()
