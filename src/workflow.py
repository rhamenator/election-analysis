"""End-to-end workflow orchestration and consistent analysis scoping."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict

import pandas as pd

from .config import load_config
from .ml_models import MLAnomalyDetector
from .models import AnalysisRun, IngestionResult, MethodState, MethodStatus
from .statistical_models import StatisticalAnomalyDetector

ALL_METHODS = (
    "turnout_share",
    "vote_share_by_count",
    "down_ballot_difference",
    "digit_diagnostics",
    "spatial",
    "isolation_forest",
    "dbscan",
)


def filter_records(
    frame: pd.DataFrame,
    *,
    jurisdictions: Iterable[str] | None = None,
    turnout_range: tuple[float, float] | None = None,
    minimum_ballots: int | None = None,
    vote_types: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return a stable-ID-preserving analysis/display/export scope."""
    mask = pd.Series(True, index=frame.index)
    if jurisdictions is not None:
        selected = set(jurisdictions)
        mask &= frame["Jurisdiction"].isin(selected)
    if turnout_range is not None:
        turnout_column = next(
            (
                column
                for column in ("Calculated_Turnout_Percent", "Reported_Turnout_Percent")
                if column in frame
            ),
            None,
        )
        if turnout_column is not None:
            mask &= pd.to_numeric(frame[turnout_column], errors="coerce").between(*turnout_range)
    if minimum_ballots is not None and "Ballots_Cast" in frame:
        mask &= pd.to_numeric(frame["Ballots_Cast"], errors="coerce") >= minimum_ballots
    if vote_types is not None and "Vote_Type" in frame:
        mask &= frame["Vote_Type"].astype(str).isin(set(vote_types))
    return frame.loc[mask].copy().reset_index(drop=True)


class ElectionAnalysisWorkflow:
    """Run selected methods and retain a complete, serializable run record."""

    def __init__(self, config_path: str | None = "config.yaml") -> None:
        self.config = load_config(config_path)
        self.statistics = StatisticalAnomalyDetector(config_path)
        self.ml = MLAnomalyDetector(config_path)

    def run(
        self,
        ingestion: IngestionResult,
        *,
        candidate_key: str,
        methods: Iterable[str],
        scope: pd.DataFrame | None = None,
    ) -> AnalysisRun:
        requested = list(dict.fromkeys(methods))
        unknown = set(requested) - set(ALL_METHODS)
        if unknown:
            raise ValueError(f"Unknown analysis methods: {sorted(unknown)}")
        candidate = next(
            (item for item in ingestion.schema.candidates if item.key == candidate_key), None
        )
        if candidate is None:
            raise ValueError(f"Unknown candidate key: {candidate_key}")
        data = ingestion.data.copy() if scope is None else scope.copy()
        if data.empty:
            run = AnalysisRun.new(
                data,
                methods=requested,
                candidate=candidate.share_column,
                seed=int(self.config["ml"]["random_state"]),
                config=self.config,
                excluded=ingestion.excluded,
                input_schema=ingestion.schema.source_schema,
            )
            for method in requested:
                run.statuses[method] = MethodStatus(
                    method, MethodState.UNAVAILABLE, "The selected filter scope is empty"
                )
            return self._attach_metadata(run, ingestion, candidate.label)

        statistical_methods = [
            method
            for method in requested
            if method
            in {
                "turnout_share",
                "vote_share_by_count",
                "down_ballot_difference",
                "digit_diagnostics",
                "spatial",
            }
        ]
        ml_methods = [method for method in requested if method in {"isolation_forest", "dbscan"}]
        combined = AnalysisRun.new(
            data,
            methods=requested,
            candidate=candidate.share_column,
            seed=int(self.config["ml"]["random_state"]),
            config=self.config,
            excluded=ingestion.excluded,
            input_schema=ingestion.schema.source_schema,
        )
        if statistical_methods:
            statistical_run = self.statistics.run(
                combined.data,
                candidate.share_column,
                candidate_vote_columns=ingestion.schema.candidate_columns,
                candidate_vote_column=candidate.column,
                down_ballot_pairs=(
                    pair
                    for pair in ingestion.schema.down_ballot_pairs
                    if pair.presidential_candidate_key == candidate.key
                ),
                methods=statistical_methods,
                input_schema=ingestion.schema.source_schema,
                excluded=ingestion.excluded,
            )
            combined.data = statistical_run.data
            combined.statuses.update(statistical_run.statuses)
            combined.diagnostics.update(statistical_run.diagnostics)
        if ml_methods:
            ml_run = self.ml.run(
                combined.data,
                candidate=candidate.share_column,
                methods=ml_methods,
                input_schema=ingestion.schema.source_schema,
                excluded=ingestion.excluded,
            )
            combined.data = ml_run.data
            combined.statuses.update(ml_run.statuses)
            combined.diagnostics.update(ml_run.diagnostics)
        return self._attach_metadata(combined, ingestion, candidate.label)

    @staticmethod
    def _attach_metadata(
        run: AnalysisRun, ingestion: IngestionResult, candidate_label: str
    ) -> AnalysisRun:
        run.metadata.update(
            {
                "candidate_label": candidate_label,
                "analysis_rows": len(run.data),
                "input_rows": ingestion.report.original_rows,
                "excluded_rows": len(ingestion.excluded),
                "provenance": ingestion.provenance,
                "validation": ingestion.report.as_dict(),
                "method_statuses": {name: asdict(status) for name, status in run.statuses.items()},
            }
        )
        return run
