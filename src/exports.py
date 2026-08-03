"""Reproducible CSV/JSON/Markdown export preparation."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, cast

import numpy as np
import pandas as pd

from .models import AnalysisRun, IngestionResult


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(cast(Any, value))
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def anomaly_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Return records flagged by at least one successful precinct-level method."""
    flag_columns = [
        column
        for column in frame
        if column
        in {"Turnout_Share_Flag", "Spatial_Significant", "IF_Anomaly_Flag", "DBSCAN_Noise_Flag"}
    ]
    if not flag_columns:
        return frame.iloc[0:0].copy()
    flags = frame[flag_columns].fillna(False).astype(bool).any(axis=1)
    return frame.loc[flags].copy()


def markdown_report(run: AnalysisRun) -> str:
    """Create a factual report from computed values only."""
    lines = [
        "# Precinct election analysis run",
        "",
        run.metadata["interpretation_warning"],
        "",
        f"Candidate: {run.metadata.get('candidate_label', run.metadata['candidate'])}",
        f"Analysis rows: {len(run.data)}",
        f"Excluded rows: {len(run.excluded)}",
        "",
        "## Method status",
        "",
    ]
    for name, status in run.statuses.items():
        lines.append(f"- {name}: {status.state.value} — {status.message}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Flags identify observations that are unusual under a particular exploratory model. "
            "They require source-data review and contextual investigation. A risk-limiting audit "
            "examines ballot evidence; this aggregate-data workflow does not confirm an outcome.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_export_bundle(ingestion: IngestionResult, run: AnalysisRun) -> bytes:
    """Return a ZIP containing validated data, results, exclusions, diagnostics, and metadata."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("validated_input.csv", ingestion.data.to_csv(index=False))
        archive.writestr("analysis_results.csv", run.data.to_csv(index=False))
        archive.writestr("excluded_records.csv", ingestion.excluded.to_csv(index=False))
        archive.writestr(
            "flagged_diagnostics.csv", anomaly_diagnostics(run.data).to_csv(index=False)
        )
        archive.writestr(
            "run_metadata.json",
            json.dumps(run.metadata, indent=2, default=_json_default, ensure_ascii=False),
        )
        archive.writestr(
            "method_diagnostics.json",
            json.dumps(run.diagnostics, indent=2, default=_json_default, ensure_ascii=False),
        )
        archive.writestr(
            "validation_report.json",
            json.dumps(
                ingestion.report.as_dict(), indent=2, default=_json_default, ensure_ascii=False
            ),
        )
        archive.writestr("report.md", markdown_report(run))
    return buffer.getvalue()
