from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.exports import _json_default, anomaly_diagnostics, build_export_bundle, markdown_report
from src.models import MethodState
from src.workflow import ElectionAnalysisWorkflow, filter_records


def test_filter_records_controls_single_scope(ingestion) -> None:
    jurisdiction = ingestion.data["Jurisdiction"].iloc[0]
    filtered = filter_records(
        ingestion.data,
        jurisdictions=[jurisdiction],
        turnout_range=(0, 100),
        minimum_ballots=500,
    )
    assert set(filtered["Jurisdiction"]) == {jurisdiction}
    assert (filtered["Ballots_Cast"] >= 500).all()
    assert filtered.index.equals(pd.RangeIndex(len(filtered)))


def test_filter_records_can_isolate_vote_types(ingestion) -> None:
    frame = ingestion.data.copy()
    frame["Vote_Type"] = ["Mail" if i % 2 else "Election Day" for i in range(len(frame))]
    filtered = filter_records(frame, vote_types=["Mail"])
    assert set(filtered["Vote_Type"]) == {"Mail"}


def test_workflow_runs_eta_method_set(ingestion) -> None:
    run = ElectionAnalysisWorkflow().run(
        ingestion,
        candidate_key="candidate_a",
        methods=["down_ballot_difference", "vote_share_by_count", "turnout_share"],
    )
    assert all(status.state == MethodState.SUCCESS for status in run.statuses.values())
    assert "Down_Ballot_Difference_Percent__candidate_a_down_ballot" in run.data


def test_export_json_converter_supported_types() -> None:
    from src.models import MethodStatus

    assert _json_default(MethodState.SUCCESS) == "successful"
    assert _json_default(MethodStatus("x", MethodState.SUCCESS, "ok"))["method"] == "x"
    assert _json_default(np.int64(2)) == 2
    assert _json_default(np.array([1, 2])) == [1, 2]
    assert _json_default(pd.Timestamp("2024-01-01")).startswith("2024-01-01")
    with pytest.raises(TypeError):
        _json_default(object())


def test_full_workflow_records_metadata_and_independent_statuses(ingestion) -> None:
    run = ElectionAnalysisWorkflow().run(
        ingestion,
        candidate_key="candidate_a",
        methods=["turnout_share", "isolation_forest", "spatial"],
    )
    assert set(run.statuses) == {"turnout_share", "isolation_forest", "spatial"}
    assert all(status.state == MethodState.SUCCESS for status in run.statuses.values())
    assert run.metadata["candidate_label"] == "Candidate A"
    assert run.metadata["provenance"]["sha256"]
    assert run.metadata["method_statuses"]
    assert "fraud" in run.metadata["interpretation_warning"]


def test_spatial_only_and_empty_scope(ingestion) -> None:
    workflow = ElectionAnalysisWorkflow()
    spatial = workflow.run(ingestion, candidate_key="candidate_a", methods=["spatial"])
    assert spatial.statuses["spatial"].state == MethodState.SUCCESS
    empty = workflow.run(
        ingestion,
        candidate_key="candidate_a",
        methods=["spatial", "isolation_forest"],
        scope=ingestion.data.iloc[0:0],
    )
    assert all(status.state == MethodState.UNAVAILABLE for status in empty.statuses.values())


def test_workflow_rejects_unknown_candidate_and_method(ingestion) -> None:
    workflow = ElectionAnalysisWorkflow()
    with pytest.raises(ValueError, match="candidate"):
        workflow.run(ingestion, candidate_key="nobody", methods=["spatial"])
    with pytest.raises(ValueError, match="methods"):
        workflow.run(ingestion, candidate_key="candidate_a", methods=["bogus"])


def test_irrelevant_preserved_column_does_not_change_if_scores(ingestion) -> None:
    workflow = ElectionAnalysisWorkflow()
    first = workflow.run(ingestion, candidate_key="candidate_a", methods=["isolation_forest"])
    ingestion.data["Irrelevant text"] = "hello"
    second = workflow.run(ingestion, candidate_key="candidate_a", methods=["isolation_forest"])
    assert first.data["IF_Anomaly_Score"].to_numpy() == pytest.approx(
        second.data["IF_Anomaly_Score"].to_numpy()
    )


def test_export_bundle_is_complete_and_consistent(ingestion) -> None:
    run = ElectionAnalysisWorkflow().run(
        ingestion, candidate_key="candidate_a", methods=["turnout_share", "isolation_forest"]
    )
    data = build_export_bundle(ingestion, run)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert set(archive.namelist()) == {
            "validated_input.csv",
            "analysis_results.csv",
            "excluded_records.csv",
            "flagged_diagnostics.csv",
            "run_metadata.json",
            "method_diagnostics.json",
            "validation_report.json",
            "report.md",
        }
        metadata = json.loads(archive.read("run_metadata.json"))
        assert metadata["candidate_label"] == "Candidate A"
        report = archive.read("report.md").decode()
        assert "not evidence of fraud" in report
        assert "risk-limiting audit" in report


def test_anomaly_diagnostics_empty_without_flags(ingestion) -> None:
    assert anomaly_diagnostics(ingestion.data).empty
    run = ElectionAnalysisWorkflow().run(
        ingestion, candidate_key="candidate_a", methods=["isolation_forest"]
    )
    flagged = anomaly_diagnostics(run.data)
    assert len(flagged) == int(run.data["IF_Anomaly_Flag"].sum())
    assert markdown_report(run).endswith("\n")
