from __future__ import annotations

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.dashboard import DashboardApp, _load_payload, _turnout_bounds
from src.data_ingestion import DataValidationError
from src.models import Severity, ValidationReport
from src.workflow import filter_records


def open_app() -> AppTest:
    return AppTest.from_file("src/dashboard.py").run(timeout=30)


def test_initial_page_and_sample_download_render() -> None:
    app = open_app()
    assert not app.exception
    assert app.title[0].value == "Precinct Election Analysis"
    assert app.get("download_button")[0].label == "Download fictional sample CSV"
    assert "not proof of fraud" in app.warning[0].value


def test_sample_load_has_correct_default_candidate_and_controls() -> None:
    app = open_app()
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert app.selectbox[0].label == "Candidate"
    assert app.selectbox[0].value == "Candidate A"
    assert "Reported_Turnout_Percent" not in app.selectbox[0].options
    assert app.multiselect[0].value == [
        "Down-ballot difference",
        "Vote share by vote count",
        "Turnout/share residuals",
    ]


def test_spatial_only_run_export_and_stale_invalidation() -> None:
    app = open_app()
    app.button[0].click().run(timeout=30)
    app.multiselect[0].set_value(["Spatial autocorrelation"])
    app.sidebar.button[0].click().run(timeout=60)
    assert not app.exception
    assert app.session_state.analysis_run.statuses["spatial"].state.value == "successful"
    labels = [item.label for item in app.get("download_button")]
    assert "Download complete analysis bundle" in labels
    app.sidebar.selectbox[0].set_value("Candidate B").run(timeout=30)
    assert app.session_state.analysis_run is None


def test_default_eta_methods_render_all_three_views() -> None:
    app = open_app()
    app.button[0].click().run(timeout=30)
    app.sidebar.button[0].click().run(timeout=60)
    assert not app.exception
    assert all(
        app.session_state.analysis_run.statuses[name].state.value == "successful"
        for name in ("down_ballot_difference", "vote_share_by_count", "turnout_share")
    )
    assert len(app.get("plotly_chart")) >= 5


def test_empty_filter_is_graceful_and_method_status_visible() -> None:
    app = open_app()
    app.button[0].click().run(timeout=30)
    app.multiselect[1].set_value([]).run(timeout=30)
    assert any("select no records" in info.value for info in app.info)
    app.sidebar.button[0].click().run(timeout=30)
    assert not app.exception
    assert all(
        status.state.value == "unavailable"
        for status in app.session_state.analysis_run.statuses.values()
    )


def test_turnout_filter_is_optional_and_handles_missing_or_constant_values() -> None:
    frame = pd.DataFrame({"Jurisdiction": ["A", "B"], "Precinct": ["1", "2"]})
    assert _turnout_bounds(frame) is None
    assert len(filter_records(frame, turnout_range=(0, 100))) == 2

    frame["Reported_Turnout_Percent"] = [50.0, 50.0]
    assert _turnout_bounds(frame) == (50.0, 50.0)
    frame["Reported_Turnout_Percent"] = [None, float("nan")]
    assert _turnout_bounds(frame) is None


def test_empty_method_selection_is_user_visible() -> None:
    app = open_app()
    app.button[0].click().run(timeout=30)
    app.multiselect[0].set_value([])
    app.sidebar.button[0].click().run(timeout=30)
    assert app.session_state.analysis_run is None
    assert any("at least one" in error.value for error in app.error)


@pytest.mark.parametrize("with_report", [False, True])
def test_load_payload_clears_stale_state_and_displays_validation(monkeypatch, with_report) -> None:
    import src.dashboard as dashboard

    report = ValidationReport()
    if with_report:
        report.add("bad", "bad input", Severity.ERROR)
    errors = []
    frames = []
    state = type("State", (), {})()
    state.ingestion = object()
    state.analysis_run = object()
    state.analysis_signature = "old"
    monkeypatch.setattr(
        dashboard,
        "st",
        type(
            "FakeStreamlit",
            (),
            {
                "session_state": state,
                "error": staticmethod(errors.append),
                "dataframe": staticmethod(frames.append),
            },
        )(),
    )
    monkeypatch.setattr(
        dashboard.ElectionDataIngester,
        "process",
        lambda self, payload: (_ for _ in ()).throw(
            DataValidationError("invalid payload", report if with_report else None)
        ),
    )
    _load_payload(b"bad", "fingerprint")
    assert state.ingestion is None
    assert state.analysis_run is None
    assert state.analysis_signature is None
    assert errors == ["Validation failed: invalid payload"]
    assert bool(frames) is with_report


def test_dashboard_facade_delegates_to_public_functions(monkeypatch) -> None:
    import src.dashboard as dashboard

    assert len(DashboardApp.create_sample_data()) == 120
    calls = []
    monkeypatch.setattr(dashboard, "main", lambda: calls.append("main"))
    DashboardApp().run()
    assert calls == ["main"]


def test_file_upload_reruns_only_when_content_changes(generalized_frame) -> None:
    payload = generalized_frame.head(12).to_csv(index=False).encode()
    app = open_app()
    app.get("file_uploader")[0].upload("official.csv", payload, "text/csv").run(timeout=30)
    assert not app.exception
    assert len(app.session_state.ingestion.data) == 12
    fingerprint = app.session_state.loaded_fingerprint
    app.run(timeout=30)
    assert app.session_state.loaded_fingerprint == fingerprint


def test_uploaded_validation_findings_and_all_excluded_rows_are_visible(
    generalized_frame,
) -> None:
    warning_frame = generalized_frame.head(2).copy()
    warning_frame.loc[0, "Latitude"] = None
    app = open_app()
    app.get("file_uploader")[0].upload(
        "warning.csv", warning_frame.to_csv(index=False).encode(), "text/csv"
    ).run(timeout=30)
    assert not app.exception
    assert any("Validation findings" in expander.label for expander in app.expander)

    invalid = generalized_frame.head(1).copy()
    invalid["Votes_Candidate_A"] = invalid["Valid_Contest_Votes"] + 1
    invalid["Votes_Candidate_B"] = 0
    app = open_app()
    app.get("file_uploader")[0].upload(
        "excluded.csv", invalid.to_csv(index=False).encode(), "text/csv"
    ).run(timeout=30)
    assert any("No validated records remain" in error.value for error in app.error)


def test_vote_type_constant_and_unavailable_turnout_controls(generalized_frame) -> None:
    constant = generalized_frame.head(4).copy()
    constant["Vote_Type"] = ["Mail", "Election Day", "Mail", "Election Day"]
    constant["Registered_Voters"] = 100
    constant["Ballots_Cast"] = 50
    constant["Valid_Contest_Votes"] = constant["Ballots_Cast"]
    constant["Votes_Candidate_A"] = constant["Valid_Contest_Votes"] // 2
    constant["Votes_Candidate_B"] = constant["Valid_Contest_Votes"] - constant["Votes_Candidate_A"]
    constant[["Write_In_Votes", "Undervotes", "Overvotes"]] = 0
    constant["Reported_Turnout_Percent"] = (
        constant["Ballots_Cast"] / constant["Registered_Voters"] * 100
    )
    app = open_app()
    app.get("file_uploader")[0].upload(
        "constant.csv", constant.to_csv(index=False).encode(), "text/csv"
    ).run(timeout=30)
    assert any(item.label == "Vote types" for item in app.multiselect)
    assert any("Turnout is constant" in caption.value for caption in app.caption)

    minimal = (
        b"Jurisdiction,Precinct,Valid_Contest_Votes,Votes_Candidate_A,Votes_Candidate_B\n"
        b"A,1,10,4,6\nA,2,12,7,5\nA,3,11,5,6\n"
    )
    app = open_app()
    app.get("file_uploader")[0].upload("minimal.csv", minimal, "text/csv").run(timeout=30)
    assert any("Turnout filter unavailable" in caption.value for caption in app.caption)
    app.multiselect[0].set_value(["Vote share by vote count"])
    app.sidebar.button[0].click().run(timeout=30)
    assert any("coordinates were not mapped" in info.value for info in app.info)


def test_optional_narrative_control_reports_disabled_status() -> None:
    app = open_app()
    app.button[0].click().run(timeout=30)
    app.sidebar.button[0].click().run(timeout=60)
    app.checkbox[0].check().run(timeout=30)
    next(
        button for button in app.button if button.label == "Generate explanatory summary"
    ).click().run(timeout=30)
    assert any("Narrative skipped" in info.value for info in app.info)
