from __future__ import annotations

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.dashboard import _turnout_bounds
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
