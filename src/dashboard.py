"""Streamlit dashboard for the validated election-analysis workflow."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

import pandas as pd
import streamlit as st

from src.data_ingestion import DataValidationError, ElectionDataIngester
from src.exports import build_export_bundle
from src.llm_integration import AnomalyReasoningAgent
from src.sample_data import generalized_sample_data
from src.visualization import ComprehensiveVisualizer
from src.workflow import ElectionAnalysisWorkflow, filter_records

METHOD_LABELS = {
    "Down-ballot difference": "down_ballot_difference",
    "Vote share by vote count": "vote_share_by_count",
    "Turnout/share residuals": "turnout_share",
    "Dataset-level digit diagnostics": "digit_diagnostics",
    "Spatial autocorrelation": "spatial",
    "Isolation Forest": "isolation_forest",
    "DBSCAN (disabled by default)": "dbscan",
}


def create_sample_data(rows: int = 120) -> pd.DataFrame:
    """Public sample factory used by the dashboard and tests."""
    return generalized_sample_data(rows)


def _state_defaults() -> None:
    defaults: dict[str, Any] = {
        "ingestion": None,
        "analysis_run": None,
        "analysis_signature": None,
        "loaded_fingerprint": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _load_payload(payload: bytes, fingerprint: str) -> None:
    try:
        ingestion = ElectionDataIngester().process(payload)
    except DataValidationError as exc:
        st.session_state.ingestion = None
        st.session_state.analysis_run = None
        st.session_state.analysis_signature = None
        st.error(f"Validation failed: {exc}")
        if exc.report:
            st.dataframe(pd.DataFrame(exc.report.as_dict()["issues"]))
        return
    st.session_state.ingestion = ingestion
    st.session_state.loaded_fingerprint = fingerprint
    st.session_state.analysis_run = None
    st.session_state.analysis_signature = None


def _signature(
    source_hash: str,
    candidate_key: str,
    methods: list[str],
    jurisdictions: list[str],
    turnout_range: tuple[float, float],
    minimum_ballots: int,
    vote_types: list[str],
) -> str:
    payload = json.dumps(
        [
            source_hash,
            candidate_key,
            methods,
            jurisdictions,
            turnout_range,
            minimum_ballots,
            vote_types,
        ],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _status_table(run: Any) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Method": name,
                "Status": status.state.value,
                "Message": status.message,
            }
            for name, status in run.statuses.items()
        ]
    )


class DashboardApp:
    """Small facade retained for programmatic and Streamlit testing."""

    @staticmethod
    def create_sample_data() -> pd.DataFrame:
        return create_sample_data()

    def run(self) -> None:
        main()


def main() -> None:
    st.set_page_config(page_title="Precinct Election Analysis", page_icon="📊", layout="wide")
    _state_defaults()
    st.title("Precinct Election Analysis")
    st.warning(
        "An anomaly is unusual under a stated model. It is not proof of fraud, misconduct, "
        "or an incorrect outcome. This aggregate-data application is not a risk-limiting audit."
    )
    st.caption(
        "Upload official precinct totals only with a documented column mapping. Source values "
        "are preserved; missing election counts and coordinates are never imputed."
    )

    sample = create_sample_data()
    st.download_button(
        "Download fictional sample CSV",
        sample.to_csv(index=False),
        file_name="fictional_michigan_compatible_sample.csv",
        mime="text/csv",
    )
    if st.button("Load fictional sample", help="Loads the same sample shown in the download"):
        payload = sample.to_csv(index=False).encode()
        _load_payload(payload, hashlib.sha256(payload).hexdigest())

    upload = st.sidebar.file_uploader("Upload precinct CSV", type=["csv"])
    if upload is not None:
        payload = upload.getvalue()
        fingerprint = hashlib.sha256(payload).hexdigest()
        if fingerprint != st.session_state.loaded_fingerprint:
            _load_payload(payload, fingerprint)

    ingestion = st.session_state.ingestion
    if ingestion is None:
        st.info("Upload a CSV or load the fictional sample to begin.")
        return

    if ingestion.report.issues:
        with st.expander(
            f"Validation findings ({len(ingestion.report.errors)} errors, "
            f"{len(ingestion.report.warnings)} warnings)",
            expanded=bool(ingestion.report.errors),
        ):
            st.dataframe(pd.DataFrame(ingestion.report.as_dict()["issues"]), width="stretch")
    if ingestion.data.empty:
        st.error("No validated records remain after exclusions.")
        return

    labels = {candidate.label: candidate.key for candidate in ingestion.schema.candidates}
    candidate_label = st.sidebar.selectbox("Candidate", list(labels), index=0)
    candidate_key = labels[candidate_label]
    selected_labels = st.sidebar.multiselect(
        "Analysis methods",
        list(METHOD_LABELS),
        default=["Down-ballot difference", "Vote share by vote count", "Turnout/share residuals"],
    )
    methods = [METHOD_LABELS[label] for label in selected_labels]

    jurisdictions = sorted(ingestion.data["Jurisdiction"].dropna().astype(str).unique())
    selected_jurisdictions = st.sidebar.multiselect(
        "Jurisdictions", jurisdictions, default=jurisdictions
    )
    vote_types: list[str] = []
    if "Vote_Type" in ingestion.data:
        available_vote_types = sorted(ingestion.data["Vote_Type"].dropna().astype(str).unique())
        vote_types = st.sidebar.multiselect(
            "Vote types", available_vote_types, default=available_vote_types
        )
    turnout_column = (
        "Calculated_Turnout_Percent"
        if "Calculated_Turnout_Percent" in ingestion.data
        else "Reported_Turnout_Percent"
    )
    turnout_values = pd.to_numeric(ingestion.data[turnout_column], errors="coerce").dropna()
    turnout_bounds = (float(turnout_values.min()), float(turnout_values.max()))
    turnout_range = st.sidebar.slider(
        "Turnout range (%)",
        min_value=turnout_bounds[0],
        max_value=turnout_bounds[1],
        value=turnout_bounds,
    )
    maximum_ballots = (
        int(ingestion.data["Ballots_Cast"].max()) if "Ballots_Cast" in ingestion.data else 0
    )
    minimum_ballots = st.sidebar.number_input(
        "Minimum ballots cast", min_value=0, max_value=maximum_ballots, value=0
    )
    scope = filter_records(
        ingestion.data,
        jurisdictions=selected_jurisdictions,
        turnout_range=turnout_range,
        minimum_ballots=int(minimum_ballots),
        vote_types=vote_types if "Vote_Type" in ingestion.data else None,
    )
    current_signature = _signature(
        ingestion.provenance["sha256"],
        candidate_key,
        methods,
        selected_jurisdictions,
        turnout_range,
        int(minimum_ballots),
        vote_types,
    )
    if (
        st.session_state.analysis_signature is not None
        and st.session_state.analysis_signature != current_signature
    ):
        st.session_state.analysis_run = None
        st.session_state.analysis_signature = None
        st.info("Analysis settings changed; previous results were cleared.")

    st.sidebar.caption(f"Analysis/display/export scope: {len(scope)} validated precincts")
    if st.sidebar.button("Run selected analysis", type="primary"):
        if not methods:
            st.sidebar.error("Select at least one analysis method.")
        else:
            with st.spinner("Running selected methods..."):
                st.session_state.analysis_run = ElectionAnalysisWorkflow().run(
                    ingestion,
                    candidate_key=candidate_key,
                    methods=methods,
                    scope=scope,
                )
                st.session_state.analysis_signature = current_signature

    run = st.session_state.analysis_run
    display = scope if run is None else run.data
    explorer, results_tab, map_tab, export_tab = st.tabs(
        ["Validated data", "Method results", "Map", "Reports and exports"]
    )
    with explorer:
        st.metric("Validated precincts in scope", len(display))
        st.metric("Excluded source records", len(ingestion.excluded))
        if display.empty:
            st.info("The current filters select no records.")
        else:
            st.dataframe(display, width="stretch")

    with results_tab:
        if run is None:
            st.info("Run one or more selected methods to view results.")
        else:
            st.subheader("Method status")
            st.dataframe(_status_table(run), hide_index=True, width="stretch")
            candidate_share = next(
                item.share_column
                for item in ingestion.schema.candidates
                if item.key == candidate_key
            )
            if (
                run.statuses.get("turnout_share")
                and run.statuses["turnout_share"].state.value == "successful"
            ):
                visualizer = ComprehensiveVisualizer()
                st.plotly_chart(
                    visualizer.shpilkin.create_turnout_scatter(
                        run.data, candidate_share, "Turnout_Share_Flag"
                    ),
                    width="stretch",
                )
                st.plotly_chart(
                    visualizer.shpilkin.create_residual_plot(run.data, candidate_share),
                    width="stretch",
                )
                st.plotly_chart(
                    visualizer.shpilkin.create_turnout_histogram(
                        run.data, ingestion.schema.candidate_columns
                    ),
                    width="stretch",
                )
            if (
                run.statuses.get("vote_share_by_count")
                and run.statuses["vote_share_by_count"].state.value == "successful"
            ):
                diagnostic = run.diagnostics["vote_share_by_count"]
                st.plotly_chart(
                    ComprehensiveVisualizer().shpilkin.create_vote_share_by_count(
                        run.data,
                        candidate_share,
                        diagnostic["candidate_vote_column"],
                        diagnostic["expected_share_column"],
                    ),
                    width="stretch",
                )
            if (
                run.statuses.get("down_ballot_difference")
                and run.statuses["down_ballot_difference"].state.value == "successful"
            ):
                for comparison in run.diagnostics["down_ballot_difference"]["comparisons"]:
                    if comparison["status"] == "successful":
                        st.plotly_chart(
                            ComprehensiveVisualizer().shpilkin.create_down_ballot_difference(
                                run.data,
                                comparison["presidential_vote_column"],
                                comparison["difference_percent_column"],
                            ),
                            width="stretch",
                        )
            st.json(run.diagnostics)

    with map_tab:
        if run is None:
            st.info("Run analysis before mapping results.")
        elif not {"Latitude", "Longitude"}.issubset(run.data):
            st.info("Map unavailable: coordinates were not mapped.")
        else:
            numeric = run.data.select_dtypes(include="number").columns.tolist()
            value_options = [
                column
                for column in numeric
                if column.startswith("Candidate_Share__") or column.endswith("_Score")
            ]
            if not value_options:
                st.info("Map unavailable: no suitable numeric result is present.")
            elif run.data[["Latitude", "Longitude"]].dropna().empty:
                st.info("Map unavailable: no validated coordinates remain in scope.")
            else:
                map_value = st.selectbox("Marker color value", value_options)
                st.plotly_chart(
                    ComprehensiveVisualizer().geospatial.create_marker_map(run.data, map_value),
                    width="stretch",
                )
                st.caption("This is a precinct marker map, not a county choropleth.")

    with export_tab:
        if run is None:
            st.info("Run analysis to enable the complete reproducibility bundle.")
        else:
            st.download_button(
                "Download complete analysis bundle",
                build_export_bundle(ingestion, run),
                file_name="election_analysis_bundle.zip",
                mime="application/zip",
            )
            st.download_button(
                "Download scoped results CSV",
                run.data.to_csv(index=False),
                file_name="scoped_analysis_results.csv",
                mime="text/csv",
            )
            if st.checkbox("Enable optional AI-generated explanatory summary") and st.button(
                "Generate explanatory summary"
            ):
                narrative = AnomalyReasoningAgent().generate_executive_summary(run)
                if narrative.status == "successful":
                    st.subheader("AI-generated explanatory summary")
                    st.write(narrative.text)
                else:
                    st.info(f"Narrative {narrative.status}: {narrative.text}")
            st.json({name: asdict(status) for name, status in run.statuses.items()})


if __name__ == "__main__":
    main()
