from __future__ import annotations

import pytest

from src.statistical_models import StatisticalAnomalyDetector
from src.visualization import ComprehensiveVisualizer, GeospatialVisualizer, StatisticalPlotter


def analyzed(ingestion):
    candidate = ingestion.schema.candidates[0].share_column
    run = StatisticalAnomalyDetector().run(
        ingestion.data,
        candidate,
        candidate_vote_columns=ingestion.schema.candidate_columns,
        methods=["turnout_share"],
    )
    return run.data, candidate


def test_turnout_and_residual_plots_work_with_repository_config(ingestion) -> None:
    frame, candidate = analyzed(ingestion)
    visualizer = ComprehensiveVisualizer()
    scatter = visualizer.shpilkin.create_turnout_scatter(frame, candidate, "Turnout_Share_Flag")
    residual = visualizer.shpilkin.create_residual_plot(frame, candidate)
    assert len(scatter.data) == 2
    assert len(residual.data) == 3
    histogram = visualizer.shpilkin.create_turnout_histogram(
        frame, ingestion.schema.candidate_columns, bins=10
    )
    assert len(histogram.data) == 2


def test_eta_compatible_descriptive_plots(ingestion) -> None:
    candidate = ingestion.schema.candidates[0]
    run = StatisticalAnomalyDetector().run(
        ingestion.data,
        candidate.share_column,
        candidate_vote_columns=ingestion.schema.candidate_columns,
        candidate_vote_column=candidate.column,
        down_ballot_pairs=ingestion.schema.down_ballot_pairs,
        methods=["vote_share_by_count", "down_ballot_difference"],
    )
    plotter = ComprehensiveVisualizer().shpilkin
    count_diagnostic = run.diagnostics["vote_share_by_count"]
    count_plot = plotter.create_vote_share_by_count(
        run.data,
        candidate.share_column,
        candidate.column,
        count_diagnostic["expected_share_column"],
    )
    comparison = run.diagnostics["down_ballot_difference"]["comparisons"][0]
    down_plot = plotter.create_down_ballot_difference(
        run.data, candidate.column, comparison["difference_percent_column"]
    )
    assert len(count_plot.data) == 2
    assert len(down_plot.data) == 1


def test_visualization_rejects_wrong_fields(ingestion) -> None:
    visualizer = ComprehensiveVisualizer()
    with pytest.raises(ValueError, match="configured candidate"):
        visualizer.shpilkin.create_turnout_scatter(ingestion.data, "Reported_Turnout_Percent")
    candidate = ingestion.schema.candidates[0].share_column
    with pytest.raises(ValueError, match="Residual diagnostics"):
        visualizer.shpilkin.create_residual_plot(ingestion.data, candidate)


def test_marker_map_is_not_mislabeled_choropleth(ingestion) -> None:
    candidate = ingestion.schema.candidates[0].share_column
    geo = GeospatialVisualizer()
    figure = geo.create_marker_map(ingestion.data, candidate)
    assert "not a choropleth" in figure.layout.title.text
    with pytest.raises(ValueError, match="polygon"):
        geo.create_choropleth_by_county(ingestion.data, candidate)
    folium_map = geo.create_folium_map(ingestion.data.head(2), candidate)
    assert folium_map is not None


def test_geospatial_visuals_handle_missing_data(ingestion) -> None:
    candidate = ingestion.schema.candidates[0].share_column
    assert len(GeospatialVisualizer().create_heatmap(ingestion.data, candidate).data) == 1
    frame = ingestion.data.copy()
    frame[["Latitude", "Longitude"]] = None
    with pytest.raises(ValueError, match="No records"):
        GeospatialVisualizer().create_marker_map(frame, candidate)
    with pytest.raises(ValueError, match="missing"):
        GeospatialVisualizer().create_heatmap(frame.drop(columns=["Latitude"]), candidate)


def test_eta_plot_validation_errors(ingestion) -> None:
    plotter = ComprehensiveVisualizer().shpilkin
    candidate = ingestion.schema.candidates[0]
    with pytest.raises(ValueError, match="vote column"):
        plotter.create_turnout_histogram(ingestion.data, [])
    with pytest.raises(ValueError, match="missing"):
        plotter.create_vote_share_by_count(ingestion.data, candidate.share_column, "missing")
    with pytest.raises(ValueError, match="unavailable"):
        plotter.create_down_ballot_difference(ingestion.data, candidate.column, "missing")


def test_statistical_plotter_validates_inputs(ingestion) -> None:
    plotter = StatisticalPlotter()
    candidate_columns = [item.share_column for item in ingestion.schema.candidates]
    assert len(plotter.create_distribution_plots(ingestion.data, candidate_columns).data) == 2
    assert len(plotter.create_correlation_heatmap(ingestion.data, candidate_columns).data) == 1
    frame = ingestion.data.copy()
    frame["Flag"] = [False] * len(frame)
    assert len(plotter.create_anomaly_summary_plot(frame, ["Flag"]).data) == 1
    with pytest.raises(ValueError):
        plotter.create_distribution_plots(frame, ["missing"])
    with pytest.raises(ValueError):
        plotter.create_correlation_heatmap(frame, candidate_columns[:1])
    with pytest.raises(ValueError):
        plotter.create_anomaly_summary_plot(frame, ["missing"])


def test_comprehensive_dashboard_writes_expected_plots(ingestion, tmp_path) -> None:
    frame, candidate = analyzed(ingestion)
    plots = ComprehensiveVisualizer().create_analysis_dashboard(frame, tmp_path, candidate)
    assert {
        "turnout_share",
        "turnout_share_residuals",
        "distributions",
        "precinct_marker_map",
    } == set(plots)
    assert (tmp_path / "precinct_marker_map.html").exists()
