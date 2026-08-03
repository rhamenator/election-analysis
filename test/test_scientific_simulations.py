from __future__ import annotations

from src.data_ingestion import ElectionDataIngester
from src.sample_data import simulated_data
from src.statistical_models import DigitAnalyzer, SpatialAnalyzer, TurnoutShareAnalyzer


def ingest_simulation(injection: str):
    frame, injected = simulated_data(300, seed=44, injection=injection)
    ingestion = ElectionDataIngester().process(frame.to_csv(index=False).encode())
    return ingestion, injected


def test_clean_null_is_not_forced_to_fixed_anomaly_rate() -> None:
    ingestion, _ = ingest_simulation("none")
    candidate = ingestion.schema.candidates[0].share_column
    result, _ = TurnoutShareAnalyzer().analyze(ingestion.data, candidate)
    rate = result["Turnout_Share_Flag"].mean()
    assert rate < 0.05
    assert rate != 0.10


def test_injected_turnout_share_distortion_is_detectable_with_low_control_flags() -> None:
    ingestion, injected = ingest_simulation("distortion")
    candidate = ingestion.schema.candidates[0].share_column
    result, _ = TurnoutShareAnalyzer().analyze(ingestion.data, candidate)
    flags = result["Turnout_Share_Flag"].to_numpy(bool)
    detection_rate = (flags & injected).sum() / injected.sum()
    false_positive_rate = (flags & ~injected).sum() / (~injected).sum()
    assert detection_rate >= 0.8
    assert false_positive_rate < 0.05


def test_geographic_cluster_is_detected_but_only_as_spatial_association() -> None:
    ingestion, injected = ingest_simulation("geographic_cluster")
    candidate = ingestion.schema.candidates[0].share_column
    result, diagnostics = SpatialAnalyzer(
        {"permutations": 999, "random_state": 42, "knn_neighbors": 8}
    ).analyze(ingestion.data, candidate)
    assert diagnostics["global_permutation_p"] <= 0.01
    local = result["Spatial_Significant"].fillna(False).to_numpy(bool)
    assert (local & injected).any()
    assert "ordinary geographic political clustering" in diagnostics["limitation"]


def test_round_number_heaping_changes_dataset_level_goodness_of_fit() -> None:
    ingestion, injected = ingest_simulation("heaping")
    assert injected.sum() == 100
    _, diagnostics = DigitAnalyzer().analyze(ingestion.data, ingestion.schema.candidate_columns)
    candidate_a = next(
        test for test in diagnostics["tests"] if test["column"] == "Votes_Candidate_A"
    )
    assert candidate_a["adjusted_p_value"] < 0.01


def test_legitimate_heterogeneity_demonstrates_spatial_limitation() -> None:
    ingestion, _ = ingest_simulation("heterogeneity")
    candidate = ingestion.schema.candidates[0].share_column
    _, spatial = SpatialAnalyzer({"permutations": 199, "random_state": 42}).analyze(
        ingestion.data, candidate
    )
    turnout, _ = TurnoutShareAnalyzer().analyze(ingestion.data, candidate)
    assert spatial["global_permutation_p"] < 0.05
    assert not turnout["Turnout_Share_Flag"].any()
