from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import DownBallotDefinition, MethodState
from src.statistical_models import (
    AnalysisUnavailable,
    DigitAnalyzer,
    DownBallotDifferenceAnalyzer,
    SpatialAnalyzer,
    StatisticalAnomalyDetector,
    TurnoutShareAnalyzer,
    VoteShareByCountAnalyzer,
    benjamini_hochberg_adjust,
    holm_adjust,
)


def test_vote_share_by_count_is_descriptive_and_reproducible(ingestion) -> None:
    candidate = ingestion.schema.candidates[0]
    result, diagnostic = VoteShareByCountAnalyzer().analyze(
        ingestion.data, candidate.share_column, candidate.column
    )
    assert diagnostic["scope"].startswith("exploratory descriptive")
    assert diagnostic["valid_observations"] == len(ingestion.data)
    assert diagnostic["expected_share_column"] in result
    assert not any("Flag" in column for column in result.columns.difference(ingestion.data.columns))
    with pytest.raises(AnalysisUnavailable, match="no variation"):
        VoteShareByCountAnalyzer().analyze(
            ingestion.data.assign(Votes_Candidate_A=1), candidate.share_column, candidate.column
        )


def test_down_ballot_difference_uses_presidential_denominator_and_allows_negative() -> None:
    frame = pd.DataFrame(
        {
            "Precinct_ID": ["a", "b", "c"],
            "President": [100, 95, 0],
            "Senate": [95, 100, 1],
        }
    )
    pair = DownBallotDefinition("candidate_a", "Senate", "A / Senate", "a_senate")
    result, diagnostic = DownBallotDifferenceAnalyzer().analyze(frame, "President", [pair])
    assert result["Down_Ballot_Difference_Percent__a_senate"].iloc[0] == 5
    assert result["Down_Ballot_Difference_Percent__a_senate"].iloc[1] == pytest.approx(-5.2631579)
    assert pd.isna(result["Down_Ballot_Difference_Percent__a_senate"].iloc[2])
    assert diagnostic["comparisons"][0]["negative_difference_precincts"] == 1
    with pytest.raises(AnalysisUnavailable, match="configured"):
        DownBallotDifferenceAnalyzer().analyze(frame, "President", [])
    with pytest.raises(AnalysisUnavailable, match="Presidential"):
        DownBallotDifferenceAnalyzer().analyze(frame, "missing", [pair])
    with pytest.raises(AnalysisUnavailable, match="positive"):
        DownBallotDifferenceAnalyzer().analyze(frame.assign(President=0), "President", [pair])


def test_eta_analyzer_precondition_errors(ingestion) -> None:
    candidate = ingestion.schema.candidates[0]
    with pytest.raises(AnalysisUnavailable, match="missing"):
        VoteShareByCountAnalyzer().analyze(ingestion.data, candidate.share_column, "missing")
    with pytest.raises(AnalysisUnavailable, match="three"):
        VoteShareByCountAnalyzer().analyze(
            ingestion.data.head(2), candidate.share_column, candidate.column
        )
    with pytest.raises(AnalysisUnavailable, match="Spatial variable"):
        SpatialAnalyzer().analyze(ingestion.data, "missing")
    with pytest.raises(AnalysisUnavailable, match="latitude"):
        SpatialAnalyzer().analyze(ingestion.data.drop(columns=["Latitude"]), candidate.share_column)
    with pytest.raises(AnalysisUnavailable, match="two valid"):
        SpatialAnalyzer()._knn_weights(np.zeros((1, 2)))


def test_turnout_analyzer_mapping_and_configuration_guards(ingestion) -> None:
    candidate = ingestion.schema.candidates[0].share_column
    reported_only = ingestion.data.drop(columns=["Calculated_Turnout_Percent"])
    result, _ = TurnoutShareAnalyzer().analyze(reported_only, candidate)
    assert len(result) == len(reported_only)
    with pytest.raises(AnalysisUnavailable, match="requires calculated turnout"):
        TurnoutShareAnalyzer().analyze(
            reported_only.drop(columns=["Reported_Turnout_Percent"]), candidate
        )
    with pytest.raises(AnalysisUnavailable, match="configured candidate"):
        TurnoutShareAnalyzer().analyze(ingestion.data, "not_a_share")
    with pytest.raises(ValueError, match="quantile"):
        TurnoutShareAnalyzer({"baseline_turnout_quantile": 0.2}).analyze(ingestion.data, candidate)


def test_benford_successful_branch() -> None:
    values = np.tile(np.arange(100, 1000), 2)
    frame = pd.DataFrame({"Votes": values})
    _, diagnostic = DigitAnalyzer(
        {
            "benford_enabled": True,
            "minimum_observations": 30,
            "benford_minimum_observations": 100,
            "benford_minimum_orders": 0.5,
        }
    ).analyze(frame, ["Votes"])
    assert any(
        test["test"] == "benford_first_digit" and test["status"] == "successful"
        for test in diagnostic["tests"]
    )


def test_holm_adjust_known_values_and_validation() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([]).size == 0
    with pytest.raises(ValueError):
        holm_adjust([-0.1])
    assert benjamini_hochberg_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    assert benjamini_hochberg_adjust([]).size == 0
    with pytest.raises(ValueError):
        benjamini_hochberg_adjust([np.nan])


def test_turnout_share_outputs_defined_prediction_diagnostics(ingestion) -> None:
    column = ingestion.schema.candidates[0].share_column
    result, diagnostics = TurnoutShareAnalyzer().analyze(ingestion.data, column)
    assert diagnostics["candidate_share_denominator"] == "Valid_Contest_Votes"
    assert result["Turnout_Share_Studentized_Residual"].notna().all()
    assert (
        result["Turnout_Share_Prediction_Lower"] < result["Turnout_Share_Prediction_Upper"]
    ).all()


def test_turnout_share_is_row_order_invariant(ingestion) -> None:
    column = ingestion.schema.candidates[0].share_column
    analyzer = TurnoutShareAnalyzer()
    first, _ = analyzer.analyze(ingestion.data, column)
    shuffled = ingestion.data.sample(frac=1, random_state=11).reset_index(drop=True)
    second, _ = analyzer.analyze(shuffled, column)
    first_scores = first.set_index("Precinct_ID")["Turnout_Share_Studentized_Residual"].sort_index()
    second_scores = second.set_index("Precinct_ID")[
        "Turnout_Share_Studentized_Residual"
    ].sort_index()
    assert np.allclose(first_scores, second_scores)


def test_candidate_swap_has_symmetric_residuals(ingestion) -> None:
    a, b = [candidate.share_column for candidate in ingestion.schema.candidates]
    analyzer = TurnoutShareAnalyzer()
    result_a, _ = analyzer.analyze(ingestion.data, a)
    result_b, _ = analyzer.analyze(ingestion.data, b)
    assert np.allclose(
        result_a["Turnout_Share_Residual"],
        -result_b["Turnout_Share_Residual"],
        atol=1e-10,
    )


def test_turnout_share_rejects_arbitrary_percent_and_small_data(ingestion) -> None:
    analyzer = TurnoutShareAnalyzer()
    with pytest.raises(AnalysisUnavailable, match="not a configured"):
        analyzer.analyze(ingestion.data, "Reported_Turnout_Percent")
    with pytest.raises(AnalysisUnavailable, match="Need at least"):
        analyzer.analyze(ingestion.data.head(3), ingestion.schema.candidates[0].share_column)


def test_turnout_share_handles_degenerate_values(ingestion) -> None:
    column = ingestion.schema.candidates[0].share_column
    identical_turnout = ingestion.data.copy()
    identical_turnout["Calculated_Turnout_Percent"] = 50
    with pytest.raises(AnalysisUnavailable, match=r"vary|identical"):
        TurnoutShareAnalyzer().analyze(identical_turnout, column)
    identical_share = ingestion.data.copy()
    identical_share[column] = 0.5
    with pytest.raises(AnalysisUnavailable, match="variance"):
        TurnoutShareAnalyzer().analyze(identical_share, column)


def test_digit_results_are_dataset_level_not_repeated_scores(ingestion) -> None:
    output, diagnostics = DigitAnalyzer().analyze(
        ingestion.data, ingestion.schema.candidate_columns
    )
    assert diagnostics["scope"].startswith("dataset-level")
    assert not any("Entropy" in column for column in output)
    assert "Round_Number_Multiple_10__Votes_Candidate_A" in output


def test_digit_small_sample_and_heaping() -> None:
    analyzer = DigitAnalyzer({"minimum_observations": 30})
    with pytest.raises(AnalysisUnavailable):
        analyzer.analyze(pd.DataFrame({"Votes": [1, 2]}), ["Votes"])
    frame = pd.DataFrame({"Votes": np.arange(10, 1010, 10)})
    _, diagnostics = analyzer.analyze(frame, ["Votes"])
    test = diagnostics["tests"][0]
    assert test["significant"] is True


def test_benford_preconditions_are_reported() -> None:
    frame = pd.DataFrame({"Votes": np.arange(100, 220)})
    analyzer = DigitAnalyzer(
        {
            "benford_enabled": True,
            "minimum_observations": 30,
            "benford_minimum_observations": 100,
            "benford_minimum_orders": 2,
        }
    )
    _, diagnostics = analyzer.analyze(frame, ["Votes"])
    benford = [test for test in diagnostics["tests"] if test["test"] == "benford_first_digit"]
    assert benford[0]["status"] == "unavailable"


def test_spatial_knn_shrinks_neighbor_count_and_is_reproducible(ingestion) -> None:
    frame = ingestion.data.head(5)
    variable = ingestion.schema.candidates[0].share_column
    analyzer = SpatialAnalyzer({"knn_neighbors": 8, "permutations": 49, "random_state": 7})
    first, diagnostics = analyzer.analyze(frame, variable)
    second, diagnostics_second = analyzer.analyze(frame, variable)
    assert diagnostics["weights"] == "K-nearest-neighbor fallback (k=4)"
    assert diagnostics["global_permutation_p"] == diagnostics_second["global_permutation_p"]
    assert np.allclose(first["Local_Moran_Raw_P"], second["Local_Moran_Raw_P"])


def test_spatial_missing_rows_remain_aligned(ingestion) -> None:
    frame = ingestion.data.head(8).copy()
    frame.loc[2, "Latitude"] = np.nan
    variable = ingestion.schema.candidates[0].share_column
    result, diagnostics = SpatialAnalyzer({"permutations": 19}).analyze(frame, variable)
    assert len(result) == len(frame)
    assert pd.isna(
        result.loc[result["Precinct_ID"] == frame.loc[2, "Precinct_ID"], "Local_Moran_I"]
    ).all()
    assert diagnostics["excluded_for_missing_coordinates_or_value"] == 1


def test_spatial_unavailable_conditions(ingestion) -> None:
    variable = ingestion.schema.candidates[0].share_column
    with pytest.raises(AnalysisUnavailable, match="three"):
        SpatialAnalyzer({"permutations": 9}).analyze(ingestion.data.head(2), variable)
    constant = ingestion.data.head(8).copy()
    constant[variable] = 0.5
    with pytest.raises(AnalysisUnavailable, match="zero variance"):
        SpatialAnalyzer({"permutations": 9}).analyze(constant, variable)
    with pytest.raises(AnalysisUnavailable, match="Geometry"):
        SpatialAnalyzer({"weights_type": "queen"}).analyze(ingestion.data.head(8), variable)


def test_statistical_orchestrator_has_statuses_and_no_composite(ingestion, monkeypatch) -> None:
    detector = StatisticalAnomalyDetector()
    candidate = ingestion.schema.candidates[0].share_column
    run = detector.run(
        ingestion.data,
        candidate,
        candidate_vote_columns=ingestion.schema.candidate_columns,
        methods=["turnout_share", "digit_diagnostics"],
    )
    assert all(status.state == MethodState.SUCCESS for status in run.statuses.values())
    assert not any("Composite" in column for column in run.data)
    assert detector.run_full_analysis(ingestion.data, candidate).shape[0] == len(ingestion.data)
    with pytest.raises(ValueError, match="Unknown"):
        detector.run(
            ingestion.data,
            candidate,
            candidate_vote_columns=ingestion.schema.candidate_columns,
            methods=["bogus"],
        )

    def fail(*args, **kwargs):
        raise RuntimeError("visible failure")

    monkeypatch.setattr(detector.turnout_share, "analyze", fail)
    failed = detector.run(
        ingestion.data,
        candidate,
        candidate_vote_columns=ingestion.schema.candidate_columns,
        methods=["turnout_share"],
    )
    assert failed.statuses["turnout_share"].state == MethodState.FAILED
    assert "visible failure" in failed.statuses["turnout_share"].message


def test_digit_and_candidate_column_unavailability_is_explicit(ingestion) -> None:
    with pytest.raises(AnalysisUnavailable, match="No configured"):
        DigitAnalyzer().analyze(ingestion.data, ["absent"])

    detector = StatisticalAnomalyDetector()
    candidate = ingestion.schema.candidates[0].share_column
    run = detector.run(
        ingestion.data,
        candidate,
        candidate_vote_columns=ingestion.schema.candidate_columns,
        methods=["vote_share_by_count", "down_ballot_difference"],
    )
    assert all(status.state == MethodState.UNAVAILABLE for status in run.statuses.values())
    assert len(run.status_records()) == 2


def test_polygon_weights_optional_dependency_and_queen_rook_selection(monkeypatch) -> None:
    import sys
    from types import ModuleType

    analyzer = SpatialAnalyzer()
    frame = pd.DataFrame({"Geometry": ["a", "b", "c"]})
    monkeypatch.setitem(sys.modules, "libpysal", None)
    with pytest.raises(AnalysisUnavailable, match="optional spatial"):
        analyzer._polygon_weights(frame, "queen")

    calls = []

    class Weights:
        transform = None

        def full(self):
            return np.eye(3), [0, 1, 2]

    class Builder:
        @classmethod
        def from_iterable(cls, geometry):
            calls.append((cls.__name__, list(geometry)))
            return Weights()

    class Queen(Builder):
        pass

    class Rook(Builder):
        pass

    package = ModuleType("libpysal")
    weights_module = ModuleType("libpysal.weights")
    weights_module.Queen = Queen
    weights_module.Rook = Rook
    package.weights = weights_module
    monkeypatch.setitem(sys.modules, "libpysal", package)
    monkeypatch.setitem(sys.modules, "libpysal.weights", weights_module)
    queen, queen_k = analyzer._polygon_weights(frame, "queen")
    rook, rook_k = analyzer._polygon_weights(frame, "rook")
    assert np.array_equal(queen, np.eye(3))
    assert np.array_equal(rook, np.eye(3))
    assert queen_k == rook_k == -1
    assert [call[0] for call in calls] == ["Queen", "Rook"]


def test_turnout_baseline_and_rank_guards(ingestion, monkeypatch) -> None:
    candidate = ingestion.schema.candidates[0].share_column
    with pytest.raises(AnalysisUnavailable, match="baseline turnout range"):
        TurnoutShareAnalyzer(
            {"minimum_observations": 70, "baseline_turnout_quantile": 0.5}
        ).analyze(ingestion.data, candidate)

    monkeypatch.setattr(np.linalg, "matrix_rank", lambda matrix: 0)
    with pytest.raises(AnalysisUnavailable, match="rank deficient"):
        TurnoutShareAnalyzer().analyze(ingestion.data, candidate)


def test_spatial_analyze_uses_explicit_polygon_weights(ingestion, monkeypatch) -> None:
    frame = ingestion.data.head(8).copy()
    frame["Geometry"] = [f"polygon-{index}" for index in range(len(frame))]
    candidate = ingestion.schema.candidates[0].share_column
    analyzer = SpatialAnalyzer({"weights_type": "queen", "permutations": 9})
    n = len(frame)
    weights = np.ones((n, n), dtype=float) - np.eye(n)
    weights /= weights.sum(axis=1, keepdims=True)
    monkeypatch.setattr(analyzer, "_polygon_weights", lambda selected, kind: (weights, -1))
    _, diagnostic = analyzer.analyze(frame, candidate)
    assert diagnostic["weights"] == "queen polygon adjacency"
