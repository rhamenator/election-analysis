from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml_models import (
    DBSCANDetector,
    FeatureEngineer,
    IsolationForestDetector,
    MLAnomalyDetector,
)
from src.models import MethodState
from src.statistical_models import AnalysisUnavailable


def test_feature_engineering_is_numeric_stable_and_leakage_aware(ingestion) -> None:
    frame = ingestion.data.copy()
    frame["Free_Text"] = "ignore"
    frame["Turnout_Share_Flag"] = True
    frame["IF_Anomaly_Score"] = 0.9
    engineer = FeatureEngineer()
    matrix = engineer.fit_transform(frame)
    assert np.isfinite(matrix).all()
    assert matrix.shape[1] == len(engineer.feature_columns)
    assert not any("Flag" in column or "Anomaly" in column for column in engineer.feature_columns)
    assert "Free_Text" not in engineer.feature_columns
    assert engineer.feature_columns == list(engineer.feature_columns)


def test_feature_transform_handles_zero_denominators_and_unseen_jurisdiction(ingestion) -> None:
    engineer = FeatureEngineer().fit(ingestion.data)
    changed = ingestion.data.head(3).copy()
    changed["Jurisdiction"] = "New jurisdiction"
    changed["Ballots_Cast"] = 0
    changed["Valid_Contest_Votes"] = 0
    matrix = engineer.transform(changed)
    assert np.isfinite(matrix).all()


def test_feature_engineer_rejects_unfitted_constant_data(ingestion) -> None:
    engineer = FeatureEngineer()
    with pytest.raises(ValueError, match="fitted"):
        engineer.transform(ingestion.data)
    one = ingestion.data.head(1)
    with pytest.raises(AnalysisUnavailable, match="non-constant"):
        FeatureEngineer().fit(one)


def test_feature_order_mismatch_is_rejected(ingestion) -> None:
    engineer = FeatureEngineer()
    engineer.prepare_features_for_ml(ingestion.data)
    with pytest.raises(ValueError, match="order"):
        engineer.prepare_features_for_ml(ingestion.data, list(reversed(engineer.feature_columns)))


def test_isolation_forest_is_deterministic_and_scores_zero_to_one(ingestion) -> None:
    matrix = FeatureEngineer().fit_transform(ingestion.data)
    config = {
        "contamination": "auto",
        "n_estimators": 50,
        "random_state": 19,
        "max_samples": "auto",
    }
    first = IsolationForestDetector(config).fit(matrix)
    second = IsolationForestDetector(config).fit(matrix)
    first_flags, first_scores = first.predict(matrix)
    second_flags, second_scores = second.predict(matrix)
    assert np.array_equal(first_flags, second_flags)
    assert np.allclose(first_scores, second_scores)
    assert np.all((first_scores >= 0) & (first_scores <= 1))


def test_isolation_forest_errors_and_deterministic_importance(ingestion) -> None:
    matrix = FeatureEngineer().fit_transform(ingestion.data)
    detector = IsolationForestDetector({"n_estimators": 20, "random_state": 3})
    with pytest.raises(ValueError, match="fitted"):
        detector.predict(matrix)
    with pytest.raises(AnalysisUnavailable):
        detector.fit(np.empty((0, matrix.shape[1])))
    detector.fit(matrix)
    names = [f"f{i}" for i in range(matrix.shape[1])]
    first = detector.get_feature_importance(matrix, names)
    second = detector.get_feature_importance(matrix, names)
    pd.testing.assert_frame_equal(first, second)
    with pytest.raises(ValueError, match="length"):
        detector.get_feature_importance(matrix, ["wrong"])


def test_dbscan_detects_degenerate_and_usable_outputs() -> None:
    with pytest.raises(AnalysisUnavailable, match="needs"):
        DBSCANDetector({"min_samples": 5}).fit_predict(np.zeros((2, 2)))
    with pytest.raises(AnalysisUnavailable, match="no usable"):
        DBSCANDetector({"eps": 0.0001, "min_samples": 2, "metric": "euclidean"}).fit_predict(
            np.arange(20).reshape(10, 2)
        )
    matrix = np.vstack(
        [
            np.random.default_rng(1).normal(0, 0.02, size=(10, 2)),
            np.random.default_rng(2).normal(3, 0.02, size=(10, 2)),
        ]
    )
    detector = DBSCANDetector({"eps": 0.2, "min_samples": 3, "metric": "euclidean"})
    labels, noise = detector.fit_predict(matrix)
    assert detector.usable
    assert detector.cluster_stats["n_clusters"] == 2
    analysis = detector.get_cluster_analysis(matrix, labels, ["x", "y"])
    assert len(analysis) == 2
    assert not noise.any()


def test_ml_run_default_dbscan_status_and_no_composite(ingestion) -> None:
    candidate = ingestion.schema.candidates[0].share_column
    detector = MLAnomalyDetector()
    run = detector.run(ingestion.data, candidate=candidate)
    assert run.statuses["isolation_forest"].state == MethodState.SUCCESS
    assert run.statuses["dbscan"].state == MethodState.SKIPPED
    assert "IF_Anomaly_Score" in run.data
    assert "ML_Composite_Score" not in run.data
    with pytest.raises(ValueError, match="Unknown"):
        detector.run(ingestion.data, candidate=candidate, methods=["bogus"])


def test_fit_predict_and_model_round_trip(ingestion, tmp_path) -> None:
    detector = MLAnomalyDetector().fit_models(ingestion.data)
    before = detector.predict_anomalies(ingestion.data)
    detector.save_models(tmp_path)
    restored = MLAnomalyDetector()
    restored.load_models(tmp_path)
    after = restored.predict_anomalies(ingestion.data)
    assert restored.feature_names == detector.feature_names
    assert np.array_equal(before["IF_Anomaly_Flag"], after["IF_Anomaly_Flag"])
    assert np.allclose(before["IF_Anomaly_Score"], after["IF_Anomaly_Score"])


def test_model_persistence_guards(ingestion, tmp_path) -> None:
    detector = MLAnomalyDetector()
    with pytest.raises(ValueError, match="fitted"):
        detector.save_models(tmp_path)
    with pytest.raises(ValueError, match="fitted"):
        detector.predict_anomalies(ingestion.data)
    import joblib

    joblib.dump("wrong", tmp_path / "ml_detector.joblib")
    with pytest.raises(ValueError, match="not an"):
        detector.load_models(tmp_path)


def test_explanation_without_shap_is_explicit(ingestion) -> None:
    detector = MLAnomalyDetector().fit_models(ingestion.data)
    explanation = detector.explain_predictions(ingestion.data)
    assert explanation["shap_status"].startswith("skipped")
    assert not explanation["isolation_forest_importance"].empty


def test_explanation_is_explicit_when_isolation_forest_is_disabled(
    ingestion, config_writer
) -> None:
    path = config_writer({"ml": {"isolation_forest": {"enabled": False}}})
    detector = MLAnomalyDetector(path).fit_models(ingestion.data)
    explanation = detector.explain_predictions(ingestion.data)
    assert explanation["isolation_forest_status"].startswith("unavailable")
    assert explanation["shap_status"].startswith("skipped")
    assert "isolation_forest_importance" not in explanation


def test_ml_orchestrator_disabled_and_failure_paths(ingestion, config_writer, monkeypatch) -> None:
    candidate = ingestion.schema.candidates[0].share_column
    disabled_path = config_writer(
        {
            "ml": {
                "isolation_forest": {"enabled": False},
                "dbscan": {"enabled": True, "eps": 100.0, "min_samples": 2},
            }
        }
    )
    configured = MLAnomalyDetector(disabled_path)
    run = configured.run(ingestion.data, candidate=candidate)
    assert run.statuses["isolation_forest"].state == MethodState.SKIPPED
    assert run.statuses["dbscan"].state == MethodState.SUCCESS

    unavailable = MLAnomalyDetector().run(
        ingestion.data.head(1), candidate=candidate, methods=["isolation_forest", "dbscan"]
    )
    assert all(status.state == MethodState.UNAVAILABLE for status in unavailable.statuses.values())

    failing = MLAnomalyDetector()
    monkeypatch.setattr(
        failing.isolation_forest,
        "fit",
        lambda matrix: (_ for _ in ()).throw(RuntimeError("if failed")),
    )
    failed = failing.run(ingestion.data, candidate=candidate, methods=["isolation_forest"])
    assert failed.statuses["isolation_forest"].state == MethodState.FAILED


def test_ml_prediction_and_explanation_guards(ingestion, config_writer) -> None:
    with pytest.raises(ValueError, match="explanation"):
        MLAnomalyDetector().explain_predictions(ingestion.data)
    matrix = FeatureEngineer().fit_transform(ingestion.data)
    with pytest.raises(ValueError, match="scoring"):
        IsolationForestDetector().raw_scores(matrix)
    path = config_writer({"ml": {"dbscan": {"enabled": True, "eps": 100.0, "min_samples": 2}}})
    detector = MLAnomalyDetector(path).fit_models(ingestion.data)
    predicted = detector.predict_anomalies(ingestion.data.head(10))
    assert {"DBSCAN_Cluster", "DBSCAN_Noise_Flag"}.issubset(predicted)
