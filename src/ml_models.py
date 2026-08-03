"""Deterministic, leakage-aware unsupervised machine-learning diagnostics."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from .config import load_config
from .models import AnalysisRun, MethodState, MethodStatus
from .statistical_models import AnalysisUnavailable

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Fit and transform a stable allow-listed numeric feature matrix."""

    def __init__(self) -> None:
        self.feature_columns: list[str] = []
        self.medians: pd.Series | None = None
        self.scaler: RobustScaler | None = None
        self.fitted = False

    @staticmethod
    def create_features(frame: pd.DataFrame) -> pd.DataFrame:
        """Create denominator-aware features without result labels or free text."""
        features = pd.DataFrame(index=frame.index)

        def numeric(column: str) -> pd.Series:
            return pd.to_numeric(cast(pd.Series, frame[column]), errors="coerce")

        if "Calculated_Turnout_Percent" in frame:
            features["turnout_fraction"] = numeric("Calculated_Turnout_Percent") / 100
        elif "Reported_Turnout_Percent" in frame:
            features["turnout_fraction"] = numeric("Reported_Turnout_Percent") / 100

        for raw_column in frame.columns:
            column = str(raw_column)
            if column.startswith("Candidate_Share__"):
                features[column] = numeric(column)

        for column in (
            "Registered_Voters",
            "Active_Registered_Voters",
            "Ballots_Cast",
            "Valid_Contest_Votes",
        ):
            if column in frame:
                values = numeric(column)
                features[f"log1p__{column}"] = np.log1p(values.where(values >= 0))

        if {"Valid_Contest_Votes", "Ballots_Cast"}.issubset(frame):
            ballots = numeric("Ballots_Cast")
            features["contest_vote_rate"] = numeric("Valid_Contest_Votes").div(
                ballots.where(ballots > 0)
            )
        for column in ("Undervotes", "Overvotes", "Write_In_Votes"):
            if column in frame and "Ballots_Cast" in frame:
                ballots = numeric("Ballots_Cast")
                features[f"rate__{column}"] = numeric(column).div(ballots.where(ballots > 0))

        if {"Latitude", "Longitude", "Jurisdiction"}.issubset(frame):
            latitude = numeric("Latitude")
            longitude = numeric("Longitude")
            centers = (
                pd.DataFrame(
                    {
                        "Latitude": latitude,
                        "Longitude": longitude,
                        "Jurisdiction": frame["Jurisdiction"],
                    }
                )
                .groupby("Jurisdiction")[["Latitude", "Longitude"]]
                .transform("mean")
            )
            features["distance_from_jurisdiction_centroid"] = np.sqrt(
                (latitude - centers["Latitude"]) ** 2 + (longitude - centers["Longitude"]) ** 2
            )

        if "Jurisdiction" in frame:
            grouping = frame["Jurisdiction"]
            for raw_feature in features.columns:
                column = str(raw_feature)
                if column == "distance_from_jurisdiction_centroid":
                    continue
                means = features[column].groupby(grouping).transform("mean")
                stds = features[column].groupby(grouping).transform("std")
                deviation = (features[column] - means).div(stds.where(stds > 0))
                features[f"jurisdiction_z__{column}"] = deviation

        return features.replace([np.inf, -np.inf], np.nan)

    def fit(self, frame: pd.DataFrame) -> FeatureEngineer:
        features = self.create_features(frame)
        usable = [
            column
            for column in features
            if features[column].notna().any() and features[column].nunique(dropna=True) > 1
        ]
        if not usable:
            raise AnalysisUnavailable(
                "No non-constant, validated numeric ML features are available"
            )
        self.feature_columns = [str(column) for column in usable]
        selected = features[usable]
        self.medians = selected.median()
        filled = selected.fillna(self.medians)
        self.scaler = RobustScaler().fit(filled)
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted or self.medians is None or self.scaler is None:
            raise ValueError("FeatureEngineer must be fitted before transform")
        features = self.create_features(frame)
        missing = sorted(set(self.feature_columns) - set(features.columns))
        if missing:
            raise ValueError(f"Input cannot reproduce fitted features: {missing}")
        selected = features[self.feature_columns].fillna(self.medians)
        array = self.scaler.transform(selected)
        if not np.isfinite(array).all():
            raise ValueError("Feature matrix contains non-finite values after preprocessing")
        return np.asarray(array, dtype=float)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def prepare_features_for_ml(
        self, frame: pd.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        """Compatibility wrapper; fitting occurs only on its first call."""
        if feature_columns is not None and self.fitted and feature_columns != self.feature_columns:
            raise ValueError("feature order differs from fitted feature order")
        return self.transform(frame) if self.fitted else self.fit_transform(frame)


class IsolationForestDetector:
    """Isolation Forest with reproducible normalized scores."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        defaults = load_config(None)["ml"]["isolation_forest"]
        self.config = {**defaults, **(config or {})}
        self.model: IsolationForest | None = None
        self.score_min: float | None = None
        self.score_max: float | None = None

    def fit(self, matrix: np.ndarray) -> IsolationForestDetector:
        if matrix.ndim != 2 or len(matrix) == 0:
            raise AnalysisUnavailable("Isolation Forest requires at least one observation")
        self.model = IsolationForest(
            contamination=self.config["contamination"],
            n_estimators=int(self.config["n_estimators"]),
            random_state=int(self.config.get("random_state", 42)),
            max_samples=self.config["max_samples"],
            n_jobs=1,
        ).fit(matrix)
        raw = -self.model.decision_function(matrix)
        self.score_min = float(raw.min())
        self.score_max = float(raw.max())
        return self

    def predict(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.score_min is None or self.score_max is None:
            raise ValueError("Isolation Forest must be fitted before prediction")
        raw = -self.model.decision_function(matrix)
        span = self.score_max - self.score_min
        normalized = (
            np.zeros_like(raw)
            if span <= np.finfo(float).eps
            else np.clip((raw - self.score_min) / span, 0, 1)
        )
        flags = self.model.predict(matrix) == -1
        return flags, normalized

    def raw_scores(self, matrix: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Isolation Forest must be fitted before scoring")
        return -self.model.decision_function(matrix)

    def get_feature_importance(self, matrix: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
        if len(feature_names) != matrix.shape[1]:
            raise ValueError("feature_names length does not match matrix width")
        baseline = self.raw_scores(matrix)
        rng = np.random.default_rng(int(self.config.get("random_state", 42)))
        importance: list[float] = []
        for index in range(matrix.shape[1]):
            permuted = matrix.copy()
            permuted[:, index] = permuted[rng.permutation(len(matrix)), index]
            importance.append(float(np.mean(np.abs(self.raw_scores(permuted) - baseline))))
        return pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
            ["importance", "feature"], ascending=[False, True], ignore_index=True
        )


class DBSCANDetector:
    """DBSCAN wrapper that explicitly rejects degenerate clustering."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        defaults = load_config(None)["ml"]["dbscan"]
        self.config = {**defaults, **(config or {})}
        self.model: DBSCAN | None = None
        self.cluster_stats: dict[str, Any] | None = None
        self.usable = False

    def fit_predict(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(matrix) < int(self.config["min_samples"]):
            raise AnalysisUnavailable(
                f"DBSCAN needs at least {self.config['min_samples']} observations; "
                f"found {len(matrix)}"
            )
        self.model = DBSCAN(
            eps=float(self.config["eps"]),
            min_samples=int(self.config["min_samples"]),
            metric=str(self.config["metric"]),
            n_jobs=1,
        )
        labels = self.model.fit_predict(matrix)
        noise = labels == -1
        clusters = sorted(set(labels) - {-1})
        self.usable = bool(clusters) and not noise.all()
        self.cluster_stats = {
            "n_clusters": len(clusters),
            "n_noise": int(noise.sum()),
            "noise_ratio": float(noise.mean()),
            "usable": self.usable,
        }
        if not self.usable:
            raise AnalysisUnavailable(
                "DBSCAN produced no usable clusters (all observations are noise or zero clusters)"
            )
        return labels, noise

    @staticmethod
    def get_cluster_analysis(
        matrix: np.ndarray, labels: np.ndarray, feature_names: list[str]
    ) -> dict[str, Any]:
        analysis: dict[str, Any] = {}
        for label in sorted(set(labels) - {-1}):
            values = matrix[labels == label]
            analysis[f"cluster_{label}"] = {
                "size": len(values),
                "feature_means": dict(
                    zip(feature_names, values.mean(axis=0).tolist(), strict=True)
                ),
            }
        return analysis


class MLAnomalyDetector:
    """Orchestrate independent ML methods without an uncalibrated composite."""

    def __init__(self, config_path: str | None = "config.yaml") -> None:
        self.full_config = load_config(config_path)
        config = self.full_config["ml"]
        isolation_config = {
            **config["isolation_forest"],
            "random_state": config["random_state"],
        }
        self.feature_engineer = FeatureEngineer()
        self.isolation_forest = IsolationForestDetector(isolation_config)
        self.dbscan = DBSCANDetector(config["dbscan"])
        self.feature_names: list[str] = []
        self.models_fitted = False
        self.training_rows = 0
        self.last_run: AnalysisRun | None = None

    def fit_models(self, frame: pd.DataFrame) -> MLAnomalyDetector:
        matrix = self.feature_engineer.fit_transform(frame)
        self.feature_names = list(self.feature_engineer.feature_columns)
        if self.full_config["ml"]["isolation_forest"]["enabled"]:
            self.isolation_forest.fit(matrix)
        self.models_fitted = True
        self.training_rows = len(frame)
        return self

    def run(
        self,
        frame: pd.DataFrame,
        *,
        candidate: str,
        methods: Iterable[str] = ("isolation_forest", "dbscan"),
        input_schema: str = "configured",
        excluded: pd.DataFrame | None = None,
    ) -> AnalysisRun:
        requested = list(dict.fromkeys(methods))
        unknown = set(requested) - {"isolation_forest", "dbscan"}
        if unknown:
            raise ValueError(f"Unknown ML methods: {sorted(unknown)}")
        run = AnalysisRun.new(
            frame,
            methods=requested,
            candidate=candidate,
            seed=int(self.full_config["ml"]["random_state"]),
            config=self.full_config["ml"],
            excluded=excluded,
            input_schema=input_schema,
        )
        try:
            matrix = self.feature_engineer.fit_transform(frame)
            self.feature_names = list(self.feature_engineer.feature_columns)
            self.models_fitted = True
            self.training_rows = len(frame)
        except AnalysisUnavailable as exc:
            for method in requested:
                run.statuses[method] = MethodStatus(method, MethodState.UNAVAILABLE, str(exc))
            self.last_run = run
            return run

        if "isolation_forest" in requested:
            if not self.full_config["ml"]["isolation_forest"]["enabled"]:
                run.statuses["isolation_forest"] = MethodStatus(
                    "isolation_forest", MethodState.SKIPPED, "Disabled by configuration"
                )
            else:
                try:
                    self.isolation_forest.fit(matrix)
                    flags, scores = self.isolation_forest.predict(matrix)
                    run.data["IF_Anomaly_Flag"] = flags
                    run.data["IF_Anomaly_Score"] = scores
                    diagnostic = {
                        "features": self.feature_names,
                        "contamination": self.isolation_forest.config["contamination"],
                        "flagged": int(flags.sum()),
                        "score_scale": (
                            "0-1 min-max calibration over the fitted dataset; ranking aid only"
                        ),
                        "limitation": (
                            "Unsupervised outliers are unusual in this feature space, not "
                            "fraudulent precincts."
                        ),
                    }
                    run.diagnostics["isolation_forest"] = diagnostic
                    run.statuses["isolation_forest"] = MethodStatus(
                        "isolation_forest", MethodState.SUCCESS, "Completed", diagnostic
                    )
                except (AnalysisUnavailable, ValueError) as exc:
                    run.statuses["isolation_forest"] = MethodStatus(
                        "isolation_forest", MethodState.UNAVAILABLE, str(exc)
                    )
                except Exception as exc:
                    logger.exception("Isolation Forest failed")
                    run.statuses["isolation_forest"] = MethodStatus(
                        "isolation_forest", MethodState.FAILED, str(exc)
                    )

        if "dbscan" in requested:
            if not self.full_config["ml"]["dbscan"]["enabled"]:
                run.statuses["dbscan"] = MethodStatus(
                    "dbscan", MethodState.SKIPPED, "Disabled by default; enable after calibration"
                )
            else:
                try:
                    labels, noise = self.dbscan.fit_predict(matrix)
                    run.data["DBSCAN_Cluster"] = labels
                    run.data["DBSCAN_Noise_Flag"] = noise
                    diagnostic = dict(self.dbscan.cluster_stats or {})
                    diagnostic["metric"] = self.dbscan.config["metric"]
                    run.diagnostics["dbscan"] = diagnostic
                    run.statuses["dbscan"] = MethodStatus(
                        "dbscan", MethodState.SUCCESS, "Completed", diagnostic
                    )
                except AnalysisUnavailable as exc:
                    run.statuses["dbscan"] = MethodStatus(
                        "dbscan", MethodState.UNAVAILABLE, str(exc), self.dbscan.cluster_stats or {}
                    )
                except Exception as exc:
                    logger.exception("DBSCAN failed")
                    run.statuses["dbscan"] = MethodStatus("dbscan", MethodState.FAILED, str(exc))
        self.last_run = run
        return run

    def predict_anomalies(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.models_fitted:
            raise ValueError("Models must be fitted before prediction")
        matrix = self.feature_engineer.transform(frame)
        output = frame.copy()
        if self.isolation_forest.model is not None:
            flags, scores = self.isolation_forest.predict(matrix)
            output["IF_Anomaly_Flag"] = flags
            output["IF_Anomaly_Score"] = scores
        if self.full_config["ml"]["dbscan"]["enabled"]:
            try:
                labels, noise = self.dbscan.fit_predict(matrix)
                output["DBSCAN_Cluster"] = labels
                output["DBSCAN_Noise_Flag"] = noise
            except AnalysisUnavailable:
                pass
        return output

    def explain_predictions(self, frame: pd.DataFrame) -> dict[str, Any]:
        if not self.models_fitted:
            raise ValueError("Models must be fitted before explanation")
        matrix = self.feature_engineer.transform(frame)
        explanation: dict[str, Any] = {
            "isolation_forest_importance": self.isolation_forest.get_feature_importance(
                matrix, self.feature_names
            )
        }
        if not self.full_config["ml"]["shap"]["enabled"]:
            explanation["shap_status"] = "skipped: disabled by configuration"
            return explanation
        try:
            import shap
        except ImportError:
            explanation["shap_status"] = "unavailable: optional shap dependency is not installed"
            return explanation
        sample_size = min(len(matrix), int(self.full_config["ml"]["shap"]["sample_size"]))
        rng = np.random.default_rng(int(self.full_config["ml"]["random_state"]))
        indices = np.sort(rng.choice(len(matrix), sample_size, replace=False))
        if self.isolation_forest.model is None:
            explanation["shap_status"] = "unavailable: Isolation Forest is not fitted"
            return explanation
        explainer = shap.Explainer(self.isolation_forest.model.decision_function, matrix[indices])
        explanation["shap_values"] = explainer(matrix[indices])
        explanation["shap_status"] = "successful"
        return explanation

    def save_models(self, model_dir: str | Path = "models") -> None:
        if not self.models_fitted:
            raise ValueError("Models must be fitted before saving")
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "ml_detector.joblib")

    def load_models(self, model_dir: str | Path = "models") -> None:
        loaded = joblib.load(Path(model_dir) / "ml_detector.joblib")
        if not isinstance(loaded, MLAnomalyDetector):
            raise ValueError("Model artifact is not an MLAnomalyDetector")
        self.__dict__.update(loaded.__dict__)


def main() -> None:
    from .data_ingestion import ElectionDataIngester
    from .sample_data import generalized_sample_data

    ingestion = ElectionDataIngester().process(
        generalized_sample_data().to_csv(index=False).encode()
    )
    detector = MLAnomalyDetector()
    run = detector.run(
        ingestion.data,
        candidate=ingestion.schema.candidates[0].share_column,
        input_schema=ingestion.schema.source_schema,
    )
    print({name: status.state for name, status in run.statuses.items()})


if __name__ == "__main__":
    main()
