"""Configuration loading, default merging, and validation."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "max_file_size_mb": 200,
        "encodings": ["utf-8-sig", "utf-8", "cp1252", "latin-1"],
        "turnout_tolerance_percentage_points": 1.0,
        "schema": {
            "jurisdiction": "Jurisdiction",
            "precinct": "Precinct",
            "registered_voters": "Registered_Voters",
            "active_registered_voters": None,
            "ballots_cast": "Ballots_Cast",
            "valid_contest_votes": "Valid_Contest_Votes",
            "write_in_votes": "Write_In_Votes",
            "undervotes": "Undervotes",
            "overvotes": "Overvotes",
            "latitude": "Latitude",
            "longitude": "Longitude",
            "reported_turnout": "Reported_Turnout_Percent",
            "vote_type": None,
            "party_registration": [],
            "down_ballot_pairs": [
                {
                    "presidential_candidate_key": "candidate_a",
                    "down_ballot_column": "Votes_Down_Ballot_A",
                    "label": "Candidate A / Down-Ballot A",
                    "key": "candidate_a_down_ballot",
                },
                {
                    "presidential_candidate_key": "candidate_b",
                    "down_ballot_column": "Votes_Down_Ballot_B",
                    "label": "Candidate B / Down-Ballot B",
                    "key": "candidate_b_down_ballot",
                },
            ],
            "contest_votes_may_exceed_ballots": False,
            "ballots_may_exceed_registration": False,
            "candidates": [
                {"column": "Votes_Candidate_A", "label": "Candidate A", "key": "candidate_a"},
                {"column": "Votes_Candidate_B", "label": "Candidate B", "key": "candidate_b"},
            ],
        },
    },
    "statistics": {
        "turnout_share": {
            "polynomial_degree": 1,
            "confidence_level": 0.95,
            "minimum_observations": 20,
            "baseline_turnout_quantile": 0.90,
            "studentized_residual_threshold": 3.0,
            "high_leverage_multiplier": 2.0,
        },
        "digits": {
            "minimum_observations": 30,
            "alpha": 0.05,
            "benford_enabled": False,
            "benford_minimum_observations": 100,
            "benford_minimum_orders": 2,
        },
        "spatial": {
            "weights_type": "knn",
            "knn_neighbors": 8,
            "permutations": 999,
            "alpha": 0.05,
            "random_state": 42,
        },
    },
    "ml": {
        "random_state": 42,
        "isolation_forest": {
            "enabled": True,
            "contamination": "auto",
            "n_estimators": 200,
            "max_samples": "auto",
        },
        "dbscan": {
            "enabled": False,
            "eps": 1.5,
            "min_samples": 5,
            "metric": "euclidean",
        },
        "shap": {"enabled": False, "sample_size": 500},
    },
    "visualization": {
        "colors": {
            "normal": "#1E88E5",
            "flagged": "#D55E00",
            "expected": "#009E73",
            "unavailable": "#777777",
        },
        "map": {"default_zoom": 6, "tile_layer": "OpenStreetMap", "marker_size": 7},
    },
    "dashboard": {"page_title": "Precinct Election Analysis", "max_file_size_mb": 200},
    "llm": {
        "enabled": False,
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "max_output_tokens": 800,
    },
    "mcp": {"transport": "stdio", "host": "127.0.0.1", "port": 8000},
}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings while replacing scalar/list leaves."""
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject unsupported or internally inconsistent settings."""
    max_size = config["data"]["max_file_size_mb"]
    if not isinstance(max_size, (int, float)) or max_size <= 0:
        raise ValueError("data.max_file_size_mb must be positive")

    candidates = config["data"]["schema"].get("candidates", [])
    if not candidates:
        raise ValueError("data.schema.candidates must contain at least one candidate")
    columns = [item.get("column") for item in candidates]
    keys = [item.get("key") for item in candidates]
    if any(not item for item in columns + keys):
        raise ValueError("every candidate requires non-empty column, label, and key values")
    if len(set(columns)) != len(columns) or len(set(keys)) != len(keys):
        raise ValueError("candidate columns and keys must be unique")

    pairs = config["data"]["schema"].get("down_ballot_pairs", [])
    candidate_keys = set(keys)
    pair_keys: list[str] = []
    for item in pairs:
        required = ("presidential_candidate_key", "down_ballot_column", "label", "key")
        if any(not item.get(field) for field in required):
            raise ValueError(
                "every down-ballot pair requires candidate key, column, label, and key"
            )
        if item["presidential_candidate_key"] not in candidate_keys:
            raise ValueError("down-ballot pairs must reference a configured presidential candidate")
        pair_keys.append(item["key"])
    if len(set(pair_keys)) != len(pair_keys):
        raise ValueError("down-ballot pair keys must be unique")

    contamination = config["ml"]["isolation_forest"]["contamination"]
    if contamination != "auto" and not (
        isinstance(contamination, (int, float)) and 0 < contamination <= 0.5
    ):
        raise ValueError("isolation_forest.contamination must be 'auto' or in (0, 0.5]")

    if config["ml"]["dbscan"]["metric"] not in {"euclidean", "manhattan", "cosine"}:
        raise ValueError("unsupported DBSCAN metric")

    spatial_type = config["statistics"]["spatial"]["weights_type"]
    if spatial_type not in {"knn", "queen", "rook"}:
        raise ValueError("spatial weights_type must be knn, queen, or rook")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML, safely merge it with defaults, and validate the result."""
    config = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        config_path = Path(path)
        if config_path.exists():
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, Mapping):
                raise ValueError("configuration root must be a mapping")
            config = deep_merge(config, loaded)
    validate_config(config)
    return config
