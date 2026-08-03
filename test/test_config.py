from __future__ import annotations

import pytest

from src.config import DEFAULT_CONFIG, deep_merge, load_config, validate_config


def test_deep_merge_preserves_nested_defaults() -> None:
    merged = deep_merge(DEFAULT_CONFIG, {"ml": {"dbscan": {"enabled": True}}})
    assert merged["ml"]["dbscan"]["enabled"] is True
    assert merged["ml"]["dbscan"]["metric"] == "euclidean"
    assert DEFAULT_CONFIG["ml"]["dbscan"]["enabled"] is False


def test_missing_config_uses_defaults(tmp_path) -> None:
    config = load_config(tmp_path / "absent.yaml")
    assert config["statistics"]["turnout_share"]["minimum_observations"] == 20
    assert config["ml"]["isolation_forest"]["max_samples"] == "auto"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("data", "max_file_size_mb"), 0),
        (("ml", "isolation_forest", "contamination"), 0.8),
        (("ml", "dbscan", "metric"), "made-up"),
        (("statistics", "spatial", "weights_type"), "hex"),
        (("mcp", "transport"), "websocket"),
        (("mcp", "host"), ""),
        (("mcp", "port"), 0),
        (("mcp", "port"), True),
        (("statistics", "turnout_share", "polynomial_degree"), 0),
        (("statistics", "turnout_share", "confidence_level"), 1),
        (("statistics", "turnout_share", "baseline_turnout_quantile"), 0.4),
        (("statistics", "digits", "alpha"), float("nan")),
        (("statistics", "spatial", "knn_neighbors"), 0),
        (("statistics", "spatial", "permutations"), 0),
        (("ml", "isolation_forest", "n_estimators"), 0),
        (("ml", "dbscan", "eps"), 0),
        (("ml", "dbscan", "min_samples"), 0),
    ],
)
def test_invalid_config_values_are_rejected(path, value) -> None:
    config = deep_merge(DEFAULT_CONFIG, {})
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_candidate_keys_must_be_unique() -> None:
    config = deep_merge(DEFAULT_CONFIG, {})
    config["data"]["schema"]["candidates"][1]["key"] = "candidate_a"
    with pytest.raises(ValueError, match="unique"):
        validate_config(config)


def test_candidate_labels_must_be_unique() -> None:
    config = deep_merge(DEFAULT_CONFIG, {})
    config["data"]["schema"]["candidates"][1]["label"] = "Candidate A"
    with pytest.raises(ValueError, match="unique"):
        validate_config(config)


def test_configuration_root_must_be_mapping(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root"):
        load_config(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config["data"]["schema"].update(candidates=[]),
        lambda config: config["data"]["schema"]["candidates"][0].update(key=""),
        lambda config: config["data"]["schema"]["down_ballot_pairs"][0].update(label=""),
        lambda config: config["data"]["schema"]["down_ballot_pairs"][0].update(
            presidential_candidate_key="missing"
        ),
        lambda config: config["data"]["schema"]["down_ballot_pairs"][1].update(
            key="candidate_a_down_ballot"
        ),
    ],
)
def test_schema_definition_guards(mutation) -> None:
    config = deep_merge(DEFAULT_CONFIG, {})
    mutation(config)
    with pytest.raises(ValueError):
        validate_config(config)
