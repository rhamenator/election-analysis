"""Shared deterministic fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_ingestion import ElectionDataIngester
from src.sample_data import generalized_sample_data, legacy_sample_data


@pytest.fixture
def generalized_frame() -> pd.DataFrame:
    return generalized_sample_data(120)


@pytest.fixture
def legacy_frame() -> pd.DataFrame:
    return legacy_sample_data(120)


@pytest.fixture
def ingestion(generalized_frame: pd.DataFrame):
    return ElectionDataIngester().process(generalized_frame.to_csv(index=False).encode())


@pytest.fixture
def config_writer(tmp_path: Path):
    def write_config(payload: dict) -> Path:
        path = tmp_path / "config.yaml"
        import yaml

        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return path

    return write_config
