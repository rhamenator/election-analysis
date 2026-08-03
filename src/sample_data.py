"""Deterministic sample and simulation data for examples and scientific tests."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def generalized_sample_data(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Create internally consistent fictional precinct results without party registration."""
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = np.random.default_rng(seed)
    jurisdiction_index = np.arange(n) % max(1, min(6, n))
    registered = rng.integers(500, 3000, size=n)
    turnout_rate = np.clip(rng.normal(0.66, 0.08, size=n), 0.35, 0.92)
    ballots = np.floor(registered * turnout_rate).astype(int)
    undervotes = rng.binomial(ballots, 0.012)
    overvotes = rng.binomial(np.maximum(ballots - undervotes, 0), 0.001)
    valid = ballots - undervotes - overvotes
    latent_share = np.clip(
        0.44 + jurisdiction_index * 0.018 + rng.normal(0, 0.035, size=n), 0.15, 0.85
    )
    candidate_a = rng.binomial(valid, latent_share)
    candidate_b = valid - candidate_a
    down_ballot_a = np.maximum(0, candidate_a - rng.binomial(candidate_a, 0.025))
    down_ballot_b = np.maximum(0, candidate_b - rng.binomial(candidate_b, 0.020))
    rows = np.arange(n)
    return pd.DataFrame(
        {
            "Jurisdiction": [f"Michigan County {value + 1}" for value in jurisdiction_index],
            "Precinct": [f"Precinct {value + 1:04d}" for value in rows],
            "Registered_Voters": registered,
            "Ballots_Cast": ballots,
            "Valid_Contest_Votes": valid,
            "Votes_Candidate_A": candidate_a,
            "Votes_Candidate_B": candidate_b,
            "Votes_Down_Ballot_A": down_ballot_a,
            "Votes_Down_Ballot_B": down_ballot_b,
            "Write_In_Votes": np.zeros(n, dtype=int),
            "Undervotes": undervotes,
            "Overvotes": overvotes,
            "Latitude": 42.0 + (rows // 20) * 0.08 + rng.normal(0, 0.008, size=n),
            "Longitude": -85.0 + (rows % 20) * 0.08 + rng.normal(0, 0.008, size=n),
            "Reported_Turnout_Percent": np.round(ballots / registered * 100, 2),
            "Source_Note": "Fictional demonstration data; not official election results",
        }
    )


def legacy_sample_data(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Create a consistent legacy-format sample for the explicit adapter."""
    modern = generalized_sample_data(n, seed)
    rng = np.random.default_rng(seed + 1)
    registered = modern["Registered_Voters"].to_numpy()
    dem = rng.binomial(registered, 0.43)
    rep = registered - dem
    return pd.DataFrame(
        {
            "County": modern["Jurisdiction"],
            "Precinct": modern["Precinct"],
            "Lat": modern["Latitude"],
            "Lon": modern["Longitude"],
            "Registered_Dem": dem,
            "Registered_Rep": rep,
            "Votes_Harris": modern["Votes_Candidate_A"],
            "Votes_Trump": modern["Votes_Candidate_B"],
            "Total_Votes": modern["Valid_Contest_Votes"],
            "Turnout_Percent": np.round(modern["Valid_Contest_Votes"] / registered * 100, 2),
            "Source_Note": modern["Source_Note"],
        }
    )


def simulated_data(
    n: int = 300,
    *,
    seed: int = 42,
    injection: Literal[
        "none", "recording_error", "geographic_cluster", "distortion", "heaping", "heterogeneity"
    ] = "none",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Create controlled data and return a mask for deliberately modified records."""
    frame = generalized_sample_data(n, seed)
    mask = np.zeros(n, dtype=bool)
    # The null scenarios intentionally remove the jurisdiction effect used by the
    # realistic sample so location and candidate share are independent in expectation.
    if n and injection != "heterogeneity":
        null_rng = np.random.default_rng(seed + 7)
        valid = frame["Valid_Contest_Votes"].to_numpy()
        null_share = np.clip(null_rng.normal(0.5, 0.035, n), 0.25, 0.75)
        frame["Votes_Candidate_A"] = np.floor(valid * null_share).astype(int)
        frame["Votes_Candidate_B"] = valid - frame["Votes_Candidate_A"].to_numpy()
    if n == 0 or injection == "none":
        return frame, mask

    rng = np.random.default_rng(seed + 99)
    count = max(1, n // 20)
    if injection == "geographic_cluster":
        indices = np.arange(min(max(4, count), n))
    elif injection == "heaping":
        # A dataset-level last-digit test needs a material prevalence shift; this
        # scenario deliberately rounds one third of records rather than a few points.
        indices = rng.choice(n, size=max(1, n // 3), replace=False)
    else:
        indices = rng.choice(n, size=min(count, n), replace=False)
    mask[indices] = True

    if injection == "recording_error":
        valid = frame.loc[indices, "Valid_Contest_Votes"].to_numpy()
        frame.loc[indices, "Votes_Candidate_A"] = valid
        frame.loc[indices, "Votes_Candidate_B"] = 0
    elif injection in {"geographic_cluster", "distortion"}:
        valid = frame.loc[indices, "Valid_Contest_Votes"].to_numpy()
        boosted = np.floor(valid * 0.9).astype(int)
        frame.loc[indices, "Votes_Candidate_A"] = boosted
        frame.loc[indices, "Votes_Candidate_B"] = valid - boosted
        if injection == "distortion":
            registered = frame.loc[indices, "Registered_Voters"].to_numpy()
            ballots = np.floor(registered * 0.96).astype(int)
            frame.loc[indices, "Ballots_Cast"] = ballots
            frame.loc[indices, "Valid_Contest_Votes"] = ballots
            frame.loc[indices, "Votes_Candidate_A"] = np.floor(ballots * 0.9).astype(int)
            frame.loc[indices, "Votes_Candidate_B"] = (
                ballots - frame.loc[indices, "Votes_Candidate_A"].to_numpy()
            )
            frame.loc[indices, ["Undervotes", "Overvotes"]] = 0
            frame.loc[indices, "Reported_Turnout_Percent"] = np.round(ballots / registered * 100, 2)
    elif injection == "heaping":
        values = frame.loc[indices, "Votes_Candidate_A"].to_numpy()
        rounded = np.maximum(0, np.round(values / 10) * 10).astype(int)
        valid = frame.loc[indices, "Valid_Contest_Votes"].to_numpy()
        rounded = np.minimum(rounded, valid)
        frame.loc[indices, "Votes_Candidate_A"] = rounded
        frame.loc[indices, "Votes_Candidate_B"] = valid - rounded
    elif injection == "heterogeneity":
        group = frame["Jurisdiction"].isin(frame["Jurisdiction"].unique()[::2])
        valid = frame.loc[group, "Valid_Contest_Votes"].to_numpy()
        shares = np.clip(rng.normal(0.7, 0.025, group.sum()), 0, 1)
        frame.loc[group, "Votes_Candidate_A"] = np.floor(valid * shares).astype(int)
        frame.loc[group, "Votes_Candidate_B"] = (
            valid - frame.loc[group, "Votes_Candidate_A"].to_numpy()
        )
        mask = group.to_numpy()
    else:
        raise ValueError(f"Unknown injection: {injection}")
    return frame, mask
