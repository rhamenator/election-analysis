from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from src.data_ingestion import DataValidationError, ElectionDataIngester
from src.models import CandidateDefinition, ContestSchema


def csv_bytes(frame: pd.DataFrame, encoding: str = "utf-8") -> bytes:
    return frame.to_csv(index=False).encode(encoding)


def test_generalized_schema_preserves_unrecognized_columns(generalized_frame) -> None:
    generalized_frame["Unrecognized official field"] = "preserved"
    result = ElectionDataIngester().process(csv_bytes(generalized_frame))
    assert len(result.data) == len(generalized_frame)
    assert (result.data["Unrecognized official field"] == "preserved").all()
    assert result.schema.source_schema == "configured"
    assert result.provenance["sha256"]
    assert result.report.encoding == "utf-8-sig"


def test_configured_schema_wins_when_input_also_contains_legacy_columns(
    generalized_frame,
) -> None:
    frame = generalized_frame.head(3).copy()
    frame["County"] = "Legacy county"
    frame["Registered_Dem"] = 100
    frame["Registered_Rep"] = 100
    frame["Votes_Harris"] = 80
    frame["Votes_Trump"] = 100
    frame["Total_Votes"] = 180
    frame["Turnout_Percent"] = 90
    result = ElectionDataIngester().process(csv_bytes(frame))
    assert result.schema.source_schema == "configured"
    assert set(result.data["Jurisdiction"]) == set(frame["Jurisdiction"])


def test_reserved_source_columns_are_preserved_before_generation(generalized_frame) -> None:
    frame = generalized_frame.head(3).copy()
    frame["Precinct_ID"] = "official-id"
    frame["Candidate_Share__candidate_a"] = 0.123
    frame["Calculated_Turnout_Percent"] = 12.3
    result = ElectionDataIngester().process(csv_bytes(frame))
    assert (result.data["Source_Original__Precinct_ID"] == "official-id").all()
    assert np.allclose(result.data["Source_Original__Candidate_Share__candidate_a"], 0.123)
    assert np.allclose(result.data["Source_Original__Calculated_Turnout_Percent"], 12.3)
    assert "reserved_column_collision" in {issue.code for issue in result.report.warnings}


def test_legacy_adapter_is_explicit_and_preserves_source(legacy_frame) -> None:
    result = ElectionDataIngester().process(csv_bytes(legacy_frame))
    assert result.schema.source_schema == "legacy_harris_trump"
    assert "Registered_Dem" in result.data
    assert "legacy_registration_derived" in {issue.code for issue in result.report.warnings}
    assert np.allclose(
        result.data["Registered_Voters"],
        result.data["Registered_Dem"] + result.data["Registered_Rep"],
    )


def test_michigan_compatible_input_requires_no_party_registration(generalized_frame) -> None:
    result = ElectionDataIngester().process(csv_bytes(generalized_frame))
    assert not result.schema.party_registration
    assert not any("party" in issue.message.lower() for issue in result.report.errors)


def test_multiple_invalid_rows_are_excluded_with_reasons(generalized_frame) -> None:
    frame = generalized_frame.head(8).copy()
    frame["Votes_Candidate_A"] = frame["Votes_Candidate_A"].astype(float)
    frame.loc[0, "Votes_Candidate_A"] = -1
    frame.loc[1, "Votes_Candidate_A"] = 1.5
    frame.loc[2, "Votes_Candidate_A"] = frame.loc[2, "Valid_Contest_Votes"] + 1
    frame.loc[3, "Valid_Contest_Votes"] = frame.loc[3, "Ballots_Cast"] + 1
    frame.loc[4, "Ballots_Cast"] = frame.loc[4, "Registered_Voters"] + 1
    frame.loc[5, "Precinct"] = ""
    frame.loc[7, "Jurisdiction"] = frame.loc[6, "Jurisdiction"]
    frame.loc[7, "Precinct"] = frame.loc[6, "Precinct"]
    result = ElectionDataIngester().process(csv_bytes(frame))
    codes = {issue.code for issue in result.report.errors}
    assert {
        "negative_count",
        "nonintegral_count",
        "candidate_votes_exceed_contest",
        "contest_votes_exceed_ballots",
        "ballots_exceed_registration",
        "missing_identifier",
        "duplicate_precinct_id",
    }.issubset(codes)
    assert result.report.excluded_rows == 8
    assert result.excluded["Exclusion_Reasons"].str.len().gt(0).all()


def test_missing_critical_count_is_never_interpolated(generalized_frame) -> None:
    frame = generalized_frame.head(3).copy()
    frame.loc[1, "Votes_Candidate_A"] = np.nan
    result = ElectionDataIngester().process(csv_bytes(frame))
    assert len(result.data) == 2
    assert len(result.excluded) == 1
    assert "missing_critical_count" in result.excluded.iloc[0]["Exclusion_Reasons"]


def test_coordinates_are_optional_and_invalid_values_are_not_fabricated(generalized_frame) -> None:
    frame = generalized_frame.head(3).copy()
    frame.loc[0, "Latitude"] = 91
    frame.loc[1, "Longitude"] = np.nan
    result = ElectionDataIngester().process(csv_bytes(frame))
    assert pd.isna(result.data.loc[0, "Latitude"])
    assert pd.isna(result.data.loc[1, "Longitude"])
    assert (
        len([i for i in result.report.warnings if i.code == "invalid_or_missing_coordinate"]) == 2
    )


def test_turnout_mismatch_is_visible(generalized_frame) -> None:
    frame = generalized_frame.head(3).copy()
    frame.loc[0, "Reported_Turnout_Percent"] = 1
    result = ElectionDataIngester().process(csv_bytes(frame))
    assert "turnout_mismatch" in {issue.code for issue in result.report.warnings}
    assert result.data.loc[0, "Turnout_Discrepancy_Percentage_Points"] > 1


@pytest.mark.parametrize("payload", [b"", b"\n"])
def test_empty_input_is_rejected(payload) -> None:
    with pytest.raises(DataValidationError):
        ElectionDataIngester().process(payload)


def test_malformed_csv_is_rejected() -> None:
    with pytest.raises(DataValidationError):
        ElectionDataIngester().process(b'"unterminated')


def test_duplicate_headers_are_rejected_as_ambiguous() -> None:
    payload = (
        b"Jurisdiction,Precinct,Valid_Contest_Votes,Votes_Candidate_A,"
        b"Votes_Candidate_A,Votes_Candidate_B\nA,1,10,4,5,6\n"
    )
    with pytest.raises(DataValidationError) as error:
        ElectionDataIngester().process(payload)
    assert error.value.report.errors[0].code == "duplicate_columns"


def test_nonnumeric_reported_turnout_excludes_the_row(generalized_frame) -> None:
    frame = generalized_frame.head(3).copy()
    frame["Reported_Turnout_Percent"] = frame["Reported_Turnout_Percent"].astype(object)
    frame.loc[1, "Reported_Turnout_Percent"] = "not reported"
    result = ElectionDataIngester().process(csv_bytes(frame))
    assert len(result.data) == 2
    assert result.excluded.iloc[0]["Exclusion_Reasons"] == "nonnumeric_reported_turnout"


def test_unsupported_mapping_reports_missing_columns() -> None:
    with pytest.raises(DataValidationError) as error:
        ElectionDataIngester().process(b"Jurisdiction,Precinct\nA,1\n")
    assert error.value.report
    assert error.value.report.errors[0].code in {
        "unsupported_column_mapping",
        "missing_contest_total_mapping",
    }


def test_file_size_limit_is_enforced(config_writer) -> None:
    path = config_writer({"data": {"max_file_size_mb": 0.000001}})
    with pytest.raises(DataValidationError, match="maximum"):
        ElectionDataIngester(path).process(b"a,b\n1,2\n")


def test_supported_cp1252_encoding(generalized_frame) -> None:
    frame = generalized_frame.head(2).copy()
    frame.loc[0, "Jurisdiction"] = "Condado de Peñón"
    result = ElectionDataIngester().process(csv_bytes(frame, "cp1252"))
    assert result.data.loc[0, "Jurisdiction"] == "Condado de Peñón"
    assert result.report.encoding == "cp1252"


def test_binary_stream_and_process_file_compatibility(generalized_frame, tmp_path) -> None:
    payload = csv_bytes(generalized_frame.head(4))
    stream = io.BytesIO(payload)
    stream.name = "upload.csv"
    assert len(ElectionDataIngester().load_csv(stream)) == 4
    path = tmp_path / "data.csv"
    path.write_bytes(payload)
    data, summary = ElectionDataIngester().process_file(path)
    assert len(data) == 4
    assert summary["total_precincts"] == 4
    with pytest.raises(FileNotFoundError):
        ElectionDataIngester().load_csv(tmp_path / "missing.csv")


def test_explicit_unsupported_schema_is_rejected(generalized_frame) -> None:
    schema = ContestSchema("Nope", "Precinct", ())
    with pytest.raises(DataValidationError):
        ElectionDataIngester().process(csv_bytes(generalized_frame), schema=schema)


def test_third_candidate_write_ins_and_undervotes(generalized_frame) -> None:
    frame = generalized_frame.head(5).copy()
    frame["Votes_Third"] = 10
    frame["Votes_Candidate_B"] -= 10
    schema = ContestSchema(
        jurisdiction="Jurisdiction",
        precinct="Precinct",
        registered_voters="Registered_Voters",
        ballots_cast="Ballots_Cast",
        valid_contest_votes="Valid_Contest_Votes",
        write_in_votes="Write_In_Votes",
        undervotes="Undervotes",
        overvotes="Overvotes",
        latitude="Latitude",
        longitude="Longitude",
        reported_turnout="Reported_Turnout_Percent",
        candidates=(
            CandidateDefinition("Votes_Candidate_A", "A", "a"),
            CandidateDefinition("Votes_Candidate_B", "B", "b"),
            CandidateDefinition("Votes_Third", "Third", "third"),
        ),
    )
    result = ElectionDataIngester().process(csv_bytes(frame), schema=schema)
    shares = result.data[[candidate.share_column for candidate in schema.candidates]].sum(axis=1)
    assert np.allclose(shares, 1)
    assert result.data["Undervotes"].equals(frame["Undervotes"].astype("Float64"))


def test_zero_registration_and_high_valid_turnout(generalized_frame) -> None:
    frame = generalized_frame.head(2).copy()
    frame.loc[
        0,
        [
            "Registered_Voters",
            "Ballots_Cast",
            "Valid_Contest_Votes",
            "Votes_Candidate_A",
            "Votes_Candidate_B",
        ],
    ] = 0
    frame.loc[0, ["Write_In_Votes", "Undervotes", "Overvotes"]] = 0
    frame.loc[0, "Reported_Turnout_Percent"] = np.nan
    frame.loc[1, "Ballots_Cast"] = frame.loc[1, "Registered_Voters"]
    frame.loc[1, "Valid_Contest_Votes"] = frame.loc[1, "Registered_Voters"]
    frame.loc[1, "Votes_Candidate_A"] = frame.loc[1, "Registered_Voters"] // 2
    frame.loc[1, "Votes_Candidate_B"] = (
        frame.loc[1, "Registered_Voters"] - frame.loc[1, "Votes_Candidate_A"]
    )
    frame.loc[1, ["Write_In_Votes", "Undervotes", "Overvotes"]] = 0
    frame.loc[1, "Reported_Turnout_Percent"] = 100
    result = ElectionDataIngester().process(csv_bytes(frame))
    assert len(result.data) == 2
    assert pd.isna(result.data.loc[0, "Calculated_Turnout_Percent"])
    assert result.data.loc[1, "Calculated_Turnout_Percent"] == 100


def test_reordered_rows_and_columns_preserve_validated_values(generalized_frame) -> None:
    ingester = ElectionDataIngester()
    first = ingester.process(csv_bytes(generalized_frame.head(20)))
    reordered = generalized_frame.head(20).sample(frac=1, random_state=2)
    reordered = reordered[list(reversed(reordered.columns))]
    second = ingester.process(csv_bytes(reordered))
    first_values = first.data.set_index("Precinct_ID")["Candidate_Share__candidate_a"].sort_index()
    second_values = second.data.set_index("Precinct_ID")[
        "Candidate_Share__candidate_a"
    ].sort_index()
    pd.testing.assert_series_equal(first_values, second_values)


def test_large_input_within_limit(generalized_frame) -> None:
    base = generalized_frame.head(100)
    copies = []
    for index in range(20):
        copy = base.copy()
        copy["Precinct"] = copy["Precinct"] + f"-{index}"
        copies.append(copy)
    result = ElectionDataIngester().process(csv_bytes(pd.concat(copies, ignore_index=True)))
    assert len(result.data) == 2000


def test_mapped_vote_type_is_part_of_stable_precinct_key(generalized_frame, config_writer) -> None:
    row = generalized_frame.head(1)
    frame = pd.concat([row.assign(Vote_Type="Mail"), row.assign(Vote_Type="Election Day")])
    path = config_writer({"data": {"schema": {"vote_type": "Vote_Type"}}})
    result = ElectionDataIngester(path).process(csv_bytes(frame))
    assert len(result.data) == 2
    assert result.data["Precinct_ID"].is_unique
    assert set(result.data["Vote_Type"]) == {"Mail", "Election Day"}
