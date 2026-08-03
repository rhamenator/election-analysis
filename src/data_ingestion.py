"""CSV ingestion, contest-schema adaptation, validation, and derived fields."""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

from .config import load_config
from .models import (
    CandidateDefinition,
    ContestSchema,
    DownBallotDefinition,
    IngestionResult,
    Severity,
    ValidationReport,
)

logger = logging.getLogger(__name__)

LEGACY_COLUMNS = {
    "County",
    "Precinct",
    "Registered_Dem",
    "Registered_Rep",
    "Votes_Harris",
    "Votes_Trump",
    "Total_Votes",
    "Turnout_Percent",
}

CANONICAL_COLUMNS = {
    "jurisdiction": "Jurisdiction",
    "precinct": "Precinct",
    "registered_voters": "Registered_Voters",
    "active_registered_voters": "Active_Registered_Voters",
    "ballots_cast": "Ballots_Cast",
    "valid_contest_votes": "Valid_Contest_Votes",
    "write_in_votes": "Write_In_Votes",
    "undervotes": "Undervotes",
    "overvotes": "Overvotes",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "reported_turnout": "Reported_Turnout_Percent",
    "vote_type": "Vote_Type",
}


class DataValidationError(ValueError):
    """Raised when a dataset cannot be interpreted under the selected schema."""

    def __init__(self, message: str, report: ValidationReport | None = None):
        super().__init__(message)
        self.report = report


def _configured_schema(config: dict[str, Any]) -> ContestSchema:
    raw = config["data"]["schema"]
    candidates = tuple(
        CandidateDefinition(item["column"], item["label"], item["key"])
        for item in raw["candidates"]
    )
    down_ballot_pairs = tuple(
        DownBallotDefinition(
            item["presidential_candidate_key"],
            item["down_ballot_column"],
            item["label"],
            item["key"],
        )
        for item in raw.get("down_ballot_pairs", [])
    )
    return ContestSchema(
        jurisdiction=raw["jurisdiction"],
        precinct=raw["precinct"],
        candidates=candidates,
        registered_voters=raw.get("registered_voters"),
        active_registered_voters=raw.get("active_registered_voters"),
        ballots_cast=raw.get("ballots_cast"),
        valid_contest_votes=raw.get("valid_contest_votes"),
        write_in_votes=raw.get("write_in_votes"),
        undervotes=raw.get("undervotes"),
        overvotes=raw.get("overvotes"),
        latitude=raw.get("latitude"),
        longitude=raw.get("longitude"),
        reported_turnout=raw.get("reported_turnout"),
        vote_type=raw.get("vote_type"),
        party_registration=tuple(raw.get("party_registration", [])),
        down_ballot_pairs=down_ballot_pairs,
        contest_votes_may_exceed_ballots=bool(raw.get("contest_votes_may_exceed_ballots", False)),
        ballots_may_exceed_registration=bool(raw.get("ballots_may_exceed_registration", False)),
    )


def legacy_schema() -> ContestSchema:
    """Return the explicit Harris/Trump legacy adapter mapping."""
    return ContestSchema(
        jurisdiction="County",
        precinct="Precinct",
        candidates=(
            CandidateDefinition("Votes_Harris", "Kamala Harris", "harris"),
            CandidateDefinition("Votes_Trump", "Donald Trump", "trump"),
        ),
        ballots_cast="Total_Votes",
        valid_contest_votes="Total_Votes",
        latitude="Lat",
        longitude="Lon",
        reported_turnout="Turnout_Percent",
        party_registration=("Registered_Dem", "Registered_Rep"),
        source_schema="legacy_harris_trump",
    )


class ElectionDataIngester:
    """Load source data without inventing election counts or coordinates."""

    def __init__(self, config_path: str | Path | None = "config.yaml") -> None:
        self.config = load_config(config_path)

    @property
    def max_bytes(self) -> int:
        return int(float(self.config["data"]["max_file_size_mb"]) * 1024 * 1024)

    def _read_bytes(self, payload: bytes, report: ValidationReport) -> pd.DataFrame:
        if len(payload) > self.max_bytes:
            raise DataValidationError(
                f"Upload is {len(payload)} bytes; configured maximum is {self.max_bytes} bytes",
                report,
            )
        if not payload:
            report.add("empty_file", "The uploaded CSV is empty", Severity.ERROR)
            raise DataValidationError("The uploaded CSV is empty", report)

        last_error: Exception | None = None
        for encoding in self.config["data"]["encodings"]:
            try:
                text = payload.decode(encoding)
                frame = pd.read_csv(io.StringIO(text))
                report.encoding = encoding
                return frame
            except UnicodeDecodeError as exc:
                last_error = exc
            except pd.errors.EmptyDataError as exc:
                report.add("empty_dataset", "The CSV has no columns or records", Severity.ERROR)
                raise DataValidationError("The CSV has no columns or records", report) from exc
            except pd.errors.ParserError as exc:
                report.add("malformed_csv", str(exc), Severity.ERROR)
                raise DataValidationError(f"Malformed CSV: {exc}", report) from exc
        raise DataValidationError("CSV is not in a supported encoding", report) from last_error

    def load_csv(self, source: str | Path | bytes | BinaryIO) -> pd.DataFrame:
        """Compatibility loader returning unvalidated source columns."""
        report = ValidationReport()
        payload, name = self._source_bytes(source)
        report.source_name = name
        return self._read_bytes(payload, report)

    def _source_bytes(self, source: str | Path | bytes | BinaryIO) -> tuple[bytes, str]:
        if isinstance(source, bytes):
            return source, "memory.csv"
        if hasattr(source, "read"):
            value = source.read()
            if isinstance(value, str):
                value = value.encode("utf-8")
            return bytes(value), getattr(source, "name", "memory.csv")
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        if path.stat().st_size > self.max_bytes:
            raise DataValidationError(
                f"File is {path.stat().st_size} bytes; configured maximum is {self.max_bytes} bytes"
            )
        return path.read_bytes(), path.name

    def _select_schema(self, frame: pd.DataFrame, schema: ContestSchema | None) -> ContestSchema:
        if schema is not None:
            return schema
        if LEGACY_COLUMNS.issubset(frame.columns):
            return legacy_schema()
        return _configured_schema(self.config)

    def _validate_mapping(
        self, frame: pd.DataFrame, schema: ContestSchema, report: ValidationReport
    ) -> None:
        required = [schema.jurisdiction, schema.precinct, *schema.candidate_columns]
        if schema.valid_contest_votes:
            required.append(schema.valid_contest_votes)
        else:
            report.add(
                "missing_contest_total_mapping",
                "A valid contest-votes column mapping is required",
                Severity.ERROR,
            )
        missing = sorted({column for column in required if column not in frame.columns})
        if missing:
            report.add(
                "unsupported_column_mapping",
                f"Mapped columns are absent: {', '.join(missing)}",
                Severity.ERROR,
            )
            raise DataValidationError(report.errors[-1].message, report)

    def _canonicalize(
        self, source: pd.DataFrame, schema: ContestSchema, report: ValidationReport
    ) -> pd.DataFrame:
        frame = source.copy(deep=True)
        for concept, canonical in CANONICAL_COLUMNS.items():
            source_column = getattr(schema, concept)
            if source_column and source_column in frame.columns:
                frame[canonical] = frame[source_column]

        if schema.source_schema == "legacy_harris_trump":
            registration = frame[["Registered_Dem", "Registered_Rep"]].apply(
                pd.to_numeric, errors="coerce"
            )
            frame["Registered_Voters"] = registration.sum(axis=1, min_count=2)
            report.add(
                "legacy_registration_derived",
                "Legacy input has no total-registration field. Registered_Voters was explicitly "
                "derived as Registered_Dem + Registered_Rep for backward compatibility; this "
                "assumption is unsuitable for jurisdictions without party registration.",
                Severity.WARNING,
            )

        return frame

    @staticmethod
    def _normalize_identifiers(frame: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
        identifier_columns = ["Jurisdiction", "Precinct"]
        if "Vote_Type" in frame:
            identifier_columns.append("Vote_Type")
        for column in identifier_columns:
            missing = frame[column].isna() | frame[column].astype("string").str.strip().eq("")
            for index in frame.index[missing]:
                report.add(
                    "missing_identifier",
                    f"Missing required {column} identifier",
                    Severity.ERROR,
                    row=int(index),
                    column=column,
                )
            frame[column] = frame[column].astype("string").str.strip()

        frame["Precinct_ID"] = (
            frame["Jurisdiction"].fillna("") + "::" + frame["Precinct"].fillna("")
        )
        if "Vote_Type" in frame:
            frame["Precinct_ID"] = frame["Precinct_ID"] + "::" + frame["Vote_Type"].fillna("")
        duplicate = frame["Precinct_ID"].duplicated(keep=False)
        for index in frame.index[duplicate]:
            report.add(
                "duplicate_precinct_id",
                f"Duplicate precinct identifier: {frame.at[index, 'Precinct_ID']}",
                Severity.ERROR,
                row=int(index),
                precinct_id=str(frame.at[index, "Precinct_ID"]),
            )
        return frame

    @staticmethod
    def _coerce_counts(
        frame: pd.DataFrame, schema: ContestSchema, report: ValidationReport
    ) -> pd.DataFrame:
        canonical_counts = [
            "Registered_Voters",
            "Active_Registered_Voters",
            "Ballots_Cast",
            "Valid_Contest_Votes",
            "Write_In_Votes",
            "Undervotes",
            "Overvotes",
        ]
        count_columns = [
            *schema.candidate_columns,
            *schema.party_registration,
            *(pair.down_ballot_column for pair in schema.down_ballot_pairs),
        ]
        count_columns.extend(column for column in canonical_counts if column in frame.columns)
        for column in dict.fromkeys(count_columns):
            if column not in frame:
                continue
            original = frame[column]
            numeric = pd.to_numeric(original, errors="coerce")
            invalid = original.notna() & numeric.isna()
            nonintegral = numeric.notna() & ~np.isclose(numeric, np.round(numeric))
            negative = numeric.notna() & (numeric < 0)
            for mask, code, message in (
                (invalid, "nonnumeric_count", "Count is not numeric"),
                (nonintegral, "nonintegral_count", "Count is not an integer"),
                (negative, "negative_count", "Count is negative"),
            ):
                for index in frame.index[mask]:
                    report.add(
                        code,
                        f"{message}: {column}",
                        Severity.ERROR,
                        row=int(index),
                        column=column,
                    )
            frame[column] = numeric.astype("Float64")
        return frame

    def _validate_rows(
        self, frame: pd.DataFrame, schema: ContestSchema, report: ValidationReport
    ) -> set[int]:
        excluded: set[int] = {issue.row for issue in report.errors if issue.row is not None}
        critical = ["Valid_Contest_Votes", *schema.candidate_columns]
        if "Ballots_Cast" in frame:
            critical.append("Ballots_Cast")
        for column in critical:
            missing = frame[column].isna()
            for index in frame.index[missing]:
                report.add(
                    "missing_critical_count",
                    f"Missing critical election count: {column}",
                    Severity.ERROR,
                    row=int(index),
                    column=column,
                )
                excluded.add(int(index))

        candidate_sum = frame[list(schema.candidate_columns)].sum(axis=1, min_count=1)
        if schema.write_in_votes and "Write_In_Votes" in frame:
            candidate_sum = candidate_sum + frame["Write_In_Votes"].fillna(0)
        exceeds_contest = candidate_sum > frame["Valid_Contest_Votes"]
        self._record_row_errors(
            frame,
            report,
            excluded,
            exceeds_contest,
            "candidate_votes_exceed_contest",
            "Candidate and write-in votes exceed valid contest votes",
        )

        if "Ballots_Cast" in frame and not schema.contest_votes_may_exceed_ballots:
            mask = frame["Valid_Contest_Votes"] > frame["Ballots_Cast"]
            self._record_row_errors(
                frame,
                report,
                excluded,
                mask,
                "contest_votes_exceed_ballots",
                "Valid contest votes exceed ballots cast",
            )
        if (
            "Ballots_Cast" in frame
            and "Registered_Voters" in frame
            and not schema.ballots_may_exceed_registration
        ):
            mask = frame["Ballots_Cast"] > frame["Registered_Voters"]
            self._record_row_errors(
                frame,
                report,
                excluded,
                mask,
                "ballots_exceed_registration",
                "Ballots cast exceed registered voters",
            )
        return excluded

    @staticmethod
    def _record_row_errors(
        frame: pd.DataFrame,
        report: ValidationReport,
        excluded: set[int],
        mask: pd.Series,
        code: str,
        message: str,
    ) -> None:
        for index in frame.index[mask.fillna(False)]:
            report.add(
                code,
                message,
                Severity.ERROR,
                row=int(index),
                precinct_id=str(frame.at[index, "Precinct_ID"]),
            )
            excluded.add(int(index))

    def _coordinates_and_turnout(
        self, frame: pd.DataFrame, report: ValidationReport
    ) -> pd.DataFrame:
        for column, low, high in (("Latitude", -90, 90), ("Longitude", -180, 180)):
            if column not in frame:
                continue
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            invalid = frame[column].isna() | ~frame[column].between(low, high)
            for index in frame.index[invalid]:
                report.add(
                    "invalid_or_missing_coordinate",
                    f"{column} is missing or outside [{low}, {high}]; "
                    "spatial analysis will exclude it",
                    Severity.WARNING,
                    row=int(index),
                    column=column,
                )
            frame.loc[invalid, column] = np.nan

        if "Reported_Turnout_Percent" in frame:
            frame["Reported_Turnout_Percent"] = pd.to_numeric(
                frame["Reported_Turnout_Percent"], errors="coerce"
            )
            impossible = frame["Reported_Turnout_Percent"].notna() & ~frame[
                "Reported_Turnout_Percent"
            ].between(0, 100)
            for index in frame.index[impossible.fillna(False)]:
                report.add(
                    "impossible_reported_turnout",
                    "Reported turnout is outside 0-100 percent",
                    Severity.ERROR,
                    row=int(index),
                    column="Reported_Turnout_Percent",
                )
        return frame

    def _derived_features(
        self, frame: pd.DataFrame, schema: ContestSchema, report: ValidationReport
    ) -> pd.DataFrame:
        total = frame["Valid_Contest_Votes"].astype(float)
        for candidate in schema.candidates:
            frame[candidate.share_column] = np.where(
                total > 0, frame[candidate.column].astype(float) / total, np.nan
            )

        if "Ballots_Cast" in frame and "Registered_Voters" in frame:
            registered = frame["Registered_Voters"].astype(float)
            calculated = np.where(
                registered > 0, frame["Ballots_Cast"].astype(float) / registered * 100, np.nan
            )
            frame["Calculated_Turnout_Percent"] = calculated
            if "Reported_Turnout_Percent" in frame:
                difference = (frame["Reported_Turnout_Percent"] - calculated).abs()
                frame["Turnout_Discrepancy_Percentage_Points"] = difference
                tolerance = float(self.config["data"]["turnout_tolerance_percentage_points"])
                for index in frame.index[difference > tolerance]:
                    report.add(
                        "turnout_mismatch",
                        f"Reported and calculated turnout differ by more than {tolerance:g} points",
                        Severity.WARNING,
                        row=int(index),
                        precinct_id=str(frame.at[index, "Precinct_ID"]),
                    )
        return frame

    def process(
        self,
        source: str | Path | bytes | BinaryIO,
        *,
        schema: ContestSchema | None = None,
    ) -> IngestionResult:
        """Load and validate a CSV, preserving source columns and excluded records."""
        payload, source_name = self._source_bytes(source)
        report = ValidationReport(source_name=source_name)
        source_frame = self._read_bytes(payload, report)
        report.original_rows = len(source_frame)
        if source_frame.empty:
            report.add("empty_dataset", "The CSV contains no data rows", Severity.ERROR)
            raise DataValidationError("The CSV contains no data rows", report)

        selected_schema = self._select_schema(source_frame, schema)
        self._validate_mapping(source_frame, selected_schema, report)
        frame = self._canonicalize(source_frame, selected_schema, report)
        frame = self._normalize_identifiers(frame, report)
        frame = self._coerce_counts(frame, selected_schema, report)
        frame = self._coordinates_and_turnout(frame, report)
        excluded_indices = self._validate_rows(frame, selected_schema, report)
        frame = self._derived_features(frame, selected_schema, report)

        excluded = frame.loc[sorted(excluded_indices)].copy()
        if not excluded.empty:
            reasons: dict[int, list[str]] = {}
            for issue in report.errors:
                if issue.row is not None:
                    reasons.setdefault(issue.row, []).append(issue.code)
            excluded["Exclusion_Reasons"] = [
                ";".join(sorted(set(reasons.get(int(index), [])))) for index in excluded.index
            ]
        accepted = frame.drop(index=list(excluded_indices)).reset_index(drop=True)
        excluded = excluded.reset_index(drop=True)
        report.accepted_rows = len(accepted)
        report.excluded_rows = len(excluded)

        provenance = {
            "source_name": source_name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "encoding": report.encoding,
            "source_schema": selected_schema.source_schema,
            "source_columns": list(source_frame.columns),
            "candidate_columns": [
                {
                    "source_column": candidate.column,
                    "label": candidate.label,
                    "key": candidate.key,
                    "share_column": candidate.share_column,
                }
                for candidate in selected_schema.candidates
            ],
        }
        return IngestionResult(accepted, excluded, report, selected_schema, provenance)

    def process_file(self, path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Backward-compatible tuple API backed by the validated result model."""
        result = self.process(path)
        return result.data, self.get_data_summary(result)

    @staticmethod
    def get_data_summary(result: IngestionResult | pd.DataFrame) -> dict[str, Any]:
        """Return denominator-aware dataset and quality totals."""
        if isinstance(result, IngestionResult):
            frame = result.data
            report = result.report
        else:
            frame = result
            report = ValidationReport(accepted_rows=len(frame))
        return {
            "total_precincts": len(frame),
            "total_jurisdictions": frame.get("Jurisdiction", pd.Series(dtype=str)).nunique(),
            "ballots_cast": frame.get("Ballots_Cast", pd.Series(dtype=float)).sum(min_count=1),
            "valid_contest_votes": frame.get("Valid_Contest_Votes", pd.Series(dtype=float)).sum(
                min_count=1
            ),
            "data_quality": {
                "errors": len(report.errors),
                "warnings": len(report.warnings),
                "excluded_rows": report.excluded_rows,
                "missing_coordinates": int(
                    frame.reindex(columns=["Latitude", "Longitude"]).isna().any(axis=1).sum()
                ),
            },
        }


def main() -> None:
    """Run a deterministic ingestion demonstration using the bundled sample factory."""
    from .sample_data import generalized_sample_data

    payload = generalized_sample_data(30).to_csv(index=False).encode()
    result = ElectionDataIngester().process(payload)
    print(
        f"Validated {len(result.data)} precincts; excluded {len(result.excluded)}; "
        f"warnings {len(result.report.warnings)}"
    )


if __name__ == "__main__":
    main()
