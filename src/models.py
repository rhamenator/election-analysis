"""Stable data contracts shared across ingestion, analysis, UI, and MCP."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import pandas as pd


class Severity(StrEnum):
    """Validation issue severity."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class MethodState(StrEnum):
    """Observable outcome of an analysis method."""

    SUCCESS = "successful"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class ValidationIssue:
    """A machine-readable validation finding."""

    code: str
    message: str
    severity: Severity
    row: int | None = None
    column: str | None = None
    precinct_id: str | None = None


@dataclass
class ValidationReport:
    """Validation findings and source decoding information."""

    issues: list[ValidationIssue] = field(default_factory=list)
    encoding: str | None = None
    source_name: str | None = None
    original_rows: int = 0
    accepted_rows: int = 0
    excluded_rows: int = 0

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == Severity.WARNING]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add(
        self,
        code: str,
        message: str,
        severity: Severity,
        *,
        row: int | None = None,
        column: str | None = None,
        precinct_id: str | None = None,
    ) -> None:
        self.issues.append(ValidationIssue(code, message, severity, row, column, precinct_id))

    def as_dict(self) -> dict[str, Any]:
        return {
            "encoding": self.encoding,
            "source_name": self.source_name,
            "original_rows": self.original_rows,
            "accepted_rows": self.accepted_rows,
            "excluded_rows": self.excluded_rows,
            "is_valid": self.is_valid,
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass(frozen=True)
class CandidateDefinition:
    """Source column and human-facing label for a contest choice."""

    column: str
    label: str
    key: str

    @property
    def share_column(self) -> str:
        return f"Candidate_Share__{self.key}"


@dataclass(frozen=True)
class DownBallotDefinition:
    """Presidential/down-ballot vote columns that should be compared."""

    presidential_candidate_key: str
    down_ballot_column: str
    label: str
    key: str


@dataclass(frozen=True)
class ContestSchema:
    """Configuration-driven mapping from source columns to election concepts."""

    jurisdiction: str
    precinct: str
    candidates: tuple[CandidateDefinition, ...]
    registered_voters: str | None = None
    active_registered_voters: str | None = None
    ballots_cast: str | None = None
    valid_contest_votes: str | None = None
    write_in_votes: str | None = None
    undervotes: str | None = None
    overvotes: str | None = None
    latitude: str | None = None
    longitude: str | None = None
    reported_turnout: str | None = None
    vote_type: str | None = None
    party_registration: tuple[str, ...] = ()
    down_ballot_pairs: tuple[DownBallotDefinition, ...] = ()
    contest_votes_may_exceed_ballots: bool = False
    ballots_may_exceed_registration: bool = False
    source_schema: str = "configured"

    @property
    def candidate_columns(self) -> tuple[str, ...]:
        return tuple(candidate.column for candidate in self.candidates)


@dataclass
class IngestionResult:
    """Validated records, rejected records, schema, and provenance."""

    data: pd.DataFrame
    excluded: pd.DataFrame
    report: ValidationReport
    schema: ContestSchema
    provenance: dict[str, Any]


@dataclass
class MethodStatus:
    """Status and diagnostics for one requested method."""

    method: str
    state: MethodState
    message: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisRun:
    """Stable analysis result independent of DataFrame attrs."""

    data: pd.DataFrame
    statuses: dict[str, MethodStatus]
    metadata: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    excluded: pd.DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def new(
        cls,
        data: pd.DataFrame,
        *,
        methods: list[str],
        candidate: str,
        seed: int,
        config: dict[str, Any],
        excluded: pd.DataFrame | None = None,
        input_schema: str = "configured",
    ) -> AnalysisRun:
        return cls(
            data=data.copy(),
            statuses={
                method: MethodStatus(method, MethodState.SKIPPED, "Not run") for method in methods
            },
            metadata={
                "created_at": datetime.now(UTC).isoformat(),
                "candidate": candidate,
                "requested_methods": list(methods),
                "random_seed": seed,
                "input_schema": input_schema,
                "configuration": config,
                "interpretation_warning": (
                    "An anomaly is unusual under a stated model. It is not evidence of "
                    "fraud, misconduct, or an incorrect election outcome. Aggregate-data "
                    "analysis is not a risk-limiting audit of ballot evidence."
                ),
            },
            excluded=pd.DataFrame() if excluded is None else excluded.copy(),
        )

    def status_records(self) -> list[dict[str, Any]]:
        return [asdict(status) for status in self.statuses.values()]
