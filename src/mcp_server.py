"""Custom MCP server exposing bounded election-analysis tools."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import __version__
from .config import load_config
from .data_ingestion import DataValidationError, ElectionDataIngester
from .sample_data import generalized_sample_data
from .workflow import ALL_METHODS, ElectionAnalysisWorkflow

CAUTION = (
    "An anomaly is unusual under a stated model. It is not proof of fraud, manipulation, "
    "misconduct, or an incorrect outcome. A risk-limiting audit examines ballot evidence; "
    "aggregate precinct diagnostics do not confirm an election outcome."
)


def _records(frame: Any, limit: int) -> list[dict[str, Any]]:
    return json.loads(frame.head(limit).to_json(orient="records"))


def _validation_payload(result: Any, preview_rows: int = 10) -> dict[str, Any]:
    return {
        "status": "valid" if result.report.is_valid else "valid_with_exclusions",
        "schema": result.schema.source_schema,
        "candidate_choices": [
            {"key": item.key, "label": item.label, "share_column": item.share_column}
            for item in result.schema.candidates
        ],
        "provenance": result.provenance,
        "validation": result.report.as_dict(),
        "accepted_preview": _records(result.data, preview_rows),
        "excluded_preview": _records(result.excluded, preview_rows),
        "caution": CAUTION,
    }


def create_server(config_path: str | Path | None = "config.yaml") -> Any:
    """Create a testable MCPServer without starting a transport."""
    try:
        from mcp.server import MCPServer
    except ImportError as exc:
        raise ImportError("Install the 'mcp' dependency group to run the MCP server") from exc

    load_config(config_path)
    server = MCPServer(
        "precinct-election-analysis",
        title="Precinct Election Analysis",
        version=__version__,
        description="Validated, exploratory precinct election diagnostics",
        instructions=CAUTION,
    )

    @server.tool()
    def health() -> dict[str, Any]:
        """Return server health, version, and non-negotiable interpretation limits."""
        return {
            "status": "healthy",
            "version": __version__,
            "available_methods": list(ALL_METHODS),
            "caution": CAUTION,
        }

    @server.tool()
    def sample_csv(rows: int = 120) -> dict[str, Any]:
        """Return internally consistent fictional generalized-schema sample CSV data."""
        if not 1 <= rows <= 1000:
            raise ValueError("rows must be between 1 and 1000")
        frame = generalized_sample_data(rows)
        return {
            "filename": "fictional_michigan_compatible_sample.csv",
            "csv": frame.to_csv(index=False),
            "fictional": True,
            "caution": CAUTION,
        }

    @server.tool()
    def validate_csv(csv_text: str, preview_rows: int = 10) -> dict[str, Any]:
        """Validate CSV text, returning provenance, issues, exclusions, and a bounded preview."""
        if not 0 <= preview_rows <= 100:
            raise ValueError("preview_rows must be between 0 and 100")
        ingester = ElectionDataIngester(config_path)
        try:
            result = ingester.process(csv_text.encode("utf-8"))
        except DataValidationError as exc:
            return {
                "status": "invalid",
                "message": str(exc),
                "validation": exc.report.as_dict() if exc.report else None,
                "caution": CAUTION,
            }
        return _validation_payload(result, preview_rows)

    @server.tool()
    def analyze_csv(
        csv_text: str,
        candidate_key: str = "candidate_a",
        methods: list[str] | None = None,
        max_records: int = 100,
    ) -> dict[str, Any]:
        """Validate and analyze CSV text using only explicitly requested methods."""
        if not 0 <= max_records <= 500:
            raise ValueError("max_records must be between 0 and 500")
        selected = list(ALL_METHODS[:-1]) if methods is None else methods
        ingester = ElectionDataIngester(config_path)
        try:
            ingestion = ingester.process(csv_text.encode("utf-8"))
        except DataValidationError as exc:
            return {
                "status": "invalid",
                "message": str(exc),
                "validation": exc.report.as_dict() if exc.report else None,
                "caution": CAUTION,
            }
        workflow_path = str(config_path) if config_path is not None else None
        run = ElectionAnalysisWorkflow(workflow_path).run(
            ingestion,
            candidate_key=candidate_key,
            methods=selected,
        )
        return {
            "status": "complete_with_method_statuses",
            "method_statuses": {name: asdict(status) for name, status in run.statuses.items()},
            "metadata": run.metadata,
            "diagnostics": run.diagnostics,
            "records_returned": min(max_records, len(run.data)),
            "records_total": len(run.data),
            "records": _records(run.data, max_records),
            "caution": CAUTION,
        }

    @server.tool()
    def narrative_context(
        csv_text: str,
        candidate_key: str = "candidate_a",
        methods: list[str] | None = None,
    ) -> dict[str, Any]:
        """Prepare computed facts and constraints for an LLM explanatory summary."""
        result = analyze_csv(csv_text, candidate_key, methods, max_records=0)
        if result["status"] == "invalid":
            return result
        return {
            "label": "Context for an AI-generated explanatory summary",
            "method_statuses": result["method_statuses"],
            "metadata": result["metadata"],
            "diagnostics": result["diagnostics"],
            "required_interpretation": CAUTION,
            "prohibited_claims": [
                "proof of fraud or manipulation",
                "confirmation of an election outcome",
                "statistical significance not present in computed diagnostics",
                "audit priority without ballot evidence and an authorized audit design",
            ],
        }

    @server.resource("methodology://interpretation-limits")
    def interpretation_limits() -> str:
        """Return the interpretation boundary that applies to every tool result."""
        return CAUTION

    @server.prompt()
    def explain_run() -> str:
        """Provide a safe prompt for explaining a computed MCP analysis response."""
        return (
            "Write an AI-generated explanatory summary using only the supplied MCP tool result. "
            "List every method and its status. Describe flags only as unusual under the named "
            f"model. {CAUTION} Do not infer causes or invent missing values."
        )

    return server


mcp = create_server()


def main() -> None:
    config = load_config("config.yaml")
    transport = str(config["mcp"]["transport"])
    kwargs: dict[str, Any] = {}
    if transport != "stdio":
        kwargs = {"host": config["mcp"]["host"], "port": int(config["mcp"]["port"])}
    mcp.run(transport=transport, **kwargs)


if __name__ == "__main__":
    main()
