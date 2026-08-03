"""Optional, guarded narrative generation over computed analysis results."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

from .config import load_config
from .models import AnalysisRun

SYSTEM_INSTRUCTIONS = """You summarize already-computed precinct election diagnostics.
Use only facts in the supplied JSON. Do not invent significance, causes, evidence, or missing
values. Never describe an anomaly as proof of fraud, manipulation, misconduct, an incorrect
outcome, or a ballot-audit priority. State that flagged observations are unusual only under the
named exploratory model and that a risk-limiting audit examines ballot evidence. Explicitly list
unavailable, skipped, and failed methods. Compare agreement or disagreement across successful
methods and identify useful contextual follow-up questions, but do not infer a cause. Label the
output 'AI-generated explanatory summary'."""


class NarrativeProvider(Protocol):
    """Minimal provider contract used by the app and tests."""

    def generate(self, *, instructions: str, input_text: str, max_output_tokens: int) -> str:
        """Generate narrative text."""


class OpenAIResponsesProvider:
    """OpenAI Responses API adapter with environment-based credentials."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        if client is not None:
            self.client = client
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the 'llm' dependency group to use OpenAI summaries") from exc
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is not configured")
        self.client = OpenAI(api_key=key)

    def generate(self, *, instructions: str, input_text: str, max_output_tokens: int) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_text,
            max_output_tokens=max_output_tokens,
            reasoning={"effort": "low"},
            text={"verbosity": "low"},
            store=False,
        )
        text = str(response.output_text).strip()
        if not text:
            raise RuntimeError("OpenAI response contained no output text")
        return text


@dataclass(frozen=True)
class NarrativeResult:
    """Explicit status for optional generated prose."""

    status: str
    text: str
    provider: str | None = None
    model: str | None = None


class AnomalyReasoningAgent:
    """Compatibility facade that generates no independent statistical judgment."""

    def __init__(
        self,
        config_path: str | None = "config.yaml",
        *,
        provider: NarrativeProvider | None = None,
    ) -> None:
        self.config = load_config(config_path)["llm"]
        self.provider = provider
        self.enabled = bool(self.config["enabled"] or provider is not None)

    def initialize_provider(self, provider: str = "openai", api_key: str | None = None) -> bool:
        if provider != "openai":
            raise ValueError("Only the maintained OpenAI Responses API adapter is supported")
        self.provider = OpenAIResponsesProvider(model=str(self.config["model"]), api_key=api_key)
        self.enabled = True
        return True

    @staticmethod
    def _payload(run: AnalysisRun) -> dict[str, Any]:
        flag_columns = [
            column
            for column in run.data
            if column
            in {"Turnout_Share_Flag", "Spatial_Significant", "IF_Anomaly_Flag", "DBSCAN_Noise_Flag"}
        ]
        flag_counts = {
            column: int(run.data[column].fillna(False).astype(bool).sum())
            for column in flag_columns
        }
        return {
            "candidate": run.metadata.get("candidate_label", run.metadata["candidate"]),
            "analysis_rows": len(run.data),
            "excluded_rows": len(run.excluded),
            "method_statuses": {
                name: {
                    "state": status.state.value,
                    "message": status.message,
                    "diagnostics": status.diagnostics,
                }
                for name, status in run.statuses.items()
            },
            "flag_counts": flag_counts,
            "interpretation_warning": run.metadata["interpretation_warning"],
        }

    def generate_executive_summary(self, run: AnalysisRun) -> NarrativeResult:
        if not self.enabled:
            return NarrativeResult("skipped", "LLM narrative generation is disabled")
        if self.provider is None:
            try:
                self.initialize_provider(str(self.config["provider"]))
            except (ImportError, ValueError) as exc:
                return NarrativeResult("unavailable", str(exc))
        assert self.provider is not None
        payload = json.dumps(self._payload(run), ensure_ascii=False, default=str)
        try:
            text = self.provider.generate(
                instructions=SYSTEM_INSTRUCTIONS,
                input_text=payload,
                max_output_tokens=int(self.config["max_output_tokens"]),
            )
        except Exception as exc:
            return NarrativeResult(
                "failed", str(exc), str(self.config["provider"]), str(self.config["model"])
            )
        return NarrativeResult(
            "successful",
            text,
            str(self.config["provider"]),
            str(self.config["model"]),
        )


def main() -> None:
    agent = AnomalyReasoningAgent()
    print("LLM narratives are optional and disabled by default.")
    print(f"Configured provider: {agent.config['provider']}; model: {agent.config['model']}")


if __name__ == "__main__":
    main()
