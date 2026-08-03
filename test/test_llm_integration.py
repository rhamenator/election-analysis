from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.llm_integration import (
    SYSTEM_INSTRUCTIONS,
    AnomalyReasoningAgent,
    OpenAIResponsesProvider,
)
from src.workflow import ElectionAnalysisWorkflow


class FakeProvider:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return "AI-generated explanatory summary: computed methods only."


def test_llm_is_disabled_by_default(ingestion) -> None:
    run = ElectionAnalysisWorkflow().run(
        ingestion, candidate_key="candidate_a", methods=["turnout_share"]
    )
    result = AnomalyReasoningAgent().generate_executive_summary(run)
    assert result.status == "skipped"


def test_injected_provider_receives_computed_statuses_and_guardrails(ingestion) -> None:
    run = ElectionAnalysisWorkflow().run(
        ingestion, candidate_key="candidate_a", methods=["turnout_share"]
    )
    provider = FakeProvider()
    result = AnomalyReasoningAgent(provider=provider).generate_executive_summary(run)
    assert result.status == "successful"
    assert "Never describe an anomaly as proof" in provider.calls[0]["instructions"]
    assert '"turnout_share"' in provider.calls[0]["input_text"]


def test_provider_failure_is_status_not_silent(ingestion) -> None:
    class Failing:
        def generate(self, **kwargs):
            raise RuntimeError("provider down")

    run = ElectionAnalysisWorkflow().run(
        ingestion, candidate_key="candidate_a", methods=["turnout_share"]
    )
    result = AnomalyReasoningAgent(provider=Failing()).generate_executive_summary(run)
    assert result.status == "failed"
    assert "provider down" in result.text


def test_openai_responses_adapter_call_shape() -> None:
    calls = []

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text=" summary ")

    client = SimpleNamespace(responses=Responses())
    provider = OpenAIResponsesProvider(model="gpt-test", client=client)
    assert (
        provider.generate(instructions="rules", input_text="facts", max_output_tokens=50)
        == "summary"
    )
    assert calls[0]["store"] is False
    assert calls[0]["reasoning"] == {"effort": "low"}
    assert calls[0]["text"] == {"verbosity": "low"}


def test_openai_adapter_rejects_empty_output_and_unknown_provider(monkeypatch) -> None:
    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(output_text=""))
    )
    provider = OpenAIResponsesProvider(model="gpt-test", client=client)
    with pytest.raises(RuntimeError, match="no output"):
        provider.generate(instructions="rules", input_text="facts", max_output_tokens=10)
    with pytest.raises(ValueError, match="Only"):
        AnomalyReasoningAgent().initialize_provider("other")
    assert "risk-limiting audit" in SYSTEM_INSTRUCTIONS


def test_openai_adapter_requires_environment_key(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: object()))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIResponsesProvider(model="gpt-test")


def test_initialize_provider_and_cli_status(monkeypatch, capsys) -> None:
    created = []

    class StubProvider:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr("src.llm_integration.OpenAIResponsesProvider", StubProvider)
    agent = AnomalyReasoningAgent()
    assert agent.initialize_provider(api_key="test")
    assert agent.enabled and created[0]["api_key"] == "test"
    from src.llm_integration import main

    main()
    assert "disabled by default" in capsys.readouterr().out
