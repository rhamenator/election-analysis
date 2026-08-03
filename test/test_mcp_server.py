from __future__ import annotations

import asyncio

import pytest
from mcp import Client

from src.mcp_server import CAUTION, create_server


async def _call(name: str, arguments: dict):
    async with Client(create_server(), raise_exceptions=True) as client:
        return await client.call_tool(name, arguments)


def test_mcp_lists_expected_contracts() -> None:
    async def scenario():
        async with Client(create_server(), raise_exceptions=True) as client:
            tools = await client.list_tools()
            resources = await client.list_resources()
            prompts = await client.list_prompts()
            return tools, resources, prompts

    tools, resources, prompts = asyncio.run(scenario())
    assert {tool.name for tool in tools.tools} == {
        "health",
        "sample_csv",
        "validate_csv",
        "analyze_csv",
        "narrative_context",
    }
    assert any(
        str(resource.uri) == "methodology://interpretation-limits"
        for resource in resources.resources
    )
    assert {prompt.name for prompt in prompts.prompts} == {"explain_run"}


def test_mcp_health_and_sample() -> None:
    health = asyncio.run(_call("health", {})).structured_content
    assert health["status"] == "healthy"
    assert "risk-limiting audit" in health["caution"]
    sample = asyncio.run(_call("sample_csv", {"rows": 30})).structured_content
    assert sample["fictional"] is True
    assert "Valid_Contest_Votes" in sample["csv"]


def test_mcp_validate_and_analyze() -> None:
    sample = asyncio.run(_call("sample_csv", {"rows": 30})).structured_content
    validated = asyncio.run(
        _call("validate_csv", {"csv_text": sample["csv"], "preview_rows": 2})
    ).structured_content
    assert validated["status"] == "valid"
    assert len(validated["accepted_preview"]) == 2
    analyzed = asyncio.run(
        _call(
            "analyze_csv",
            {
                "csv_text": sample["csv"],
                "candidate_key": "candidate_a",
                "methods": ["turnout_share"],
                "max_records": 3,
            },
        )
    ).structured_content
    assert analyzed["method_statuses"]["turnout_share"]["state"] == "successful"
    assert analyzed["records_returned"] == 3


def test_mcp_invalid_data_and_narrative_context() -> None:
    invalid = asyncio.run(
        _call("validate_csv", {"csv_text": "Jurisdiction,Precinct\nA,1\n"})
    ).structured_content
    assert invalid["status"] == "invalid"
    sample = asyncio.run(_call("sample_csv", {"rows": 30})).structured_content
    context = asyncio.run(
        _call(
            "narrative_context",
            {
                "csv_text": sample["csv"],
                "candidate_key": "candidate_a",
                "methods": ["turnout_share"],
            },
        )
    ).structured_content
    assert context["required_interpretation"] == CAUTION
    assert any("fraud" in claim for claim in context["prohibited_claims"])
    invalid_context = asyncio.run(
        _call("narrative_context", {"csv_text": "Jurisdiction,Precinct\nA,1\n"})
    ).structured_content
    assert invalid_context["status"] == "invalid"


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("sample_csv", {"rows": 0}),
        ("validate_csv", {"csv_text": "x\n1\n", "preview_rows": 101}),
        ("analyze_csv", {"csv_text": "x\n1\n", "max_records": 501}),
    ],
)
def test_mcp_rejects_unbounded_requests(tool, arguments) -> None:
    result = asyncio.run(_call(tool, arguments))
    assert result.is_error


def test_mcp_analyze_invalid_payload() -> None:
    result = asyncio.run(
        _call("analyze_csv", {"csv_text": "Jurisdiction,Precinct\nA,1\n"})
    ).structured_content
    assert result["status"] == "invalid"


def test_mcp_main_uses_configured_transport(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("src.mcp_server.mcp.run", lambda **kwargs: calls.append(kwargs))
    from src.mcp_server import main

    main()
    assert calls == [{"transport": "stdio"}]
