"""Unit tests for :mod:`flowagent.state_executor`."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from flowagent.errors import (
    ConfigurationError,
    InvalidTransitionError,
    MaxToolCallsExceeded,
    NoTransitionDeclaredError,
    ToolNotAllowedError,
)
from flowagent.state_executor import StateExecutor
from flowagent.state_store import StateStore
from flowagent.tool import tool
from flowagent.transition import TRANSITION_TOOL_NAME
from flowagent.types import LLMResponse, StateDef, StateMode, TransitionDef

from tests.conftest import StepToolCall


class Extracted(BaseModel):
    vendor: str | None = None
    amount: float | None = None


class InvoiceInput(BaseModel):
    raw_text: str


@tool(name="extract_fields", description="extract")
def extract_fields(raw_text: str) -> dict:
    return {"vendor": "Acme", "amount": 50.0}


@tool(name="extract_noisy", description="extra key")
def extract_noisy(raw_text: str) -> dict:
    return {"vendor": "Acme", "amount": 50.0, "junk": "ignore"}


@tool(name="bang", description="raises")
def bang() -> dict:
    raise RuntimeError("kaboom")


def _extract_state(**overrides) -> StateDef:
    base = dict(
        state_id="extract",
        description="Extract fields",
        mode=StateMode.AGENTIC,
        tools=[extract_fields],
        input_schema=InvoiceInput,
        output_schema=Extracted,
        transitions=[
            TransitionDef("validate", "Proceed"),
            TransitionDef("failed", "Abort"),
        ],
    )
    base.update(overrides)
    return StateDef(**base)


def test_agentic_happy_path(make_mock_adapter):
    adapter = make_mock_adapter(
        [
            [StepToolCall("extract_fields", {"raw_text": "inv"})],
            [
                StepToolCall(
                    TRANSITION_TOOL_NAME,
                    {"transition": "validate", "reasoning": "all good"},
                )
            ],
        ]
    )
    store = StateStore({"raw_text": "invoice 42"})
    executor = StateExecutor(adapter)

    result = executor.execute(_extract_state(), store)

    assert result.transition == "validate"
    assert result.reasoning == "all good"
    assert result.output_data == {"vendor": "Acme", "amount": 50.0}
    assert result.llm_calls_count == 2
    assert len(result.tool_calls_made) == 1
    assert result.tool_calls_made[0].tool_name == "extract_fields"
    assert result.tool_calls_made[0].success is True

    # Adapter saw the transition tool appended to the scope
    first_call = adapter.calls[0]
    assert "extract_fields" in first_call.tool_names
    assert TRANSITION_TOOL_NAME in first_call.tool_names
    assert first_call.available_transitions == {
        "validate": "Proceed",
        "failed": "Abort",
    }


def test_output_schema_projects_fields(make_mock_adapter):
    adapter = make_mock_adapter(
        [
            [StepToolCall("extract_noisy", {"raw_text": "inv"})],
            [
                StepToolCall(
                    TRANSITION_TOOL_NAME,
                    {"transition": "validate", "reasoning": "ok"},
                )
            ],
        ]
    )
    state = _extract_state(tools=[extract_noisy])
    store = StateStore({"raw_text": "inv"})
    result = StateExecutor(adapter).execute(state, store)
    # junk is not part of Extracted — dropped from output
    assert "junk" not in result.output_data
    assert result.output_data == {"vendor": "Acme", "amount": 50.0}


def test_no_output_schema_returns_all_tool_outputs(make_mock_adapter):
    adapter = make_mock_adapter(
        [
            [StepToolCall("extract_noisy", {"raw_text": "inv"})],
            [
                StepToolCall(
                    TRANSITION_TOOL_NAME,
                    {"transition": "validate", "reasoning": "ok"},
                )
            ],
        ]
    )
    state = _extract_state(tools=[extract_noisy], output_schema=None)
    store = StateStore({"raw_text": "inv"})
    result = StateExecutor(adapter).execute(state, store)
    assert result.output_data == {
        "vendor": "Acme",
        "amount": 50.0,
        "junk": "ignore",
    }


def test_tool_not_allowed(make_mock_adapter):
    adapter = make_mock_adapter(
        [[StepToolCall("forbidden", {})]]
    )
    executor = StateExecutor(adapter)
    with pytest.raises(ToolNotAllowedError) as excinfo:
        executor.execute(_extract_state(), StateStore({"raw_text": "x"}))
    assert "forbidden" in str(excinfo.value)


def test_invalid_transition_choice(make_mock_adapter):
    adapter = make_mock_adapter(
        [
            [
                StepToolCall(
                    TRANSITION_TOOL_NAME,
                    {"transition": "nope", "reasoning": "?"},
                )
            ]
        ]
    )
    with pytest.raises(InvalidTransitionError):
        StateExecutor(adapter).execute(
            _extract_state(), StateStore({"raw_text": "x"})
        )


def test_max_tool_calls_exceeded(make_mock_adapter):
    # Always respond with extract_fields — never transitions.
    adapter = make_mock_adapter(
        [[StepToolCall("extract_fields", {"raw_text": "r"})] for _ in range(10)]
    )
    state = _extract_state(max_tool_calls=2)
    with pytest.raises(MaxToolCallsExceeded):
        StateExecutor(adapter).execute(state, StateStore({"raw_text": "r"}))


def test_no_transition_declared_when_response_empty(make_mock_adapter):
    adapter = make_mock_adapter(
        [
            LLMResponse(
                tool_calls=[],
                text_content="I'm done but I forgot to transition",
                transition=None,
                reasoning=None,
                raw_response=None,
            )
        ]
    )
    with pytest.raises(NoTransitionDeclaredError):
        StateExecutor(adapter).execute(
            _extract_state(), StateStore({"raw_text": "x"})
        )


def test_adapter_sets_transition_directly(make_mock_adapter):
    """Adapter may parse transition from text — executor accepts it."""
    adapter = make_mock_adapter(
        [
            LLMResponse(
                tool_calls=[],
                text_content=None,
                transition="failed",
                reasoning="text path",
                raw_response=None,
            )
        ]
    )
    state = _extract_state()
    result = StateExecutor(adapter).execute(state, StateStore({"raw_text": "x"}))
    assert result.transition == "failed"
    assert result.reasoning == "text path"


def test_tool_error_is_fed_back_to_agent(make_mock_adapter):
    adapter = make_mock_adapter(
        [
            [StepToolCall("bang", {})],
            [
                StepToolCall(
                    TRANSITION_TOOL_NAME,
                    {"transition": "failed", "reasoning": "bang failed"},
                )
            ],
        ]
    )
    state = _extract_state(tools=[bang])
    result = StateExecutor(adapter).execute(state, StateStore({"raw_text": "x"}))
    assert result.transition == "failed"
    assert result.tool_calls_made[0].success is False
    assert "kaboom" in result.tool_calls_made[0].result["error"]
    # Second LLM call must have received the tool result message
    tool_msgs = [
        m for m in adapter.calls[1].messages if m.get("role") == "tool"
    ]
    assert any("kaboom" in m["content"] for m in tool_msgs)


def test_deterministic_state_single_transition():
    state = StateDef(
        state_id="tag",
        description="",
        mode=StateMode.DETERMINISTIC,
        handler=lambda data: {"tagged": True},
        transitions=[TransitionDef("done", "F")],
    )
    result = StateExecutor(None).execute(state, StateStore())
    assert result.transition == "done"
    assert result.output_data == {"tagged": True}
    assert result.llm_calls_count == 0
    assert result.reasoning == "deterministic"


def test_deterministic_state_resolver():
    state = StateDef(
        state_id="route",
        description="",
        mode=StateMode.DETERMINISTIC,
        transition_resolver=lambda data: "b" if data.get("flag") else "a",
        transitions=[TransitionDef("a", "A"), TransitionDef("b", "B")],
    )
    result = StateExecutor(None).execute(state, StateStore({"flag": True}))
    assert result.transition == "b"


def test_deterministic_invalid_resolver_return():
    state = StateDef(
        state_id="route",
        description="",
        mode=StateMode.DETERMINISTIC,
        transition_resolver=lambda data: "ghost",
        transitions=[TransitionDef("a", "A")],
    )
    with pytest.raises(InvalidTransitionError):
        StateExecutor(None).execute(state, StateStore())


def test_deterministic_ambiguous_without_resolver():
    state = StateDef(
        state_id="route",
        description="",
        mode=StateMode.DETERMINISTIC,
        transitions=[TransitionDef("a", "A"), TransitionDef("b", "B")],
    )
    with pytest.raises(ConfigurationError):
        StateExecutor(None).execute(state, StateStore())


def test_agentic_without_adapter_raises():
    with pytest.raises(ConfigurationError):
        StateExecutor(None).execute(_extract_state(), StateStore({"raw_text": "x"}))
