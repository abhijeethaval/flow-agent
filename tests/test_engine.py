"""End-to-end workflow tests via :class:`WorkflowEngine` + MockLLMAdapter."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from flowagent.engine import WorkflowEngine
from flowagent.errors import WorkflowValidationError
from flowagent.tool import tool
from flowagent.transition import TRANSITION_TOOL_NAME
from flowagent.types import StateDef, StateMode, TransitionDef, WorkflowDefinition

from tests.conftest import StepToolCall


class InvoiceInput(BaseModel):
    raw_text: str


class Extracted(BaseModel):
    vendor: str | None = None
    amount: float | None = None


class Validation(BaseModel):
    valid: bool = False
    errors: list[str] = []


@tool(name="extract_fields", description="extract")
def extract_fields(raw_text: str) -> dict:
    return {"vendor": "Acme", "amount": 42.0}


@tool(name="validate_schema", description="validate")
def validate_schema(vendor: str | None = None, amount: float | None = None) -> dict:
    errs = []
    if amount is None or amount <= 0:
        errs.append("amount must be positive")
    return {"valid": not errs, "errors": errs}


def _invoice_workflow() -> WorkflowDefinition:
    return WorkflowDefinition(
        name="invoice",
        version="1.0",
        description="test",
        initial_state="extract",
        terminal_states=["done", "failed"],
        global_system_prompt="You process invoices.",
        states={
            "extract": StateDef(
                state_id="extract",
                description="Extract fields",
                mode=StateMode.AGENTIC,
                tools=[extract_fields],
                input_schema=InvoiceInput,
                output_schema=Extracted,
                transitions=[
                    TransitionDef("validate", "Ready to validate"),
                    TransitionDef("failed", "Unreadable"),
                ],
            ),
            "validate": StateDef(
                state_id="validate",
                description="Validate",
                mode=StateMode.AGENTIC,
                tools=[validate_schema],
                input_schema=Extracted,
                output_schema=Validation,
                transitions=[
                    TransitionDef("done", "Validation passed"),
                    TransitionDef("failed", "Validation failed"),
                ],
            ),
            "done": StateDef(
                state_id="done",
                description="",
                mode=StateMode.DETERMINISTIC,
                transitions=[],
            ),
            "failed": StateDef(
                state_id="failed",
                description="",
                mode=StateMode.DETERMINISTIC,
                transitions=[],
            ),
        },
    )


def test_end_to_end_happy_path(make_mock_adapter):
    adapter = make_mock_adapter(
        [
            # extract state
            [StepToolCall("extract_fields", {"raw_text": "inv"})],
            [StepToolCall(TRANSITION_TOOL_NAME, {"transition": "validate", "reasoning": "ok"})],
            # validate state
            [StepToolCall("validate_schema", {"vendor": "Acme", "amount": 42.0})],
            [StepToolCall(TRANSITION_TOOL_NAME, {"transition": "done", "reasoning": "valid"})],
        ]
    )
    engine = WorkflowEngine(adapter)
    result = engine.run(_invoice_workflow(), {"raw_text": "Invoice #1"})

    assert result.status == "completed"
    assert result.final_state == "done"
    assert result.total_llm_calls == 4
    assert result.total_tool_calls == 2
    assert result.state_store["vendor"] == "Acme"
    assert result.state_store["valid"] is True
    assert [r.state_id for r in result.execution_trace] == ["extract", "validate"]


def test_end_to_end_branches_to_failed(make_mock_adapter):
    adapter = make_mock_adapter(
        [
            [StepToolCall(TRANSITION_TOOL_NAME, {"transition": "failed", "reasoning": "bad"})],
        ]
    )
    engine = WorkflowEngine(adapter)
    result = engine.run(_invoice_workflow(), {"raw_text": "rubbish"})
    assert result.status == "completed"
    assert result.final_state == "failed"
    assert result.total_tool_calls == 0


def test_invalid_workflow_raises():
    bad = WorkflowDefinition(
        name="bad",
        version="0",
        description="",
        initial_state="missing",
        terminal_states=[],
        states={},
    )
    engine = WorkflowEngine(llm_adapter=None)
    with pytest.raises(WorkflowValidationError):
        engine.run(bad, {})


def test_max_total_steps_bound(make_mock_adapter):
    # Two states that loop forever.
    @tool(name="nop", description="")
    def nop() -> dict:
        return {}

    wf = WorkflowDefinition(
        name="loop",
        version="0",
        description="",
        initial_state="a",
        terminal_states=["done"],
        max_total_steps=3,
        states={
            "a": StateDef(
                state_id="a",
                description="",
                mode=StateMode.AGENTIC,
                tools=[nop],
                transitions=[
                    TransitionDef("b", "to b"),
                    TransitionDef("done", "end"),
                ],
            ),
            "b": StateDef(
                state_id="b",
                description="",
                mode=StateMode.AGENTIC,
                tools=[nop],
                transitions=[TransitionDef("a", "back to a")],
            ),
            "done": StateDef(
                state_id="done",
                description="",
                mode=StateMode.DETERMINISTIC,
                transitions=[],
            ),
        },
    )

    script = []
    # Enough bouncing to exceed max_total_steps=3.
    for target in ("b", "a", "b", "a", "b"):
        script.append(
            [StepToolCall(TRANSITION_TOOL_NAME, {"transition": target, "reasoning": "loop"})]
        )
    adapter = make_mock_adapter(script)
    engine = WorkflowEngine(adapter)
    result = engine.run(wf, {})
    assert result.status == "max_steps_exceeded"
    assert len(result.execution_trace) == 3


def test_deterministic_intermediate_state(make_mock_adapter):
    @tool(name="work", description="")
    def work() -> dict:
        return {"k": 1}

    wf = WorkflowDefinition(
        name="det",
        version="0",
        description="",
        initial_state="agentic",
        terminal_states=["done"],
        states={
            "agentic": StateDef(
                state_id="agentic",
                description="",
                mode=StateMode.AGENTIC,
                tools=[work],
                transitions=[TransitionDef("route", "route")],
            ),
            "route": StateDef(
                state_id="route",
                description="",
                mode=StateMode.DETERMINISTIC,
                handler=lambda data: {"routed": True},
                transitions=[TransitionDef("done", "onward")],
            ),
            "done": StateDef(
                state_id="done",
                description="",
                mode=StateMode.DETERMINISTIC,
                transitions=[],
            ),
        },
    )
    adapter = make_mock_adapter(
        [
            [StepToolCall("work", {})],
            [StepToolCall(TRANSITION_TOOL_NAME, {"transition": "route", "reasoning": "ok"})],
        ]
    )
    engine = WorkflowEngine(adapter)
    result = engine.run(wf, {})
    assert result.status == "completed"
    assert result.final_state == "done"
    assert result.state_store["routed"] is True
    assert result.state_store["k"] == 1


def test_state_enter_exit_hooks_fire(make_mock_adapter):
    events = []

    def on_enter(store):
        events.append(("enter", store.data.get("raw_text")))

    def on_exit(store, transition):
        events.append(("exit", transition))

    wf = _invoice_workflow()
    wf.states["extract"].on_enter = on_enter
    wf.states["extract"].on_exit = on_exit

    adapter = make_mock_adapter(
        [
            [StepToolCall(TRANSITION_TOOL_NAME, {"transition": "failed", "reasoning": "x"})],
        ]
    )
    WorkflowEngine(adapter).run(wf, {"raw_text": "inv"})
    assert events == [("enter", "inv"), ("exit", "failed")]


def test_engine_callbacks_receive_events(make_mock_adapter):
    enters = []
    exits = []

    adapter = make_mock_adapter(
        [
            [StepToolCall(TRANSITION_TOOL_NAME, {"transition": "failed", "reasoning": "x"})],
        ]
    )
    engine = WorkflowEngine(adapter)
    engine.run(
        _invoice_workflow(),
        {"raw_text": "inv"},
        on_state_enter=lambda run_id, state, store: enters.append(state),
        on_state_exit=lambda run_id, state, result, store: exits.append((state, result.transition)),
    )
    assert enters == ["extract"]
    assert exits == [("extract", "failed")]


def test_failure_surfaces_in_result(make_mock_adapter):
    adapter = make_mock_adapter(
        [[StepToolCall("ghost", {})]]
    )
    engine = WorkflowEngine(adapter)
    result = engine.run(_invoice_workflow(), {"raw_text": "x"})
    assert result.status == "failed"
    assert result.error and "ghost" in result.error
    assert result.final_state == "extract"
