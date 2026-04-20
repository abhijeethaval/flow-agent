"""Unit tests for :mod:`flowagent.transition`."""

from __future__ import annotations

import pytest

from flowagent.transition import (
    TRANSITION_TOOL_NAME,
    TransitionParams,
    extract_transition_call,
    make_transition_tool,
    prepare_state_tools,
    transitions_as_dict,
)
from flowagent.tool import ToolDef, tool
from flowagent.types import ToolCall, TransitionDef


@tool(name="noop", description="no op")
def _noop_tool() -> dict:
    return {}


def test_make_transition_tool_includes_targets_in_description():
    td = make_transition_tool({"done": "Finished", "retry": "Try again"})
    assert td.name == TRANSITION_TOOL_NAME
    assert "'done': Finished" in td.description
    assert "'retry': Try again" in td.description
    assert td.parameters_schema is TransitionParams


def test_make_transition_tool_requires_transitions():
    with pytest.raises(ValueError):
        make_transition_tool({})


def test_transition_params_requires_fields():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TransitionParams()  # type: ignore[call-arg]


def test_transitions_as_dict():
    tds = [TransitionDef("a", "desc-a"), TransitionDef("b", "desc-b")]
    assert transitions_as_dict(tds) == {"a": "desc-a", "b": "desc-b"}


def test_prepare_state_tools_appends_transition_tool():
    prepared = prepare_state_tools([_noop_tool], [TransitionDef("done", "Finish")])
    assert [t.name for t in prepared] == ["noop", TRANSITION_TOOL_NAME]


def test_prepare_state_tools_no_transitions():
    prepared = prepare_state_tools([_noop_tool], [])
    assert prepared == [_noop_tool]


def test_extract_transition_call_found():
    calls = [
        ToolCall(id="1", tool_name="work", arguments={}),
        ToolCall(id="2", tool_name=TRANSITION_TOOL_NAME, arguments={"transition": "done"}),
    ]
    found = extract_transition_call(calls)
    assert found is not None and found.id == "2"


def test_extract_transition_call_missing():
    calls = [ToolCall(id="1", tool_name="work", arguments={})]
    assert extract_transition_call(calls) is None
