"""Unit tests for :mod:`flowagent.validation`."""

from __future__ import annotations

import pytest

from flowagent.errors import WorkflowValidationError
from flowagent.tool import tool
from flowagent.types import StateDef, StateMode, TransitionDef, WorkflowDefinition


@tool(name="work", description="w")
def _work() -> dict:
    return {}


def _terminal(state_id: str = "done") -> StateDef:
    return StateDef(
        state_id=state_id,
        description="terminal",
        mode=StateMode.DETERMINISTIC,
        transitions=[],
    )


def _agentic(state_id: str, transitions, tools=None) -> StateDef:
    return StateDef(
        state_id=state_id,
        description=f"state {state_id}",
        mode=StateMode.AGENTIC,
        tools=tools if tools is not None else [_work],
        transitions=list(transitions),
    )


def _wf(states, initial="start", terminals=("done",)):
    return WorkflowDefinition(
        name="wf",
        version="0.1",
        description="",
        states={s.state_id: s for s in states},
        initial_state=initial,
        terminal_states=list(terminals),
    )


def test_valid_workflow():
    wf = _wf(
        states=[
            _agentic("start", [TransitionDef("done", "Finish")]),
            _terminal("done"),
        ]
    )
    assert wf.validate() == []


def test_missing_initial_state():
    wf = _wf(states=[_terminal("done")], initial="nope")
    errors = wf.validate()
    assert any("Initial state 'nope'" in e for e in errors)


def test_missing_terminal_state():
    wf = _wf(
        states=[_agentic("start", [TransitionDef("done", "F")])],
        terminals=("done",),
    )
    errors = wf.validate()
    assert any("Terminal state 'done'" in e for e in errors)


def test_terminal_with_outgoing_transitions_is_invalid():
    wf = _wf(
        states=[
            _agentic("start", [TransitionDef("done", "F")]),
            StateDef(
                state_id="done",
                description="t",
                mode=StateMode.DETERMINISTIC,
                transitions=[TransitionDef("start", "loop")],
            ),
        ],
        terminals=("done",),
    )
    errors = wf.validate()
    assert any("outgoing transitions" in e for e in errors)


def test_transition_to_non_existent_state():
    wf = _wf(
        states=[
            _agentic("start", [TransitionDef("ghost", "never")]),
            _terminal("done"),
        ]
    )
    errors = wf.validate()
    assert any("non-existent state 'ghost'" in e for e in errors)


def test_non_terminal_without_transitions():
    wf = _wf(
        states=[
            _agentic("start", []),
            _terminal("done"),
        ]
    )
    errors = wf.validate()
    assert any("no transitions" in e for e in errors)


def test_unreachable_state():
    wf = _wf(
        states=[
            _agentic("start", [TransitionDef("done", "F")]),
            _agentic("orphan", [TransitionDef("done", "F")]),
            _terminal("done"),
        ]
    )
    errors = wf.validate()
    assert any("'orphan' is not reachable" in e for e in errors)


def test_no_reachable_terminal_state():
    wf = _wf(
        states=[
            _agentic("start", [TransitionDef("start", "loop")]),
            _terminal("done"),
        ],
        terminals=("done",),
    )
    errors = wf.validate()
    assert any("No terminal state is reachable" in e for e in errors)


def test_agentic_state_without_tools():
    wf = _wf(
        states=[
            _agentic("start", [TransitionDef("done", "F")], tools=[]),
            _terminal("done"),
        ]
    )
    errors = wf.validate()
    assert any("no tools" in e for e in errors)


def test_deterministic_with_ambiguous_transitions():
    wf = _wf(
        states=[
            StateDef(
                state_id="start",
                description="",
                mode=StateMode.DETERMINISTIC,
                transitions=[
                    TransitionDef("done", "D"),
                    TransitionDef("other", "O"),
                ],
            ),
            _terminal("done"),
            _terminal("other"),
        ],
        terminals=("done", "other"),
    )
    errors = wf.validate()
    assert any("multiple transitions but no handler/resolver" in e for e in errors)


def test_workflow_validation_error_wraps_errors():
    exc = WorkflowValidationError(["a", "b"])
    assert exc.errors == ["a", "b"]
    assert "a" in str(exc) and "b" in str(exc)
