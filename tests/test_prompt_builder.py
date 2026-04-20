"""Unit tests for :mod:`flowagent.prompt_builder`."""

from __future__ import annotations

from flowagent.prompt_builder import build_system_prompt
from flowagent.tool import tool
from flowagent.transition import TRANSITION_TOOL_NAME
from flowagent.types import StateDef, StateMode, TransitionDef


@tool(name="dummy", description="dummy")
def _dummy() -> dict:
    return {}


def _state(**kwargs) -> StateDef:
    defaults = dict(
        state_id="s",
        description="State description",
        mode=StateMode.AGENTIC,
        tools=[_dummy],
        transitions=[
            TransitionDef("done", "Completed"),
            TransitionDef("retry", "Try again"),
        ],
    )
    defaults.update(kwargs)
    return StateDef(**defaults)


def test_prompt_includes_description_and_transitions():
    prompt = build_system_prompt(_state())
    assert "## Current Task" in prompt
    assert "State description" in prompt
    assert "## Available Transitions" in prompt
    assert '"done": Completed' in prompt
    assert '"retry": Try again' in prompt
    assert TRANSITION_TOOL_NAME in prompt


def test_prompt_prepends_global():
    prompt = build_system_prompt(_state(), global_prompt="You are an agent.")
    assert prompt.startswith("You are an agent.")


def test_prompt_template_override_is_verbatim():
    prompt = build_system_prompt(
        _state(system_prompt_template="OVERRIDE"),
        global_prompt="ignored",
    )
    assert prompt == "OVERRIDE"


def test_prompt_without_transitions():
    state = _state(transitions=[])
    prompt = build_system_prompt(state)
    assert "## Available Transitions" not in prompt
    assert "State description" in prompt
