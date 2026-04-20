"""Auto-generated ``declare_transition`` tool and interception helpers.

Every agentic state gets an implicit ``declare_transition(transition, reasoning)``
tool appended to its work tools. When the LLM invokes it, the
:class:`~flowagent.state_executor.StateExecutor` treats the call as the terminal
action for the state and exits the tool-calling loop.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field

from flowagent.tool import ToolDef
from flowagent.types import TransitionDef

TRANSITION_TOOL_NAME = "declare_transition"


class TransitionParams(BaseModel):
    """Schema for the auto-generated ``declare_transition`` tool."""

    transition: str = Field(
        ..., description="The target state_id chosen from allowed transitions."
    )
    reasoning: str = Field(
        ..., description="Concise justification for why this transition was chosen."
    )


def make_transition_tool(transitions: Dict[str, str]) -> ToolDef:
    """Build the ``declare_transition`` tool for a given set of transitions.

    ``transitions`` maps target state_id -> human-readable description. The
    descriptions are inlined into the tool description so the LLM sees them
    alongside the parameter schema.
    """
    if not transitions:
        raise ValueError("Cannot build declare_transition tool with no transitions")

    listing = "\n".join(f"  - '{target}': {desc}" for target, desc in transitions.items())
    description = (
        "Call this tool when you have completed your work in the current step "
        "and are ready to move to the next state. Choose exactly one of the "
        f"available transitions based on your findings.\nAvailable transitions:\n{listing}"
    )

    return ToolDef(
        name=TRANSITION_TOOL_NAME,
        description=description,
        parameters_schema=TransitionParams,
        return_schema=None,
        function=_transition_passthrough,
        is_async=False,
    )


def _transition_passthrough(**kwargs: Any) -> Dict[str, Any]:
    """Passthrough used by :func:`make_transition_tool`.

    The executor intercepts ``declare_transition`` before invoking the tool, so
    this body is only reached if someone calls it directly (e.g. from tests).
    """
    return dict(kwargs)


def transitions_as_dict(transitions: Iterable[TransitionDef]) -> Dict[str, str]:
    """Project a list of :class:`TransitionDef` into the ``target -> desc`` mapping."""
    return {t.target: t.description for t in transitions}


def prepare_state_tools(
    work_tools: List[ToolDef],
    transitions: Iterable[TransitionDef],
) -> List[ToolDef]:
    """Return ``work_tools`` with the auto-generated transition tool appended.

    If the state has no outgoing transitions (e.g. a terminal state), the work
    tools are returned unchanged.
    """
    mapping = transitions_as_dict(transitions)
    if not mapping:
        return list(work_tools)
    return [*work_tools, make_transition_tool(mapping)]


def extract_transition_call(
    tool_calls: Iterable[Any],
) -> Optional[Any]:
    """Return the first ``declare_transition`` tool call, or ``None``."""
    for tc in tool_calls:
        if getattr(tc, "tool_name", None) == TRANSITION_TOOL_NAME:
            return tc
    return None
