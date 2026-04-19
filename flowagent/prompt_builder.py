"""System prompt assembly for agentic state execution.

The system prompt is structured as:

1. Optional global prompt (agent identity / persona).
2. State-specific description (what this state does).
3. Available transitions with their descriptions.
4. Instruction to call ``declare_transition`` when finished.

A state may override the generated prompt by providing ``system_prompt_template``
on its :class:`~flowagent.types.StateDef`. When a template is provided it is
used verbatim and the standard sections are skipped.
"""

from __future__ import annotations

from typing import Optional

from flowagent.transition import TRANSITION_TOOL_NAME
from flowagent.types import StateDef


def build_system_prompt(
    state_def: StateDef,
    global_prompt: Optional[str] = None,
) -> str:
    """Construct the system prompt for an agentic state execution."""
    if state_def.system_prompt_template is not None:
        return state_def.system_prompt_template

    parts: list[str] = []

    if global_prompt:
        parts.append(global_prompt.strip())

    parts.append(f"## Current Task\n{state_def.description.strip()}")

    if state_def.transitions:
        transition_text = "\n".join(
            f'- "{t.target}": {t.description}' for t in state_def.transitions
        )
        parts.append(
            "## Available Transitions\n"
            "When you have completed your work in this step, call the "
            f"`{TRANSITION_TOOL_NAME}` tool with one of the transitions below "
            "and a concise reasoning string:\n"
            f"{transition_text}\n\n"
            "Do not declare a transition until you have gathered the "
            "information you need from your tools."
        )

    return "\n\n".join(parts)
