"""Workflow graph validation rules."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from flowagent.types import WorkflowDefinition

from flowagent.types import StateMode


def validate_workflow(workflow: WorkflowDefinition) -> List[str]:
    """Validate workflow graph integrity. Returns list of error strings."""
    errors: list[str] = []

    # 1. Initial state must exist
    if workflow.initial_state not in workflow.states:
        errors.append(
            f"Initial state '{workflow.initial_state}' not defined in states"
        )

    # 2. All terminal states must exist
    for ts in workflow.terminal_states:
        if ts not in workflow.states:
            errors.append(f"Terminal state '{ts}' not defined in states")

    # 3. Terminal states must have no outgoing transitions
    for ts in workflow.terminal_states:
        if ts in workflow.states and workflow.states[ts].transitions:
            errors.append(f"Terminal state '{ts}' has outgoing transitions")

    # 4. All transition targets must reference existing states
    for state_id, state_def in workflow.states.items():
        for t in state_def.transitions:
            if t.target not in workflow.states:
                errors.append(
                    f"State '{state_id}' has transition to "
                    f"non-existent state '{t.target}'"
                )

    # 5. Non-terminal states must have at least one transition
    for state_id, state_def in workflow.states.items():
        if state_id not in workflow.terminal_states:
            if not state_def.transitions:
                errors.append(
                    f"Non-terminal state '{state_id}' has no transitions"
                )

    # 6. All states must be reachable from initial state
    if workflow.initial_state in workflow.states:
        reachable = workflow._find_reachable_states(workflow.initial_state)
        for state_id in workflow.states:
            if state_id not in reachable:
                errors.append(
                    f"State '{state_id}' is not reachable from "
                    f"initial state '{workflow.initial_state}'"
                )

        # 7. At least one terminal state must be reachable
        if not any(ts in reachable for ts in workflow.terminal_states):
            errors.append(
                "No terminal state is reachable from initial state"
            )

    # 8. Deterministic states must have handler or single transition
    for state_id, state_def in workflow.states.items():
        if state_def.mode == StateMode.DETERMINISTIC:
            if not state_def.handler and not state_def.transition_resolver:
                if len(state_def.transitions) > 1:
                    errors.append(
                        f"Deterministic state '{state_id}' has "
                        f"multiple transitions but no handler/resolver"
                    )

    # 9. Agentic states must have at least one tool
    for state_id, state_def in workflow.states.items():
        if state_def.mode == StateMode.AGENTIC:
            if not state_def.tools:
                errors.append(
                    f"Agentic state '{state_id}' has no tools"
                )

    return errors
