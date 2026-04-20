"""Workflow orchestration.

:class:`WorkflowEngine` drives a :class:`~flowagent.types.WorkflowDefinition`
from its initial state to a terminal state. Phase 1 scope:

* Validation is enforced at run-time.
* Deterministic + agentic states are executed via :class:`StateExecutor`.
* Each state transition is counted against ``max_total_steps`` as a hard
  safety bound.
* ``on_enter``/``on_exit`` callbacks on ``StateDef`` fire around each step.

Checkpointing, hooks, error policy, and resume-from-checkpoint are Phase 3+.
The engine exposes hook hook-points as optional callback arguments so Phase 3
can layer on without changing the signature.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from flowagent.adapters.base import LLMAdapter
from flowagent.errors import (
    ConfigurationError,
    FlowAgentError,
    WorkflowValidationError,
)
from flowagent.state_executor import StateExecutor
from flowagent.state_store import StateStore
from flowagent.types import (
    StateExecutionResult,
    WorkflowDefinition,
    WorkflowExecutionResult,
)


OnStateEnter = Callable[[str, str, StateStore], None]
OnStateExit = Callable[[str, str, StateExecutionResult, StateStore], None]


class WorkflowEngine:
    """Drive a workflow from start to terminal state."""

    def __init__(
        self,
        llm_adapter: Optional[LLMAdapter] = None,
        executor: Optional[StateExecutor] = None,
    ) -> None:
        if executor is None:
            executor = StateExecutor(llm_adapter)
        self.executor = executor

    def run(
        self,
        workflow: WorkflowDefinition,
        initial_input: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        on_state_enter: Optional[OnStateEnter] = None,
        on_state_exit: Optional[OnStateExit] = None,
    ) -> WorkflowExecutionResult:
        """Execute ``workflow`` from its initial state to termination."""
        errors = workflow.validate()
        if errors:
            raise WorkflowValidationError(errors)

        run_id = run_id or str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)

        state_store = StateStore(initial_input)
        current_state = workflow.initial_state
        trace: List[StateExecutionResult] = []
        visits: Dict[str, int] = {}

        status = "completed"
        error: Optional[str] = None
        step_count = 0

        try:
            while current_state not in workflow.terminal_states:
                if step_count >= workflow.max_total_steps:
                    status = "max_steps_exceeded"
                    error = (
                        f"Workflow exceeded max_total_steps "
                        f"({workflow.max_total_steps}) before reaching a "
                        "terminal state"
                    )
                    break

                if current_state not in workflow.states:
                    raise ConfigurationError(
                        f"Current state '{current_state}' not defined in workflow"
                    )

                state_def = workflow.states[current_state]
                visits[current_state] = visits.get(current_state, 0) + 1

                if state_def.on_enter is not None:
                    state_def.on_enter(state_store)
                if on_state_enter is not None:
                    on_state_enter(run_id, current_state, state_store)

                result = self.executor.execute(
                    state_def=state_def,
                    state_store=state_store,
                    global_system_prompt=workflow.global_system_prompt,
                )

                if result.output_data:
                    state_store.update(
                        result.output_data, source_state=current_state
                    )

                trace.append(result)

                if on_state_exit is not None:
                    on_state_exit(run_id, current_state, result, state_store)
                if state_def.on_exit is not None:
                    state_def.on_exit(state_store, result.transition)

                current_state = result.transition
                step_count += 1

        except FlowAgentError as exc:
            status = "failed"
            error = str(exc)

        completed_at = datetime.now(timezone.utc)

        return WorkflowExecutionResult(
            workflow_name=workflow.name,
            workflow_version=workflow.version,
            run_id=run_id,
            final_state=current_state,
            state_store=state_store.to_dict(),
            execution_trace=trace,
            total_llm_calls=sum(r.llm_calls_count for r in trace),
            total_tool_calls=sum(len(r.tool_calls_made) for r in trace),
            started_at=started_at,
            completed_at=completed_at,
            status=status,
            error=error,
        )
