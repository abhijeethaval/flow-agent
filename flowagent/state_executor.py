"""Per-state agentic execution loop.

:class:`StateExecutor` runs the LLM-driven tool-calling loop within a single
state. It enforces tool scope, intercepts the auto-generated
``declare_transition`` tool, validates the chosen transition, and raises the
appropriate :class:`~flowagent.errors.StateExecutionError` subclass on any
scope / contract violation.

Only AGENTIC and DETERMINISTIC modes are handled here; CONVERSATIONAL mode is
a Phase 4 deliverable and currently falls back to AGENTIC semantics (no user
injection between calls).
"""

from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from flowagent.adapters.base import LLMAdapter
from flowagent.errors import (
    ConfigurationError,
    InvalidTransitionError,
    MaxToolCallsExceeded,
    NoTransitionDeclaredError,
    ToolNotAllowedError,
)
from flowagent.prompt_builder import build_system_prompt
from flowagent.state_store import StateStore
from flowagent.tool import ToolDef
from flowagent.transition import (
    TRANSITION_TOOL_NAME,
    prepare_state_tools,
    transitions_as_dict,
)
from flowagent.types import (
    StateDef,
    StateExecutionResult,
    StateMode,
    ToolCall,
    ToolCallRecord,
)


OnToolCall = Callable[[str, ToolCall], None]
OnToolResult = Callable[[str, ToolCallRecord], None]
OnLLMCall = Callable[[str, List[Dict[str, Any]], Any], None]


class StateExecutor:
    """Execute a single state: either the agentic tool-loop or a pure handler."""

    def __init__(self, llm_adapter: Optional[LLMAdapter]) -> None:
        self.llm = llm_adapter

    def execute(
        self,
        state_def: StateDef,
        state_store: StateStore,
        global_system_prompt: Optional[str] = None,
        on_tool_call: Optional[OnToolCall] = None,
        on_tool_result: Optional[OnToolResult] = None,
        on_llm_call: Optional[OnLLMCall] = None,
    ) -> StateExecutionResult:
        """Run the state and return its :class:`StateExecutionResult`."""
        if state_def.mode == StateMode.DETERMINISTIC:
            return self._execute_deterministic(state_def, state_store)

        if self.llm is None:
            raise ConfigurationError(
                f"State '{state_def.state_id}' requires an LLM adapter"
            )

        return self._execute_agentic(
            state_def,
            state_store,
            global_system_prompt,
            on_tool_call,
            on_tool_result,
            on_llm_call,
        )

    # ------------------------------------------------------------------
    # Agentic path
    # ------------------------------------------------------------------

    def _execute_agentic(
        self,
        state_def: StateDef,
        state_store: StateStore,
        global_system_prompt: Optional[str],
        on_tool_call: Optional[OnToolCall],
        on_tool_result: Optional[OnToolResult],
        on_llm_call: Optional[OnLLMCall],
    ) -> StateExecutionResult:
        system_prompt = build_system_prompt(state_def, global_system_prompt)

        if state_def.input_schema:
            context = state_store.extract(state_def.input_schema).model_dump()
        else:
            context = state_store.data

        transitions_map = transitions_as_dict(state_def.transitions)
        work_tools_by_name = {t.name: t for t in state_def.tools}
        all_tools = prepare_state_tools(state_def.tools, state_def.transitions)
        allowed_tool_names = {t.name for t in all_tools}

        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": self._format_context(context)}
        ]

        tool_records: List[ToolCallRecord] = []
        aggregated_tool_outputs: Dict[str, Any] = {}
        llm_calls = 0

        for _ in range(state_def.max_tool_calls + 1):
            response = self.llm.call(
                system_prompt=system_prompt,
                messages=messages,
                tools=all_tools,
                available_transitions=transitions_map,
            )
            llm_calls += 1
            if on_llm_call is not None:
                on_llm_call(state_def.state_id, messages, response)

            # Path 1: adapter parsed a transition directly (no tool call path).
            if not response.tool_calls and response.transition is not None:
                return self._finalize_transition(
                    state_def=state_def,
                    transitions_map=transitions_map,
                    transition=response.transition,
                    reasoning=response.reasoning,
                    tool_outputs=aggregated_tool_outputs,
                    tool_records=tool_records,
                    llm_calls=llm_calls,
                    messages=messages,
                )

            # Path 2: no tool calls, no transition — the agent stopped early.
            if not response.tool_calls:
                raise NoTransitionDeclaredError(
                    f"Agent in state '{state_def.state_id}' returned a final "
                    "response without calling declare_transition or any tool."
                )

            # Path 3: tool calls present. Execute work tools, intercept
            # declare_transition. Record the assistant turn once with all
            # tool calls, then append a single tool message per call.
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "name": tc.tool_name,
                            "arguments": tc.arguments,
                        }
                        for tc in response.tool_calls
                    ],
                }
            )

            for tc in response.tool_calls:
                if tc.tool_name == TRANSITION_TOOL_NAME:
                    transition = tc.arguments.get("transition")
                    reasoning = tc.arguments.get("reasoning") or response.reasoning
                    return self._finalize_transition(
                        state_def=state_def,
                        transitions_map=transitions_map,
                        transition=transition,
                        reasoning=reasoning,
                        tool_outputs=aggregated_tool_outputs,
                        tool_records=tool_records,
                        llm_calls=llm_calls,
                        messages=messages,
                    )

                if tc.tool_name not in allowed_tool_names:
                    raise ToolNotAllowedError(
                        f"Tool '{tc.tool_name}' not allowed in state "
                        f"'{state_def.state_id}'. Allowed: "
                        f"{sorted(allowed_tool_names)}"
                    )

                tool_def = work_tools_by_name[tc.tool_name]
                if on_tool_call is not None:
                    on_tool_call(state_def.state_id, tc)

                record, tool_output, success = self._run_tool(tool_def, tc)
                tool_records.append(record)
                if on_tool_result is not None:
                    on_tool_result(state_def.state_id, record)

                if success and isinstance(tool_output, dict):
                    aggregated_tool_outputs.update(tool_output)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "tool_name": tc.tool_name,
                        "content": json.dumps(record.result, default=str),
                    }
                )

        raise MaxToolCallsExceeded(
            f"State '{state_def.state_id}' exceeded max_tool_calls "
            f"({state_def.max_tool_calls}) without declaring a transition"
        )

    # ------------------------------------------------------------------
    # Deterministic path
    # ------------------------------------------------------------------

    def _execute_deterministic(
        self,
        state_def: StateDef,
        state_store: StateStore,
    ) -> StateExecutionResult:
        output: Dict[str, Any] = {}
        if state_def.handler:
            raw = state_def.handler(state_store.data)
            if raw is None:
                output = {}
            elif isinstance(raw, dict):
                output = raw
            else:
                raise ConfigurationError(
                    f"Deterministic handler for '{state_def.state_id}' must "
                    f"return a dict or None, got {type(raw).__name__}"
                )

        if state_def.transition_resolver is not None:
            transition = state_def.transition_resolver(state_store.data)
        elif not state_def.transitions:
            transition = ""  # Terminal state — engine stops before using this.
        elif len(state_def.transitions) == 1:
            transition = state_def.transitions[0].target
        else:
            raise ConfigurationError(
                f"Deterministic state '{state_def.state_id}' has multiple "
                "transitions but no transition_resolver"
            )

        if transition and state_def.transitions:
            allowed = {t.target for t in state_def.transitions}
            if transition not in allowed:
                raise InvalidTransitionError(
                    f"Resolver for '{state_def.state_id}' returned "
                    f"'{transition}' not in {sorted(allowed)}"
                )

        return StateExecutionResult(
            state_id=state_def.state_id,
            transition=transition,
            reasoning="deterministic",
            output_data=output,
            tool_calls_made=[],
            llm_calls_count=0,
            messages=[],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_tool(
        self, tool_def: ToolDef, call: ToolCall
    ) -> tuple[ToolCallRecord, Any, bool]:
        start = time.perf_counter()
        try:
            raw = tool_def.function(**call.arguments)
            success = True
        except Exception as exc:  # noqa: BLE001 — feed errors to audit trail
            raw = {"error": str(exc), "error_type": type(exc).__name__}
            success = False
        duration_ms = (time.perf_counter() - start) * 1000.0

        result_dict = self._normalize_tool_result(raw)

        record = ToolCallRecord(
            timestamp=datetime.now(timezone.utc),
            tool_name=call.tool_name,
            arguments=copy.deepcopy(call.arguments),
            result=result_dict,
            success=success,
            duration_ms=duration_ms,
        )
        return record, result_dict, success

    @staticmethod
    def _normalize_tool_result(raw: Any) -> Dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        if hasattr(raw, "model_dump"):
            return raw.model_dump()
        return {"value": raw}

    def _finalize_transition(
        self,
        *,
        state_def: StateDef,
        transitions_map: Dict[str, str],
        transition: Optional[str],
        reasoning: Optional[str],
        tool_outputs: Dict[str, Any],
        tool_records: List[ToolCallRecord],
        llm_calls: int,
        messages: List[Dict[str, Any]],
    ) -> StateExecutionResult:
        if transition is None:
            raise NoTransitionDeclaredError(
                f"Agent in state '{state_def.state_id}' attempted to "
                "declare_transition without specifying a transition"
            )
        if transition not in transitions_map:
            raise InvalidTransitionError(
                f"Transition '{transition}' not allowed from state "
                f"'{state_def.state_id}'. Allowed: "
                f"{sorted(transitions_map)}"
            )

        output_data = self._project_output(tool_outputs, state_def)

        return StateExecutionResult(
            state_id=state_def.state_id,
            transition=transition,
            reasoning=reasoning,
            output_data=output_data,
            tool_calls_made=list(tool_records),
            llm_calls_count=llm_calls,
            messages=list(messages),
        )

    @staticmethod
    def _project_output(
        aggregated: Dict[str, Any], state_def: StateDef
    ) -> Dict[str, Any]:
        if not aggregated:
            return {}
        if state_def.output_schema is None:
            return copy.deepcopy(aggregated)
        fields = state_def.output_schema.model_fields.keys()
        return {k: copy.deepcopy(aggregated[k]) for k in fields if k in aggregated}

    @staticmethod
    def _format_context(context: Dict[str, Any]) -> str:
        if not context:
            return "No prior context is available. Begin by calling your tools."
        return (
            "The following context from the workflow state store is available "
            "for this step:\n"
            f"{json.dumps(context, indent=2, default=str)}"
        )
