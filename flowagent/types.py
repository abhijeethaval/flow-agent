"""Core dataclasses and enums for FlowAgent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel

if TYPE_CHECKING:
    from flowagent.tool import ToolDef


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StateMode(str, Enum):
    """How the LLM is invoked within a state."""

    AGENTIC = "agentic"  # Tool-calling loop + transition decision
    CONVERSATIONAL = "conversational"  # Multi-turn chat within state
    DETERMINISTIC = "deterministic"  # No LLM call — pure function, auto-transition


# ---------------------------------------------------------------------------
# Workflow graph primitives
# ---------------------------------------------------------------------------


@dataclass
class TransitionDef:
    """A possible outgoing edge from a state."""

    target: str  # Target state ID
    description: str  # Semantic description (shown to LLM)
    condition: Optional[str] = None  # Human-readable condition label for docs/audit


@dataclass
class StateDef:
    """Definition of a single workflow state."""

    state_id: str  # Unique identifier
    description: str  # What this state does (for system prompt)
    mode: StateMode = StateMode.AGENTIC  # Execution mode

    # Tools (only for AGENTIC and CONVERSATIONAL modes)
    tools: List[ToolDef] = field(default_factory=list)
    max_tool_calls: int = 5  # Bound tool-calling loop within state

    # Transitions (outgoing edges)
    transitions: List[TransitionDef] = field(default_factory=list)

    # Context shaping
    input_schema: Optional[Type[BaseModel]] = None  # What to extract from state store
    output_schema: Optional[Type[BaseModel]] = None  # What this state writes

    # Deterministic mode only
    handler: Optional[Callable] = None  # Pure function for DETERMINISTIC states
    transition_resolver: Optional[Callable] = None  # (state_store) -> target_state_id

    # System prompt override (optional — otherwise auto-generated)
    system_prompt_template: Optional[str] = None

    # Hooks
    on_enter: Optional[Callable] = None  # Called when entering this state
    on_exit: Optional[Callable] = None  # Called when leaving this state


@dataclass
class WorkflowDefinition:
    """The complete workflow graph."""

    name: str
    version: str
    description: str

    states: Dict[str, StateDef]  # state_id -> StateDef
    initial_state: str  # Entry point state_id
    terminal_states: List[str]  # States that end the workflow

    # Global configuration
    global_system_prompt: Optional[str] = None  # Base system prompt for all states
    max_total_steps: int = 50  # Global safety bound

    def validate(self) -> List[str]:
        """Validate graph integrity. Returns list of errors, empty if valid."""
        # Delegated to validation.py — imported here to keep this module lean.
        from flowagent.validation import validate_workflow

        return validate_workflow(self)

    def _find_reachable_states(self, start: str) -> set[str]:
        """BFS to find all states reachable from *start*."""
        visited: set[str] = set()
        queue = [start]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if current in self.states:
                for t in self.states[current].transitions:
                    if t.target not in visited:
                        queue.append(t.target)
        return visited


# ---------------------------------------------------------------------------
# LLM interaction types
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""

    id: str  # Unique call ID (from LLM)
    tool_name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a tool call."""

    call_id: str  # Matches ToolCall.id
    tool_name: str
    result: Dict[str, Any]
    success: bool
    error: Optional[str] = None


@dataclass
class LLMResponse:
    """Parsed response from an LLM call."""

    tool_calls: List[ToolCall]  # Empty if no tool calls
    text_content: Optional[str]  # Text response (present when no tool calls)
    transition: Optional[str]  # Extracted transition choice
    reasoning: Optional[str]  # Why the agent chose this transition
    raw_response: Any = None  # Provider-specific raw response


# ---------------------------------------------------------------------------
# Execution records
# ---------------------------------------------------------------------------


@dataclass
class ToolCallRecord:
    """Audit record of a tool call within a state."""

    timestamp: datetime
    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    success: bool
    duration_ms: float


@dataclass
class StateExecutionResult:
    """Result of executing a single state."""

    state_id: str
    transition: str  # Chosen target state
    reasoning: Optional[str]  # Agent's reasoning for transition
    output_data: Dict[str, Any]  # Data to write to state store
    tool_calls_made: List[ToolCallRecord]  # Audit trail
    llm_calls_count: int  # Number of LLM invocations in this state
    messages: List[Dict[str, Any]]  # Full message history within this state


@dataclass
class WorkflowExecutionResult:
    """Final result of a complete workflow run."""

    workflow_name: str
    workflow_version: str
    run_id: str
    final_state: str
    state_store: Dict[str, Any]
    execution_trace: List[StateExecutionResult]
    total_llm_calls: int
    total_tool_calls: int
    started_at: datetime
    completed_at: datetime
    status: str  # "completed" | "failed" | "max_steps_exceeded"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Error policy
# ---------------------------------------------------------------------------


@dataclass
class ErrorPolicy:
    """Configure error handling behavior per state or globally."""

    # On ToolExecutionError
    tool_error_action: str = "feed_to_llm"
    # "feed_to_llm" — Send error as tool result, let agent decide
    # "retry"       — Retry the tool call (up to max_retries)
    # "fail"        — Fail the state immediately
    max_tool_retries: int = 2

    # On InvalidTransitionError
    invalid_transition_action: str = "retry_llm"
    # "retry_llm" — Ask LLM again with error feedback
    # "fail"      — Fail the state
    max_transition_retries: int = 2

    # On MaxToolCallsExceeded
    max_steps_action: str = "fail"
    # "fail"              — Fail the state
    # "force_transition"  — Ask LLM one final time for transition only


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


@dataclass
class Checkpoint:
    """A snapshot of workflow state at a point in time."""

    run_id: str
    current_state: str  # State about to be executed
    store_data: Dict[str, Any]  # Full state store snapshot
    trace: List[StateExecutionResult]  # Execution history so far
    timestamp: datetime
