# FlowAgent: State-Gated Agentic Workflow Framework

## Specification Document v1.0

**Author:** Abhijeet  
**Date:** February 2026  
**Status:** Draft — Ready for Implementation

---

## 1. Executive Summary

FlowAgent is a Python framework for building deterministic workflow applications powered by a single LLM agent. The core innovation is **state-scoped tool binding with agent-driven navigation**: the workflow is defined as a directed graph of states, each state exposes a scoped set of tools to the LLM, and the agent decides which available transition to take based on tool results, user input, and its own reasoning.

The workflow graph constrains **where** the agent can go. The agent decides **which path** to take through the graph.

### Key Principles

1. **Single agent, single identity** — One LLM, one system prompt template, one orchestrator. No multi-agent persona switching.
2. **Stateless between states** — Each state gets a fresh, constructed context from the state store. No accumulated chat history between states.
3. **Agentic within states** — Within a state, the agent runs a standard tool-calling loop: call tools, see results, decide to call more tools or declare a transition.
4. **Deterministic graph, agentic navigation** — The graph topology is immutable. The agent chooses among predefined outgoing transitions per state, including back-edges.
5. **Checkpoint-based persistence** — Snapshot state between steps. No event sourcing. Resume from last checkpoint on failure.
6. **LLM-provider agnostic** — Support any LLM with function-calling/tool-use capability (OpenAI, Anthropic, Azure, local models).

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    FlowAgent Runtime                     │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │  Workflow     │    │  Workflow     │    │  State    │ │
│  │  Definition   │───►│  Engine      │◄──►│  Store    │ │
│  │  (Graph DSL)  │    │  (Runtime)   │    │  (Data)   │ │
│  └──────────────┘    └──────┬───────┘    └───────────┘ │
│                             │                           │
│                    ┌────────▼────────┐                  │
│                    │ State Executor  │                  │
│                    │                 │                  │
│                    │ ┌─────────────┐ │                  │
│                    │ │ LLM Adapter │ │                  │
│                    │ └──────┬──────┘ │                  │
│                    │        │        │                  │
│                    │ ┌──────▼──────┐ │                  │
│                    │ │ Tool Runner │ │                  │
│                    │ └─────────────┘ │                  │
│                    └─────────────────┘                  │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │ Checkpoint   │    │  Audit Log   │    │  Hooks /  │ │
│  │ Persistence  │    │  (Trace)     │    │  Plugins  │ │
│  └──────────────┘    └──────────────┘    └───────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Core Abstractions

### 3.1 WorkflowDefinition

The immutable specification of the workflow graph. Defined once, versioned per release.

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Callable, Any
from pydantic import BaseModel
from enum import Enum


class StateMode(str, Enum):
    """How the LLM is invoked within this state."""
    AGENTIC = "agentic"              # Tool-calling loop + transition decision
    CONVERSATIONAL = "conversational" # Multi-turn chat within state (e.g., user clarification)
    DETERMINISTIC = "deterministic"   # No LLM call — pure function, auto-transition


@dataclass
class TransitionDef:
    """A possible outgoing edge from a state."""
    target: str                       # Target state ID
    description: str                  # Semantic description (shown to LLM)
    condition: Optional[str] = None   # Optional human-readable condition label for docs/audit


@dataclass
class StateDef:
    """Definition of a single workflow state."""
    state_id: str                              # Unique identifier
    description: str                           # What this state does (for system prompt)
    mode: StateMode = StateMode.AGENTIC        # Execution mode

    # Tools (only for AGENTIC and CONVERSATIONAL modes)
    tools: List[Callable] = field(default_factory=list)
    max_tool_calls: int = 5                    # Bound tool-calling loop within state

    # Transitions (outgoing edges)
    transitions: List[TransitionDef] = field(default_factory=list)

    # Context shaping
    input_schema: Optional[Type[BaseModel]] = None   # What to extract from state store
    output_schema: Optional[Type[BaseModel]] = None  # What this state writes to state store

    # Deterministic mode only
    handler: Optional[Callable] = None                # Pure function for DETERMINISTIC states
    transition_resolver: Optional[Callable] = None    # (state_store) -> target_state_id

    # System prompt override (optional — otherwise auto-generated)
    system_prompt_template: Optional[str] = None

    # Hooks
    on_enter: Optional[Callable] = None               # Called when entering this state
    on_exit: Optional[Callable] = None                # Called when leaving this state


@dataclass
class WorkflowDefinition:
    """The complete workflow graph."""
    name: str
    version: str
    description: str

    states: Dict[str, StateDef]                       # state_id -> StateDef
    initial_state: str                                 # Entry point state_id
    terminal_states: List[str]                         # States that end the workflow

    # Global configuration
    global_system_prompt: Optional[str] = None         # Base system prompt for all states
    max_total_steps: int = 50                          # Global safety bound

    def validate(self) -> List[str]:
        """Validate graph integrity. Returns list of errors, empty if valid."""
        # Implementation: see Section 6.1
        pass
```

### 3.2 StateStore

The shared data layer that accumulates results across states. Immutable snapshots for auditability.

```python
from datetime import datetime
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel
import copy


class StateStore:
    """
    Accumulates data across workflow states.
    Each update creates an immutable snapshot.
    The state store is the ONLY mechanism for passing data between states.
    """

    def __init__(self, initial_data: Optional[Dict[str, Any]] = None):
        self._data: Dict[str, Any] = initial_data or {}
        self._history: List[StateStoreSnapshot] = []

    @property
    def data(self) -> Dict[str, Any]:
        """Read-only access to current data."""
        return copy.deepcopy(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the store."""
        return copy.deepcopy(self._data.get(key, default))

    def update(self, updates: Dict[str, Any], source_state: str) -> None:
        """
        Apply updates and record snapshot.

        Args:
            updates: Key-value pairs to merge into store.
            source_state: The state_id that produced this update.
        """
        snapshot = StateStoreSnapshot(
            timestamp=datetime.utcnow(),
            source_state=source_state,
            data_before=copy.deepcopy(self._data),
            updates=copy.deepcopy(updates),
        )
        self._data.update(updates)
        self._history.append(snapshot)

    def extract(self, schema: Type[BaseModel]) -> BaseModel:
        """
        Extract typed context for a state from the store.
        Only fields defined in the schema are included.
        Raises ValidationError if required fields are missing.
        """
        return schema(**{
            k: self._data[k]
            for k in schema.model_fields
            if k in self._data
        })

    def to_dict(self) -> Dict[str, Any]:
        """Serialize current state for checkpointing."""
        return copy.deepcopy(self._data)

    @property
    def history(self) -> List["StateStoreSnapshot"]:
        """Full history of updates for audit."""
        return list(self._history)


@dataclass
class StateStoreSnapshot:
    """Immutable record of a state store update."""
    timestamp: datetime
    source_state: str
    data_before: Dict[str, Any]
    updates: Dict[str, Any]
```

### 3.3 Tool

Tools are plain Python functions decorated with metadata. Framework-agnostic — no LangChain or DSPy dependency required.

```python
from typing import Callable, Optional, Type, Any
from pydantic import BaseModel
from functools import wraps


@dataclass
class ToolDef:
    """Metadata for a tool."""
    name: str
    description: str
    parameters_schema: Type[BaseModel]    # Pydantic model for input validation
    return_schema: Type[BaseModel]        # Pydantic model for output typing
    function: Callable                     # The actual implementation
    is_async: bool = False


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    params: Optional[Type[BaseModel]] = None,
    returns: Optional[Type[BaseModel]] = None,
):
    """
    Decorator to register a function as a FlowAgent tool.

    Usage:
        class ValidateParams(BaseModel):
            data: dict
            strict: bool = True

        class ValidateResult(BaseModel):
            valid: bool
            errors: List[str] = []

        @tool(
            name="validate_schema",
            description="Validate data against the required schema",
            params=ValidateParams,
            returns=ValidateResult,
        )
        def validate_schema(data: dict, strict: bool = True) -> dict:
            # ... implementation
            return {"valid": True, "errors": []}
    """
    def decorator(fn: Callable) -> ToolDef:
        tool_name = name or fn.__name__
        tool_desc = description or fn.__doc__ or ""

        tool_def = ToolDef(
            name=tool_name,
            description=tool_desc,
            parameters_schema=params,
            return_schema=returns,
            function=fn,
            is_async=asyncio.iscoroutinefunction(fn),
        )
        # Preserve original function attributes
        tool_def.__wrapped__ = fn
        return tool_def

    return decorator
```

### 3.4 LLMAdapter

Abstract interface for LLM providers. Handles tool-calling protocol differences.

```python
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""
    id: str                            # Unique call ID (from LLM)
    tool_name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a tool call."""
    call_id: str                       # Matches ToolCall.id
    tool_name: str
    result: Dict[str, Any]
    success: bool
    error: Optional[str] = None


@dataclass
class LLMResponse:
    """Parsed response from an LLM call."""
    tool_calls: List[ToolCall]         # Empty if no tool calls
    text_content: Optional[str]        # Text response (present when no tool calls)
    transition: Optional[str]          # Extracted transition choice (from final response)
    reasoning: Optional[str]           # Why the agent chose this transition
    raw_response: Any                  # Provider-specific raw response


class LLMAdapter(ABC):
    """
    Abstract adapter for LLM providers.
    Implementations handle provider-specific tool-calling formats.
    """

    @abstractmethod
    def call(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],    # Conversation within this state
        tools: List[ToolDef],               # Scoped tools for current state
        available_transitions: Dict[str, str],  # target_id -> description
        temperature: float = 0.0,
    ) -> LLMResponse:
        """
        Make an LLM call with scoped tools and transition options.

        The adapter is responsible for:
        1. Converting ToolDef list to provider-specific tool format
        2. Including transition selection mechanism in the prompt/tools
        3. Parsing the response into LLMResponse format

        Transition mechanism options (adapter decides best approach per provider):
        - Include a special "choose_transition" tool
        - Use structured output with a "transition" field
        - Parse transition from the final text response
        """
        pass

    @abstractmethod
    async def acall(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: List[ToolDef],
        available_transitions: Dict[str, str],
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Async version of call."""
        pass
```

---

## 4. State Executor — The Core Loop

The State Executor runs the agent within a single state. This is where the tool-calling loop and transition decision happen.

### 4.1 Execution Flow Per State

```
┌─────────────────────────────────────────────────────────────┐
│  State Executor: execute(state_def, state_store)            │
│                                                             │
│  1. Extract context from state_store via input_schema       │
│  2. Build system prompt (state description + tool info      │
│     + available transitions with descriptions)              │
│  3. Initialize empty message list for this state            │
│                                                             │
│  ┌─── Tool-Calling Loop (max_tool_calls iterations) ──┐    │
│  │                                                     │    │
│  │  4. LLM call with messages + scoped tools           │    │
│  │      + available transitions                        │    │
│  │                                                     │    │
│  │  5. Parse response:                                 │    │
│  │     ├── Has tool_calls?                             │    │
│  │     │   ├── Validate tool names ∈ allowed set       │    │
│  │     │   ├── Execute each tool call                  │    │
│  │     │   ├── Append tool call + results to messages  │    │
│  │     │   ├── Record in audit log                     │    │
│  │     │   └── Continue loop                           │    │
│  │     │                                               │    │
│  │     └── No tool_calls (final response)?             │    │
│  │         ├── Extract transition choice               │    │
│  │         ├── Validate transition ∈ allowed set       │    │
│  │         ├── Extract output data per output_schema   │    │
│  │         ├── Update state_store                      │    │
│  │         └── Return (transition, output)             │    │
│  │                                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  6. If max_tool_calls exceeded without transition:          │
│     → Raise MaxToolCallsExceeded error                     │
│     → Workflow engine handles per error policy              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 StateExecutor Implementation

```python
@dataclass
class StateExecutionResult:
    """Result of executing a single state."""
    state_id: str
    transition: str                         # Chosen target state
    reasoning: Optional[str]                # Agent's reasoning for transition
    output_data: Dict[str, Any]             # Data to write to state store
    tool_calls_made: List[ToolCallRecord]   # Audit trail
    llm_calls_count: int                    # Number of LLM invocations in this state
    messages: List[Dict[str, Any]]          # Full message history within this state


@dataclass
class ToolCallRecord:
    """Audit record of a tool call within a state."""
    timestamp: datetime
    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    success: bool
    duration_ms: float


class StateExecutor:
    """
    Executes the agent within a single workflow state.
    Handles the tool-calling loop and transition decision.
    """

    def __init__(self, llm_adapter: LLMAdapter):
        self.llm = llm_adapter

    def execute(
        self,
        state_def: StateDef,
        state_store: StateStore,
        global_system_prompt: Optional[str] = None,
    ) -> StateExecutionResult:
        """
        Run the agent in a single state.

        For AGENTIC mode:
          - Constructs scoped context from state_store
          - Runs tool-calling loop until agent declares transition
          - Returns chosen transition + output data

        For CONVERSATIONAL mode:
          - Same as AGENTIC but messages may include user interaction
          - External caller must inject user messages between LLM calls

        For DETERMINISTIC mode:
          - Calls handler function directly
          - Uses transition_resolver to determine next state
          - No LLM involvement
        """

        if state_def.mode == StateMode.DETERMINISTIC:
            return self._execute_deterministic(state_def, state_store)

        # Build system prompt
        system_prompt = self._build_system_prompt(
            state_def, global_system_prompt
        )

        # Extract typed context
        context = {}
        if state_def.input_schema:
            context = state_store.extract(state_def.input_schema).model_dump()
        else:
            context = state_store.data

        # Build available transitions dict
        transitions = {t.target: t.description for t in state_def.transitions}

        # Initialize message list with context
        messages = [
            {"role": "user", "content": self._format_context(context)}
        ]

        tool_records = []
        llm_calls = 0

        # Tool-calling loop
        for step in range(state_def.max_tool_calls + 1):
            response = self.llm.call(
                system_prompt=system_prompt,
                messages=messages,
                tools=state_def.tools,
                available_transitions=transitions,
            )
            llm_calls += 1

            if response.tool_calls:
                # Agent wants to call tools — execute them
                for tc in response.tool_calls:
                    # ENFORCE: tool must be in allowed set
                    allowed_names = {t.name for t in state_def.tools}
                    if tc.tool_name not in allowed_names:
                        raise ToolNotAllowedError(
                            f"Tool '{tc.tool_name}' not allowed in "
                            f"state '{state_def.state_id}'. "
                            f"Allowed: {allowed_names}"
                        )

                    # Execute tool
                    start = time.time()
                    try:
                        tool_def = next(
                            t for t in state_def.tools
                            if t.name == tc.tool_name
                        )
                        result = tool_def.function(**tc.arguments)
                        success = True
                        error = None
                    except Exception as e:
                        result = {"error": str(e)}
                        success = False
                        error = str(e)
                    duration = (time.time() - start) * 1000

                    # Record for audit
                    tool_records.append(ToolCallRecord(
                        timestamp=datetime.utcnow(),
                        tool_name=tc.tool_name,
                        arguments=tc.arguments,
                        result=result,
                        success=success,
                        duration_ms=duration,
                    ))

                    # Append to messages for next LLM call
                    messages.append({
                        "role": "assistant",
                        "tool_calls": [{"id": tc.id, "name": tc.tool_name, "arguments": tc.arguments}]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    })
            else:
                # No tool calls — agent is declaring transition
                if response.transition is None:
                    raise NoTransitionDeclaredError(
                        f"Agent returned final response without "
                        f"declaring a transition in state '{state_def.state_id}'"
                    )

                # ENFORCE: transition must be in allowed set
                if response.transition not in transitions:
                    raise InvalidTransitionError(
                        f"Transition '{response.transition}' not allowed "
                        f"from state '{state_def.state_id}'. "
                        f"Allowed: {list(transitions.keys())}"
                    )

                # Extract output data
                output_data = {}
                if response.text_content:
                    output_data = self._extract_output(
                        response.text_content,
                        state_def.output_schema,
                    )

                return StateExecutionResult(
                    state_id=state_def.state_id,
                    transition=response.transition,
                    reasoning=response.reasoning,
                    output_data=output_data,
                    tool_calls_made=tool_records,
                    llm_calls_count=llm_calls,
                    messages=messages,
                )

        # Loop exhausted without transition
        raise MaxToolCallsExceeded(
            f"State '{state_def.state_id}' exceeded "
            f"max_tool_calls ({state_def.max_tool_calls}) "
            f"without declaring a transition"
        )

    def _execute_deterministic(
        self, state_def: StateDef, state_store: StateStore
    ) -> StateExecutionResult:
        """Execute a deterministic (no-LLM) state."""
        # Run handler
        output = {}
        if state_def.handler:
            output = state_def.handler(state_store.data)

        # Resolve transition
        if state_def.transition_resolver:
            transition = state_def.transition_resolver(state_store.data)
        else:
            # Single outgoing edge — auto-transition
            if len(state_def.transitions) == 1:
                transition = state_def.transitions[0].target
            else:
                raise ConfigurationError(
                    f"Deterministic state '{state_def.state_id}' has "
                    f"multiple transitions but no transition_resolver"
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

    def _build_system_prompt(
        self, state_def: StateDef, global_prompt: Optional[str]
    ) -> str:
        """
        Construct the system prompt for this state.

        Structure:
        1. Global system prompt (agent identity / persona)
        2. State-specific description (what this state does)
        3. Available transitions (with descriptions)
        4. Instructions for transition declaration

        The tools are NOT described in the system prompt — they are
        passed via the LLM's native tool-binding mechanism.
        """
        parts = []

        if global_prompt:
            parts.append(global_prompt)

        parts.append(f"## Current Task\n{state_def.description}")

        # Transition instructions
        transition_text = "\n".join(
            f"- \"{t.target}\": {t.description}"
            for t in state_def.transitions
        )
        parts.append(
            f"## Available Transitions\n"
            f"When you have completed your work in this step, you must "
            f"choose one of the following transitions based on your "
            f"findings and reasoning:\n{transition_text}\n\n"
            f"Do NOT choose a transition until you have gathered enough "
            f"information from your tools. When ready, respond with your "
            f"transition choice and reasoning."
        )

        return "\n\n".join(parts)
```

---

## 5. Workflow Engine — The Orchestrator

The Workflow Engine drives the entire workflow execution from start to terminal state.

### 5.1 Engine Implementation

```python
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
    status: str                          # "completed" | "failed" | "max_steps_exceeded"
    error: Optional[str] = None


class WorkflowEngine:
    """
    Drives workflow execution from initial to terminal state.
    Handles checkpointing, error recovery, and observability.
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        checkpoint_store: Optional["CheckpointStore"] = None,
        hooks: Optional[List["WorkflowHook"]] = None,
    ):
        self.executor = StateExecutor(llm_adapter)
        self.checkpoint_store = checkpoint_store
        self.hooks = hooks or []

    def run(
        self,
        workflow: WorkflowDefinition,
        initial_input: Dict[str, Any],
        run_id: Optional[str] = None,
        resume_from: Optional[str] = None,  # checkpoint ID to resume from
    ) -> WorkflowExecutionResult:
        """
        Execute a workflow from start to completion.

        Args:
            workflow: The workflow definition to execute.
            initial_input: Initial data for the state store.
            run_id: Unique run identifier (auto-generated if None).
            resume_from: Checkpoint ID to resume from (for recovery).

        Returns:
            WorkflowExecutionResult with full trace and final state.
        """
        # Validate workflow
        errors = workflow.validate()
        if errors:
            raise WorkflowValidationError(errors)

        # Initialize
        run_id = run_id or str(uuid.uuid4())
        started_at = datetime.utcnow()
        trace: List[StateExecutionResult] = []

        # Resume from checkpoint or start fresh
        if resume_from and self.checkpoint_store:
            checkpoint = self.checkpoint_store.load(resume_from)
            state_store = StateStore(checkpoint.store_data)
            current_state = checkpoint.current_state
            trace = checkpoint.trace
        else:
            state_store = StateStore(initial_input)
            current_state = workflow.initial_state

        # Notify hooks
        for hook in self.hooks:
            hook.on_workflow_start(run_id, workflow, state_store)

        step_count = 0
        status = "completed"
        error = None

        try:
            while current_state not in workflow.terminal_states:
                # Safety bound
                if step_count >= workflow.max_total_steps:
                    status = "max_steps_exceeded"
                    break

                state_def = workflow.states[current_state]

                # Notify hooks
                for hook in self.hooks:
                    hook.on_state_enter(run_id, current_state, state_store)

                # Execute state
                result = self.executor.execute(
                    state_def=state_def,
                    state_store=state_store,
                    global_system_prompt=workflow.global_system_prompt,
                )

                # Update state store with outputs
                if result.output_data:
                    state_store.update(result.output_data, source_state=current_state)

                # Record trace
                trace.append(result)

                # Notify hooks
                for hook in self.hooks:
                    hook.on_state_exit(
                        run_id, current_state, result.transition, state_store
                    )

                # Checkpoint
                if self.checkpoint_store:
                    self.checkpoint_store.save(Checkpoint(
                        run_id=run_id,
                        current_state=result.transition,
                        store_data=state_store.to_dict(),
                        trace=trace,
                        timestamp=datetime.utcnow(),
                    ))

                # Transition
                current_state = result.transition
                step_count += 1

        except FlowAgentError as e:
            status = "failed"
            error = str(e)
            for hook in self.hooks:
                hook.on_error(run_id, current_state, e)

        completed_at = datetime.utcnow()

        execution_result = WorkflowExecutionResult(
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

        # Notify hooks
        for hook in self.hooks:
            hook.on_workflow_complete(run_id, execution_result)

        return execution_result
```

---

## 6. Workflow Validation

### 6.1 Graph Validation Rules

The `WorkflowDefinition.validate()` method must enforce these rules:

```python
def validate(self) -> List[str]:
    """
    Validate workflow graph integrity.
    Returns list of error strings. Empty list = valid.
    """
    errors = []

    # 1. Initial state must exist
    if self.initial_state not in self.states:
        errors.append(
            f"Initial state '{self.initial_state}' not defined in states"
        )

    # 2. All terminal states must exist
    for ts in self.terminal_states:
        if ts not in self.states:
            errors.append(f"Terminal state '{ts}' not defined in states")

    # 3. Terminal states must have no outgoing transitions
    for ts in self.terminal_states:
        if ts in self.states and self.states[ts].transitions:
            errors.append(
                f"Terminal state '{ts}' has outgoing transitions"
            )

    # 4. All transition targets must reference existing states
    for state_id, state_def in self.states.items():
        for t in state_def.transitions:
            if t.target not in self.states:
                errors.append(
                    f"State '{state_id}' has transition to "
                    f"non-existent state '{t.target}'"
                )

    # 5. Non-terminal states must have at least one transition
    for state_id, state_def in self.states.items():
        if state_id not in self.terminal_states:
            if not state_def.transitions:
                errors.append(
                    f"Non-terminal state '{state_id}' has no transitions"
                )

    # 6. All states must be reachable from initial state (BFS/DFS)
    reachable = self._find_reachable_states(self.initial_state)
    for state_id in self.states:
        if state_id not in reachable:
            errors.append(
                f"State '{state_id}' is not reachable from "
                f"initial state '{self.initial_state}'"
            )

    # 7. At least one terminal state must be reachable
    if not any(ts in reachable for ts in self.terminal_states):
        errors.append("No terminal state is reachable from initial state")

    # 8. Deterministic states must have handler or single transition
    for state_id, state_def in self.states.items():
        if state_def.mode == StateMode.DETERMINISTIC:
            if not state_def.handler and not state_def.transition_resolver:
                if len(state_def.transitions) > 1:
                    errors.append(
                        f"Deterministic state '{state_id}' has "
                        f"multiple transitions but no handler/resolver"
                    )

    # 9. Agentic states must have at least one tool
    for state_id, state_def in self.states.items():
        if state_def.mode == StateMode.AGENTIC:
            if not state_def.tools:
                errors.append(
                    f"Agentic state '{state_id}' has no tools"
                )

    return errors
```

---

## 7. Transition Mechanism — LLM Integration

### 7.1 How the Agent Declares Transitions

The transition declaration is handled through the LLM's native capabilities. The recommended approach uses a **dedicated transition tool** that the agent calls when done working.

```python
# The framework auto-generates this tool for every agentic state
class TransitionParams(BaseModel):
    transition: str        # Must be one of the allowed transition targets
    reasoning: str         # Why this transition was chosen


def _make_transition_tool(transitions: Dict[str, str]) -> ToolDef:
    """
    Auto-generate a 'declare_transition' tool for the current state.

    This tool is always the last tool called. When the agent calls it,
    the state executor treats it as the terminal action and exits the
    tool-calling loop.
    """
    transition_descriptions = "\n".join(
        f"  - '{target}': {desc}"
        for target, desc in transitions.items()
    )

    return ToolDef(
        name="declare_transition",
        description=(
            f"Call this tool when you have completed your work in this "
            f"step and are ready to move to the next step. Choose one "
            f"of the available transitions based on your findings.\n"
            f"Available transitions:\n{transition_descriptions}"
        ),
        parameters_schema=TransitionParams,
        return_schema=BaseModel,  # Not used — intercepted by executor
        function=lambda **kwargs: kwargs,  # Passthrough
    )
```

### 7.2 Tool Injection Strategy

When the StateExecutor prepares tools for an LLM call, it:

1. Takes the state's defined tools (work tools)
2. Appends the auto-generated `declare_transition` tool
3. Passes all tools to the LLM via native tool binding

```python
def _prepare_tools(self, state_def: StateDef) -> List[ToolDef]:
    """Prepare the full tool set for an LLM call."""
    work_tools = list(state_def.tools)

    if state_def.transitions:
        transitions = {t.target: t.description for t in state_def.transitions}
        nav_tool = self._make_transition_tool(transitions)
        work_tools.append(nav_tool)

    return work_tools
```

### 7.3 Detecting Transition in the Loop

The StateExecutor's tool-calling loop checks if `declare_transition` was called:

```python
# Inside the tool-calling loop
for tc in response.tool_calls:
    if tc.tool_name == "declare_transition":
        # This is the transition declaration — validate and exit
        transition = tc.arguments["transition"]
        reasoning = tc.arguments.get("reasoning", "")

        allowed = {t.target for t in state_def.transitions}
        if transition not in allowed:
            raise InvalidTransitionError(...)

        return StateExecutionResult(
            transition=transition,
            reasoning=reasoning,
            ...
        )
    else:
        # Regular tool call — execute normally
        ...
```

This approach is clean because:

- The agent uses the **same mechanism** (tool calling) for both work and navigation
- The LLM sees `declare_transition` as just another tool with typed parameters
- The framework intercepts it specially, but the LLM doesn't know that
- Works with every LLM provider that supports function calling

---

## 8. Checkpoint and Persistence

### 8.1 Checkpoint Store Interface

```python
@dataclass
class Checkpoint:
    """A snapshot of workflow state at a point in time."""
    run_id: str
    current_state: str                      # State about to be executed
    store_data: Dict[str, Any]              # Full state store snapshot
    trace: List[StateExecutionResult]       # Execution history so far
    timestamp: datetime


class CheckpointStore(ABC):
    """
    Abstract interface for checkpoint persistence.
    Implementations can target different backends.
    """

    @abstractmethod
    def save(self, checkpoint: Checkpoint) -> str:
        """Save checkpoint, return checkpoint ID."""
        pass

    @abstractmethod
    def load(self, checkpoint_id: str) -> Checkpoint:
        """Load checkpoint by ID."""
        pass

    @abstractmethod
    def load_latest(self, run_id: str) -> Optional[Checkpoint]:
        """Load the latest checkpoint for a run."""
        pass

    @abstractmethod
    def list_checkpoints(self, run_id: str) -> List[Checkpoint]:
        """List all checkpoints for a run (for debugging/replay)."""
        pass
```

### 8.2 Built-in Implementations

```
CheckpointStore (ABC)
├── InMemoryCheckpointStore        # For testing / short-lived workflows
├── FileCheckpointStore            # JSON files on disk
├── RedisCheckpointStore           # Redis (for fast, ephemeral checkpoints)
└── SQLCheckpointStore             # PostgreSQL / SQLite (for durable storage)
```

---

## 9. Hooks and Observability

### 9.1 Hook Interface

```python
class WorkflowHook(ABC):
    """
    Lifecycle hooks for observability, logging, and custom behavior.
    All methods are optional (default no-op).
    """

    def on_workflow_start(
        self, run_id: str, workflow: WorkflowDefinition, state_store: StateStore
    ) -> None: pass

    def on_state_enter(
        self, run_id: str, state_id: str, state_store: StateStore
    ) -> None: pass

    def on_tool_call(
        self, run_id: str, state_id: str, tool_call: ToolCall
    ) -> None: pass

    def on_tool_result(
        self, run_id: str, state_id: str, tool_result: ToolResult
    ) -> None: pass

    def on_llm_call(
        self, run_id: str, state_id: str, messages: List, response: LLMResponse
    ) -> None: pass

    def on_transition(
        self, run_id: str, from_state: str, to_state: str, reasoning: str
    ) -> None: pass

    def on_state_exit(
        self, run_id: str, state_id: str, transition: str, state_store: StateStore
    ) -> None: pass

    def on_error(
        self, run_id: str, state_id: str, error: Exception
    ) -> None: pass

    def on_workflow_complete(
        self, run_id: str, result: WorkflowExecutionResult
    ) -> None: pass
```

### 9.2 Built-in Hooks

```
WorkflowHook (ABC)
├── ConsoleLogHook               # Pretty-print execution to stdout
├── JSONFileLogHook              # Write structured logs to JSON file
├── AuditTrailHook               # Full audit trail for compliance
├── MetricsHook                  # Collect timing, token usage, tool call counts
└── OpenTelemetryHook            # Export traces to OTEL-compatible backends
```

---

## 10. LLM Adapter Implementations

### 10.1 Supported Providers (MVP)

```
LLMAdapter (ABC)
├── AnthropicAdapter             # Claude via Anthropic API
├── OpenAIAdapter                # GPT-4o/etc via OpenAI API
├── AzureOpenAIAdapter           # Azure OpenAI deployments
└── LiteLLMAdapter               # Any provider via LiteLLM proxy
```

### 10.2 Adapter Responsibilities

Each adapter must handle:

1. **Tool format conversion** — Convert `ToolDef` list to provider-specific format (OpenAI function-calling schema, Anthropic tool-use schema, etc.)

2. **Transition tool handling** — Include `declare_transition` alongside work tools in the native tool format

3. **Response parsing** — Parse provider response into unified `LLMResponse` (detect tool calls vs final text vs transition declaration)

4. **Error handling** — Handle rate limits, context window overflow, malformed responses

### 10.3 AnthropicAdapter Example Sketch

```python
class AnthropicAdapter(LLMAdapter):
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str = None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def call(self, system_prompt, messages, tools, available_transitions, temperature=0.0):
        # Convert ToolDef list to Anthropic tool format
        anthropic_tools = [self._convert_tool(t) for t in tools]

        # Convert messages to Anthropic format
        anthropic_messages = self._convert_messages(messages)

        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=anthropic_messages,
            tools=anthropic_tools,
            max_tokens=4096,
            temperature=temperature,
        )

        return self._parse_response(response)
```

---

## 11. Error Handling

### 11.1 Error Hierarchy

```
FlowAgentError
├── WorkflowValidationError        # Graph integrity issues
├── StateExecutionError
│   ├── ToolNotAllowedError        # Agent called a tool not in scope
│   ├── InvalidTransitionError     # Agent chose invalid transition
│   ├── NoTransitionDeclaredError  # Agent exhausted text without transition
│   ├── MaxToolCallsExceeded       # Tool-calling loop exceeded bound
│   └── ToolExecutionError         # Tool raised an exception
├── LLMAdapterError
│   ├── RateLimitError
│   ├── ContextOverflowError
│   └── MalformedResponseError
├── CheckpointError
│   ├── CheckpointSaveError
│   └── CheckpointLoadError
└── ConfigurationError             # Misconfigured state/workflow
```

### 11.2 Error Recovery Strategy

```python
@dataclass
class ErrorPolicy:
    """Configure error handling behavior per state or globally."""

    # On ToolExecutionError
    tool_error_action: str = "feed_to_llm"
    # Options:
    #   "feed_to_llm" — Send error as tool result, let agent decide
    #   "retry"       — Retry the tool call (up to max_retries)
    #   "fail"        — Fail the state immediately

    max_tool_retries: int = 2

    # On InvalidTransitionError
    invalid_transition_action: str = "retry_llm"
    # Options:
    #   "retry_llm" — Ask LLM again with error feedback
    #   "fail"      — Fail the state

    max_transition_retries: int = 2

    # On MaxToolCallsExceeded
    max_steps_action: str = "fail"
    # Options:
    #   "fail"              — Fail the state
    #   "force_transition"  — Ask LLM one final time for transition only
```

---

## 12. User Interaction (Conversational States)

### 12.1 Human-in-the-Loop Pattern

Some states require user input mid-workflow (e.g., clarification, approval). These use `StateMode.CONVERSATIONAL`.

```python
class WorkflowEngine:
    def run_interactive(
        self, workflow, initial_input, user_input_handler
    ):
        """
        Run workflow with user interaction support.

        user_input_handler: Callable that blocks until user provides input.
        Called when a CONVERSATIONAL state needs user input.
        """
        ...
        while current_state not in workflow.terminal_states:
            state_def = workflow.states[current_state]

            if state_def.mode == StateMode.CONVERSATIONAL:
                result = self._execute_conversational(
                    state_def, state_store, user_input_handler
                )
            else:
                result = self.executor.execute(state_def, state_store)
            ...
```

### 12.2 Conversational State Execution

Within a conversational state, the agent can:

- Call tools (same as agentic mode)
- Generate a text response to the user
- Wait for user input
- Continue the conversation
- Declare a transition when done

The message history within the conversational state is maintained for the duration of that state only. It is discarded on transition.

---

## 13. Complete Usage Example

### 13.1 Invoice Processing Workflow

```python
from flowagent import (
    WorkflowDefinition, StateDef, TransitionDef, StateMode,
    tool, WorkflowEngine, AnthropicAdapter,
)
from pydantic import BaseModel
from typing import List, Optional


# --- Schemas ---
class InvoiceData(BaseModel):
    raw_text: str

class ExtractedFields(BaseModel):
    vendor: Optional[str] = None
    amount: Optional[float] = None
    date: Optional[str] = None
    line_items: List[dict] = []

class ValidationResult(BaseModel):
    extracted: ExtractedFields
    valid: Optional[bool] = None
    errors: List[str] = []

class TransformResult(BaseModel):
    extracted: ExtractedFields
    transformed_record: Optional[dict] = None


# --- Tools ---
@tool(
    name="extract_fields",
    description="Extract structured fields from raw invoice text",
    params=InvoiceData,
    returns=ExtractedFields,
)
def extract_fields(raw_text: str) -> dict:
    # Implementation: OCR / regex / ML extraction
    return {"vendor": "Acme Corp", "amount": 5200.00, "date": "2026-02-15", "line_items": []}

@tool(
    name="validate_schema",
    description="Validate extracted invoice data against business rules",
    params=ExtractedFields,
    returns=ValidationResult,
)
def validate_schema(vendor: str, amount: float, date: str, **kwargs) -> dict:
    errors = []
    if amount > 10000:
        errors.append("Amount exceeds auto-approval threshold")
    if not vendor:
        errors.append("Vendor name is required")
    return {"valid": len(errors) == 0, "errors": errors}

@tool(
    name="ask_user",
    description="Ask the user a clarifying question",
    params=...,
    returns=...,
)
def ask_user(question: str) -> dict:
    # In production: sends to UI, waits for response
    return {"user_response": "Yes, that amount is correct"}

@tool(
    name="transform_to_erp",
    description="Transform validated invoice into ERP-compatible record",
    params=ExtractedFields,
    returns=TransformResult,
)
def transform_to_erp(vendor: str, amount: float, date: str, **kwargs) -> dict:
    return {"record": {"erp_vendor_id": "V-001", "amount_usd": amount, "posting_date": date}}

@tool(
    name="commit_to_erp",
    description="Commit the transformed record to the ERP system",
    params=TransformResult,
    returns=...,
)
def commit_to_erp(transformed_record: dict) -> dict:
    return {"transaction_id": "TXN-2026-0042", "status": "committed"}


# --- Workflow Definition ---
invoice_workflow = WorkflowDefinition(
    name="invoice_processing",
    version="1.0.0",
    description="Process incoming invoices from extraction to ERP commitment",

    global_system_prompt=(
        "You are an invoice processing agent. You process invoices "
        "step-by-step: extract data, validate it, get clarification "
        "if needed, transform it, and commit to the ERP system. "
        "Be thorough and careful with financial data."
    ),

    initial_state="extract",
    terminal_states=["done", "failed"],

    states={
        "extract": StateDef(
            state_id="extract",
            description="Extract structured fields from the raw invoice text.",
            mode=StateMode.AGENTIC,
            tools=[extract_fields],
            input_schema=InvoiceData,
            output_schema=ExtractedFields,
            transitions=[
                TransitionDef("validate", "Fields extracted successfully, ready to validate"),
                TransitionDef("clarify", "Invoice is unreadable or ambiguous, need user help"),
            ],
        ),

        "validate": StateDef(
            state_id="validate",
            description="Validate the extracted invoice data against business rules.",
            mode=StateMode.AGENTIC,
            tools=[validate_schema],
            input_schema=ExtractedFields,
            output_schema=ValidationResult,
            transitions=[
                TransitionDef("transform", "Validation passed, proceed to transformation"),
                TransitionDef("clarify", "Validation issues require user clarification"),
                TransitionDef("extract", "Data is too corrupted, re-extract from source"),
            ],
        ),

        "clarify": StateDef(
            state_id="clarify",
            description="Ask the user for clarification on ambiguous or missing data.",
            mode=StateMode.CONVERSATIONAL,
            tools=[ask_user],
            transitions=[
                TransitionDef("validate", "Got clarification, re-validate with updated data"),
                TransitionDef("extract", "User provided new source, re-extract"),
                TransitionDef("failed", "User wants to abort processing"),
            ],
        ),

        "transform": StateDef(
            state_id="transform",
            description="Transform validated invoice data into ERP-compatible format.",
            mode=StateMode.AGENTIC,
            tools=[transform_to_erp],
            input_schema=ValidationResult,
            output_schema=TransformResult,
            transitions=[
                TransitionDef("commit", "Transformation successful, ready to commit"),
                TransitionDef("validate", "Transformation revealed data issues, revalidate"),
            ],
        ),

        "commit": StateDef(
            state_id="commit",
            description="Commit the transformed record to the ERP system.",
            mode=StateMode.AGENTIC,
            tools=[commit_to_erp],
            input_schema=TransformResult,
            transitions=[
                TransitionDef("done", "Successfully committed to ERP"),
                TransitionDef("transform", "Commit failed, retry transformation"),
            ],
        ),

        "done": StateDef(
            state_id="done",
            description="Invoice processing completed successfully.",
            mode=StateMode.DETERMINISTIC,
            transitions=[],
        ),

        "failed": StateDef(
            state_id="failed",
            description="Invoice processing failed or was aborted.",
            mode=StateMode.DETERMINISTIC,
            transitions=[],
        ),
    },
)


# --- Run ---
engine = WorkflowEngine(
    llm_adapter=AnthropicAdapter(model="claude-sonnet-4-20250514"),
)

result = engine.run(
    workflow=invoice_workflow,
    initial_input={"raw_text": "Invoice #1042 from Acme Corp..."},
)

print(f"Status: {result.status}")
print(f"Final state: {result.final_state}")
print(f"LLM calls: {result.total_llm_calls}")
print(f"Tool calls: {result.total_tool_calls}")
```

---

## 14. Project Structure

```
flowagent/
├── pyproject.toml
├── README.md
├── LICENSE
│
├── flowagent/
│   ├── __init__.py                    # Public API exports
│   ├── types.py                       # Core dataclasses and enums
│   │                                  # (StateDef, TransitionDef, WorkflowDefinition,
│   │                                  #  StateMode, ToolDef, ToolCall, ToolResult, etc.)
│   │
│   ├── state_store.py                 # StateStore + StateStoreSnapshot
│   │
│   ├── tool.py                        # @tool decorator and ToolDef
│   │
│   ├── state_executor.py              # StateExecutor (tool-calling loop per state)
│   │
│   ├── engine.py                      # WorkflowEngine (orchestrator)
│   │
│   ├── validation.py                  # WorkflowDefinition.validate() logic
│   │
│   ├── prompt_builder.py              # System prompt construction per state
│   │
│   ├── transition.py                  # declare_transition tool generation + interception
│   │
│   ├── errors.py                      # Error hierarchy
│   │
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py                    # LLMAdapter ABC
│   │   ├── anthropic_adapter.py       # Anthropic Claude
│   │   ├── openai_adapter.py          # OpenAI GPT
│   │   ├── azure_adapter.py           # Azure OpenAI
│   │   └── litellm_adapter.py         # LiteLLM (any provider)
│   │
│   ├── checkpoints/
│   │   ├── __init__.py
│   │   ├── base.py                    # CheckpointStore ABC
│   │   ├── memory.py                  # InMemoryCheckpointStore
│   │   ├── file.py                    # FileCheckpointStore (JSON)
│   │   ├── redis.py                   # RedisCheckpointStore
│   │   └── sql.py                     # SQLCheckpointStore
│   │
│   ├── hooks/
│   │   ├── __init__.py
│   │   ├── base.py                    # WorkflowHook ABC
│   │   ├── console.py                 # ConsoleLogHook
│   │   ├── json_log.py               # JSONFileLogHook
│   │   ├── audit.py                   # AuditTrailHook
│   │   └── metrics.py                 # MetricsHook
│   │
│   └── utils/
│       ├── __init__.py
│       ├── graph_viz.py               # Workflow graph visualization (Mermaid / Graphviz)
│       └── replay.py                  # Replay execution from checkpoint for debugging
│
├── tests/
│   ├── test_state_store.py
│   ├── test_state_executor.py
│   ├── test_engine.py
│   ├── test_validation.py
│   ├── test_transition.py
│   ├── test_adapters/
│   │   ├── test_anthropic.py
│   │   └── test_openai.py
│   ├── test_checkpoints/
│   │   └── ...
│   └── test_examples/
│       └── test_invoice_workflow.py
│
└── examples/
    ├── invoice_processing.py          # Full invoice workflow (Section 13)
    ├── customer_support.py            # Support ticket triage + resolution
    └── data_pipeline.py               # ETL with validation + retry
```

---

## 15. Dependencies

### Core (minimal)

```toml
[project]
dependencies = [
    "pydantic>=2.0",       # Schema validation and typing
]
```

### Optional (extras)

```toml
[project.optional-dependencies]
anthropic = ["anthropic>=0.40"]
openai = ["openai>=1.50"]
azure = ["openai>=1.50"]
litellm = ["litellm>=1.50"]
redis = ["redis>=5.0"]
sql = ["sqlalchemy>=2.0", "alembic>=1.13"]
viz = ["graphviz>=0.20"]
all = ["flowagent[anthropic,openai,redis,sql,viz]"]
```

---

## 16. Implementation Plan

### Phase 1 — Core Runtime (Week 1-2)

- [ ] `types.py` — All core dataclasses and enums
- [ ] `errors.py` — Error hierarchy
- [ ] `tool.py` — `@tool` decorator and `ToolDef`
- [ ] `state_store.py` — `StateStore` with snapshot history
- [ ] `transition.py` — `declare_transition` tool generation
- [ ] `prompt_builder.py` — System prompt construction
- [ ] `validation.py` — Graph validation rules
- [ ] `state_executor.py` — Full tool-calling loop with transition detection
- [ ] `engine.py` — `WorkflowEngine` with basic run loop
- [ ] Unit tests for all above

### Phase 2 — LLM Adapters (Week 2-3)

- [ ] `adapters/base.py` — `LLMAdapter` ABC
- [ ] `adapters/anthropic_adapter.py` — Claude adapter
- [ ] `adapters/openai_adapter.py` — OpenAI adapter
- [ ] Integration tests with real LLM calls
- [ ] Invoice processing example (Section 13) working end-to-end

### Phase 3 — Persistence and Observability (Week 3-4)

- [ ] `checkpoints/memory.py` — InMemoryCheckpointStore
- [ ] `checkpoints/file.py` — FileCheckpointStore
- [ ] `hooks/console.py` — ConsoleLogHook
- [ ] `hooks/json_log.py` — JSONFileLogHook
- [ ] Resume-from-checkpoint in WorkflowEngine
- [ ] `utils/graph_viz.py` — Mermaid diagram generation

### Phase 4 — Conversational States and Polish (Week 4-5)

- [ ] Conversational state execution mode
- [ ] Human-in-the-loop interaction pattern
- [ ] Error recovery policies
- [ ] `utils/replay.py` — Replay from checkpoint
- [ ] Additional examples (customer support, data pipeline)
- [ ] Documentation and README

### Phase 5 — Production Hardening (Week 5-6)

- [ ] `checkpoints/redis.py` and `checkpoints/sql.py`
- [ ] `hooks/audit.py` and `hooks/metrics.py`
- [ ] Async support (`acall` throughout)
- [ ] `adapters/azure_adapter.py` and `adapters/litellm_adapter.py`
- [ ] Performance benchmarks
- [ ] PyPI packaging

---

## 17. Design Decisions Summary

| Decision              | Choice                          | Rationale                                                                  |
| --------------------- | ------------------------------- | -------------------------------------------------------------------------- |
| State management      | Checkpoint snapshots            | Temporal-style event sourcing is overkill; snapshot + resume is sufficient |
| Inter-state context   | State store (not chat history)  | DSPy-style stateless calls; each state gets fresh typed context            |
| Intra-state execution | Standard tool-calling loop      | Agent needs tool results before deciding transition                        |
| Transition mechanism  | `declare_transition` tool       | Unified mechanism; works with every LLM's tool-calling API                 |
| Transition decisions  | Agent-driven, graph-constrained | Workflow defines allowed paths; agent navigates based on reasoning         |
| LLM integration       | Provider-agnostic adapter       | No vendor lock-in; support Anthropic, OpenAI, Azure, local models          |
| Persistence           | Pluggable checkpoint store      | Simple default (file/memory); production options (Redis, SQL)              |
| Observability         | Hook-based plugin system        | Composable; don't bloat core with logging/metrics/tracing                  |
| Dependencies          | Minimal core (only pydantic)    | LLM SDKs, Redis, SQL are optional extras                                   |
| Non-agentic states    | DETERMINISTIC mode              | Not every state needs LLM; pure functions for routing/checks               |

---

## 18. Open Questions for Implementation

1. **Streaming support** — Should the framework support streaming LLM responses within a state? If so, only for conversational states, or for all? (Recommendation: add in Phase 4+, conversational only.)

2. **Parallel tool execution** — When an LLM returns multiple tool calls in one response, execute sequentially or in parallel? (Recommendation: parallel by default with sequential option.)

3. **State re-entry limits** — Should the framework enforce max re-entries per state to prevent infinite loops on back-edges? (Recommendation: yes, configurable `max_visits` per state, default 3.)

4. **Workflow composition** — Should workflows be composable (a state can invoke a sub-workflow)? (Recommendation: defer to Phase 5+, implement as a special DETERMINISTIC state that runs a child WorkflowEngine.)

5. **Framework name** — "FlowAgent" is a working title. Alternatives: "StateAgent", "PathAgent", "GatedAgent", "NavAgent". Choose before PyPI packaging.
