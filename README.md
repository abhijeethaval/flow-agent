# FlowAgent

**State-Gated Agentic Workflow Framework**

FlowAgent is a Python framework for building deterministic workflow applications powered by a single LLM agent. The workflow is defined as a directed graph of states; each state exposes a scoped set of tools to the LLM; the agent decides which available transition to take based on tool results, user input, and its own reasoning.

> The workflow graph constrains **where** the agent can go. The agent decides **which path** to take through the graph.

---

## Key Principles

1. **Single agent, single identity** — One LLM, one system prompt template, one orchestrator. No multi-agent persona switching.
2. **Stateless between states** — Each state gets a fresh, constructed context from the state store. No accumulated chat history across states.
3. **Agentic within states** — Within a state, the agent runs a standard tool-calling loop until it declares a transition.
4. **Deterministic graph, agentic navigation** — Graph topology is immutable; the agent chooses among predefined outgoing edges (including back-edges).
5. **Checkpoint-based persistence** — Snapshot state between steps. Resume from the last checkpoint on failure.
6. **LLM-provider agnostic** — Any LLM with function-calling support (Anthropic, OpenAI, Azure, LiteLLM).

---

## Installation

Requires Python 3.11+.

```bash
pip install flowagent                 # core (pydantic only)
pip install "flowagent[anthropic]"    # + Anthropic Claude
pip install "flowagent[openai]"       # + OpenAI
pip install "flowagent[redis]"        # + Redis checkpoint store
pip install "flowagent[sql]"          # + SQLAlchemy checkpoint store
pip install "flowagent[viz]"          # + Graphviz rendering
pip install "flowagent[all]"          # everything
```

---

## Quick Start

```python
from flowagent import (
    WorkflowDefinition, StateDef, TransitionDef, StateMode,
    tool, WorkflowEngine,
)
from flowagent.adapters import AnthropicAdapter
from pydantic import BaseModel


class InvoiceData(BaseModel):
    raw_text: str

class ExtractedFields(BaseModel):
    vendor: str | None = None
    amount: float | None = None
    date: str | None = None


@tool(
    name="extract_fields",
    description="Extract structured fields from raw invoice text",
    params=InvoiceData,
    returns=ExtractedFields,
)
def extract_fields(raw_text: str) -> dict:
    return {"vendor": "Acme Corp", "amount": 5200.00, "date": "2026-02-15"}


workflow = WorkflowDefinition(
    name="invoice_processing",
    version="1.0.0",
    description="Extract, validate, and commit invoices",
    initial_state="extract",
    terminal_states=["done", "failed"],
    states={
        "extract": StateDef(
            state_id="extract",
            description="Extract structured fields from the raw invoice.",
            mode=StateMode.AGENTIC,
            tools=[extract_fields],
            input_schema=InvoiceData,
            output_schema=ExtractedFields,
            transitions=[
                TransitionDef("done", "Extraction succeeded"),
                TransitionDef("failed", "Invoice unreadable"),
            ],
        ),
        "done":   StateDef(state_id="done",   description="Completed.", mode=StateMode.DETERMINISTIC, transitions=[]),
        "failed": StateDef(state_id="failed", description="Aborted.",   mode=StateMode.DETERMINISTIC, transitions=[]),
    },
)

engine = WorkflowEngine(llm_adapter=AnthropicAdapter(model="claude-sonnet-4-6"))
result = engine.run(workflow, initial_input={"raw_text": "Invoice #1042..."})

print(result.status, result.final_state, result.total_llm_calls, result.total_tool_calls)
```

See `examples/invoice_processing.py` for the full end-to-end workflow (extract → validate → clarify → transform → commit).

---

## Core Concepts

### WorkflowDefinition

Immutable graph of states plus global configuration. Validated at run time — `validate()` enforces reachability, non-terminal states must have transitions, terminal states must not, agentic states must define at least one tool, etc.

### StateDef and StateMode

Each state picks one of three modes:

| Mode              | Behavior                                                                 |
| ----------------- | ------------------------------------------------------------------------ |
| `AGENTIC`         | Tool-calling loop; agent picks a transition via `declare_transition`.    |
| `CONVERSATIONAL`  | Same, but external caller can inject user messages between LLM calls.    |
| `DETERMINISTIC`   | No LLM. Pure `handler` plus `transition_resolver` (or single outgoing edge). |

A state declares `input_schema` / `output_schema` (pydantic models) to type the data it reads from and writes to the state store.

### StateStore

The only data-passing mechanism between states. Each `update()` appends an immutable snapshot so execution is fully auditable. A state extracts only the fields declared in its `input_schema`.

### Tools

Plain Python functions with a `@tool` decorator. Parameters and return values are pydantic models — no LangChain / DSPy dependency required.

```python
@tool(name="validate_schema", description="...", params=ExtractedFields, returns=ValidationResult)
def validate_schema(vendor: str, amount: float, date: str, **_) -> dict:
    ...
```

### Transitions

Every agentic state auto-generates a `declare_transition(transition, reasoning)` tool alongside its work tools. The executor intercepts that call, validates the choice is in the state's allowed set, and exits the tool-calling loop. Tools and transitions that aren't in scope raise `ToolNotAllowedError` / `InvalidTransitionError`.

### StateExecutor

Runs a single state. For each agentic step it:

1. Builds a system prompt (global + state description + available transitions).
2. Extracts typed context from the state store.
3. Loops up to `max_tool_calls`, executing tools and feeding results back to the LLM.
4. Returns when the agent calls `declare_transition`, or raises `MaxToolCallsExceeded`.

### WorkflowEngine

Drives the whole graph from initial to terminal state. Validates the workflow, manages the state store, calls hooks, writes checkpoints, and enforces `max_total_steps`. Supports resume-from-checkpoint and (in conversational mode) a user-input handler.

### Checkpoints

Pluggable via the `CheckpointStore` ABC. Built-in implementations: in-memory, JSON file, Redis, SQL.

### Hooks

`WorkflowHook` has optional lifecycle methods (`on_workflow_start`, `on_state_enter`, `on_tool_call`, `on_llm_call`, `on_transition`, `on_state_exit`, `on_error`, `on_workflow_complete`). Built-in hooks cover console logging, JSON logs, audit trails, metrics, and OpenTelemetry.

### LLM Adapters

`LLMAdapter` normalizes tool-calling across providers. Built-in: Anthropic, OpenAI, Azure OpenAI, LiteLLM. Each adapter converts `ToolDef` to the provider's format, includes `declare_transition` alongside work tools, and parses the response into a unified `LLMResponse`.

---

## Errors

```
FlowAgentError
├── WorkflowValidationError
├── StateExecutionError
│   ├── ToolNotAllowedError
│   ├── InvalidTransitionError
│   ├── NoTransitionDeclaredError
│   ├── MaxToolCallsExceeded
│   └── ToolExecutionError
├── LLMAdapterError
│   ├── RateLimitError
│   ├── ContextOverflowError
│   └── MalformedResponseError
├── CheckpointError
│   ├── CheckpointSaveError
│   └── CheckpointLoadError
└── ConfigurationError
```

`ErrorPolicy` configures per-state or global recovery: `feed_to_llm` / `retry` / `fail` for tool errors, retry-LLM for invalid transitions, and fail-or-force-transition on `MaxToolCallsExceeded`.

---

## Project Layout

```
flowagent/
├── types.py              # StateDef, TransitionDef, WorkflowDefinition, StateMode, ...
├── state_store.py        # StateStore + snapshots
├── tool.py               # @tool decorator, ToolDef
├── state_executor.py     # per-state tool-calling loop
├── engine.py             # WorkflowEngine orchestrator
├── validation.py         # graph validation rules
├── prompt_builder.py     # system prompt assembly
├── transition.py         # declare_transition generation + interception
├── errors.py             # error hierarchy
├── adapters/             # anthropic, openai, azure, litellm
├── checkpoints/          # memory, file, redis, sql
├── hooks/                # console, json_log, audit, metrics
└── utils/                # graph_viz, replay
```

---

## Status

Draft — ready for implementation. See `flow_agent_spec.md` for the full specification (v1.0) including architecture diagrams, execution semantics, validation rules, and the phased implementation plan.

| Phase | Scope                                              |
| ----- | -------------------------------------------------- |
| 1     | Core runtime: types, tool, state store, executor, engine |
| 2     | Anthropic + OpenAI adapters, invoice example e2e   |
| 3     | Checkpoints (memory, file), console/JSON hooks, graph viz |
| 4     | Conversational states, HITL, error policies, replay |
| 5     | Redis/SQL checkpoints, audit/metrics hooks, async, Azure/LiteLLM adapters |

---

## License

See `LICENSE`.
