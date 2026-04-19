# FlowAgent Phased Implementation Plan

Derived from `spec.md` §16 and expanded with deliverables, acceptance criteria, and dependencies per phase. Each phase is shippable on its own — later phases layer capability onto earlier ones without refactoring the core.

Legend: **D** deliverable, **AC** acceptance criteria, **Deps** dependencies.

---

## Phase 1 — Core Runtime (Week 1-2)

Goal: a workflow graph can be defined, validated, and executed against a mock LLM adapter.

### Deliverables

- `flowagent/types.py` — `StateMode`, `TransitionDef`, `StateDef`, `WorkflowDefinition`, `ToolCall`, `ToolResult`, `LLMResponse`, `ToolCallRecord`, `StateExecutionResult`, `WorkflowExecutionResult`, `Checkpoint`, `ErrorPolicy`.
- `flowagent/errors.py` — Full error hierarchy (`FlowAgentError` and subclasses from spec §11.1).
- `flowagent/tool.py` — `@tool` decorator and `ToolDef` (pydantic-typed params / returns, sync + async detection).
- `flowagent/state_store.py` — `StateStore` with `get`, `update`, `extract(schema)`, `to_dict`, immutable snapshot history.
- `flowagent/transition.py` — `_make_transition_tool(transitions)` auto-generator, interception helpers.
- `flowagent/prompt_builder.py` — `_build_system_prompt(state_def, global_prompt)` per spec §4.2.
- `flowagent/validation.py` — `WorkflowDefinition.validate()` implementing all 9 rules from spec §6.1.
- `flowagent/state_executor.py` — `StateExecutor.execute` covering AGENTIC + DETERMINISTIC modes, tool-call loop, transition detection, scope enforcement, `max_tool_calls` bound.
- `flowagent/engine.py` — `WorkflowEngine.run` driving states to a terminal state; step/visit bounds.
- `tests/` — Unit tests for every module above, plus a `MockLLMAdapter` fixture that scripts tool calls and transition declarations.

### Acceptance Criteria

- [ ] `pytest` passes with >90% branch coverage on core modules.
- [ ] Graph validation rejects: missing initial state, unreachable states, terminal-with-transitions, agentic-without-tools, deterministic-with-ambiguous-transition.
- [ ] A workflow running against `MockLLMAdapter` executes an end-to-end path that hits a terminal state without calling any real LLM.
- [ ] `ToolNotAllowedError`, `InvalidTransitionError`, `MaxToolCallsExceeded`, `NoTransitionDeclaredError` all fire on their respective triggers.

### Dependencies

- Runtime: `pydantic>=2.0` only. No LLM SDK yet.

---

## Phase 2 — LLM Adapters (Week 2-3)

Goal: drive a real workflow against Anthropic and OpenAI.

### Deliverables

- `flowagent/adapters/base.py` — `LLMAdapter` ABC with sync `call` and async `acall`.
- `flowagent/adapters/anthropic_adapter.py` — Anthropic tool-use schema, `declare_transition` injection, response parsing, basic retry on transient errors.
- `flowagent/adapters/openai_adapter.py` — OpenAI function-calling schema, same surface.
- `examples/invoice_processing.py` — Full invoice workflow from spec §13 runnable end-to-end with either adapter.
- Provider integration tests (gated on env-var API keys, skipped otherwise).

### Acceptance Criteria

- [ ] Adapter conversion is symmetric: `ToolDef` → provider format → parsed `ToolCall` with identical name + arguments.
- [ ] `declare_transition` is always present alongside work tools in the adapter's outbound payload.
- [ ] Invoice example completes with `status="completed"` on a fixture invoice against both providers.
- [ ] `RateLimitError`, `ContextOverflowError`, `MalformedResponseError` surface with provider context preserved.

### Dependencies

- Phase 1 complete.
- Optional extras: `anthropic>=0.40`, `openai>=1.50`.

---

## Phase 3 — Persistence and Observability (Week 3-4)

Goal: workflows can be checkpointed, resumed, logged, and visualized.

### Deliverables

- `flowagent/checkpoints/base.py` — `CheckpointStore` ABC (`save`, `load`, `load_latest`, `list_checkpoints`).
- `flowagent/checkpoints/memory.py` — `InMemoryCheckpointStore`.
- `flowagent/checkpoints/file.py` — `FileCheckpointStore` (JSON on disk).
- `flowagent/hooks/base.py` — `WorkflowHook` ABC with all lifecycle methods from spec §9.1.
- `flowagent/hooks/console.py` — `ConsoleLogHook` (pretty-print execution).
- `flowagent/hooks/json_log.py` — `JSONFileLogHook` (structured JSONL per run).
- `WorkflowEngine.run(..., resume_from=...)` restoring store + trace + current state.
- `flowagent/utils/graph_viz.py` — Mermaid and Graphviz renderers for `WorkflowDefinition`.

### Acceptance Criteria

- [ ] A workflow interrupted mid-run (simulated exception between states) resumes from the last checkpoint and produces the same final state as an uninterrupted run.
- [ ] Checkpoint round-trip (`save` → `load`) preserves `StateStore.data`, `trace`, and `current_state` byte-for-byte via JSON.
- [ ] Mermaid output renders every state node and every transition edge with its description.
- [ ] `ConsoleLogHook` output for the invoice example reads as a human-legible timeline.

### Dependencies

- Phase 2 complete.
- Optional extras: `graphviz>=0.20` for Graphviz rendering.

---

## Phase 4 — Conversational States and Polish (Week 4-5)

Goal: human-in-the-loop flows, error recovery, and replay debugging.

### Deliverables

- `StateExecutor` support for `StateMode.CONVERSATIONAL`: messages persist across LLM calls within the state, a user-input handler is invoked when the LLM asks for input, history discarded on transition.
- `WorkflowEngine.run_interactive(workflow, initial_input, user_input_handler)` per spec §12.1.
- `ErrorPolicy` enforcement in `StateExecutor`:
  - `tool_error_action`: `feed_to_llm` | `retry` | `fail` with `max_tool_retries`.
  - `invalid_transition_action`: `retry_llm` | `fail` with `max_transition_retries`.
  - `max_steps_action`: `fail` | `force_transition`.
- `flowagent/utils/replay.py` — replay a checkpoint's trace for debugging (no LLM calls; validates determinism of tool outcomes).
- Additional examples: `examples/customer_support.py`, `examples/data_pipeline.py`.
- Expanded user-facing documentation under `docs/` (guides, not just the spec).

### Acceptance Criteria

- [ ] Conversational state round-trip: LLM asks a question, `user_input_handler` returns, LLM receives the reply in-context and declares a transition.
- [ ] A tool raising an exception under `feed_to_llm` produces a tool-result message containing the error text; the agent can react and recover.
- [ ] Replay of a stored trace reproduces the same state transitions without invoking the LLM.
- [ ] Customer-support and data-pipeline examples run end-to-end on `MockLLMAdapter`.

### Dependencies

- Phase 3 complete.

---

## Phase 5 — Production Hardening (Week 5-6)

Goal: durable persistence, full observability, async throughout, additional providers, and PyPI release.

### Deliverables

- `flowagent/checkpoints/redis.py` — `RedisCheckpointStore` (TTL-configurable, namespaced keys).
- `flowagent/checkpoints/sql.py` — `SQLCheckpointStore` (SQLAlchemy + Alembic migrations for checkpoint / trace tables).
- `flowagent/hooks/audit.py` — `AuditTrailHook` capturing every tool call / LLM call / transition with timestamps, arguments, and results for compliance replay.
- `flowagent/hooks/metrics.py` — `MetricsHook` emitting counters (tool calls, LLM calls) and histograms (state duration, tokens if provider reports them).
- `flowagent/hooks/otel.py` — `OpenTelemetryHook` exporting spans per state and per tool call.
- Async path: `LLMAdapter.acall` implemented for every adapter, `WorkflowEngine.arun` + `StateExecutor.aexecute`.
- `flowagent/adapters/azure_adapter.py`, `flowagent/adapters/litellm_adapter.py`.
- Performance benchmarks (latency per state, checkpoint overhead, tool-call throughput) with a reproducible harness.
- PyPI packaging: trusted publishing, versioned release notes, extras matching spec §15.

### Acceptance Criteria

- [ ] Redis and SQL checkpoint stores pass the same round-trip test suite as `FileCheckpointStore`.
- [ ] `arun` against `MockLLMAdapter` produces bit-identical `WorkflowExecutionResult` to `run` for the same workflow.
- [ ] OpenTelemetry hook emits one span per state execution with tool calls as child spans, verified via an in-memory OTEL collector.
- [ ] `flowagent==0.1.0` (or chosen release) installs cleanly from TestPyPI with each declared extra.
- [ ] Benchmark report committed under `docs/benchmarks/` covering p50/p95 for a reference workflow.

### Dependencies

- Phase 4 complete.
- Optional extras: `redis>=5.0`, `sqlalchemy>=2.0`, `alembic>=1.13`, `opentelemetry-sdk`.

---

## Cross-Phase Concerns

These are not phase-specific but must hold from Phase 1 onward.

- **Tool-scope enforcement.** Tools called outside a state's allowed set always raise `ToolNotAllowedError` — never a silent pass-through.
- **Transition-scope enforcement.** Transitions outside a state's allowed set always raise `InvalidTransitionError`.
- **Stateless between states.** No chat history crosses state boundaries; only the `StateStore` carries data.
- **Graph immutability.** `WorkflowDefinition` is constructed once per release; the engine never mutates it.
- **Minimal core deps.** Optional provider / backend SDKs are extras — importing `flowagent` with only `pydantic` installed must work.
- **Observability without coupling.** Hooks are the only way logging/metrics/tracing enter the core.

---

## Open Questions (from spec §18)

Resolve before the phase that first touches each area.

| Question                      | Resolve by | Current recommendation                                  |
| ----------------------------- | ---------- | ------------------------------------------------------- |
| Streaming support             | Phase 4    | Conversational states only.                             |
| Parallel tool execution       | Phase 2    | Parallel by default with sequential opt-out.            |
| State re-entry limits         | Phase 1    | `max_visits` per state, default 3.                      |
| Workflow composition          | Phase 5    | Special `DETERMINISTIC` state invoking a child engine.  |
| Framework name                | Phase 5    | Lock before PyPI packaging.                             |
