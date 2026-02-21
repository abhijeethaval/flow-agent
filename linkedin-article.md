# Why I'm Building a New Agentic AI Framework — And Why Existing Ones Aren't Enough

I've spent the last several months working with enterprise AI agent systems — evaluating frameworks, building proof-of-concepts, debugging production workflows. And I've arrived at a conclusion that I think many enterprise architects are quietly reaching: **the current generation of agentic frameworks is built for demos, not for deterministic business processes.**

So I'm building something different. This is the story of why.

---

## The Problem With Today's Agent Frameworks

Every major agentic framework — LangGraph, CrewAI, Semantic Kernel, AutoGen — shares a common assumption: give the LLM a set of tools and a goal, and let it figure out the path.

This works beautifully for open-ended tasks like research, exploration, and creative work. But enterprise workflows aren't open-ended. An invoice processing pipeline has defined steps. A customer onboarding flow has required stages. A compliance review has a mandated sequence.

When I deploy an agent into these workflows, I need to answer questions that current frameworks make surprisingly hard:

- Can the agent skip a required validation step?
- What tools was the agent able to access at each stage?
- Can I guarantee the agent will follow the approved process?
- Can I reproduce and audit every decision the agent made?

The honest answer with existing frameworks is: not really. The LLM can hallucinate tool calls, skip steps, or take unexpected paths. You can add guardrails, but you're fighting the framework's design, not working with it.

## What I Actually Need

After working through dozens of enterprise AI scenarios, the pattern I keep arriving at is:

**A single agent that follows a predefined workflow graph — but retains genuine agency at each step.**

Think of it like a GPS navigation system. The road network (workflow graph) is fixed. The valid routes are predefined. But the driver (agent) decides which turn to take at each intersection based on real-time conditions — traffic, weather, passenger preferences.

The map is deterministic. The navigation is agentic.

Concretely, this means:

1. The workflow is a directed graph of states with predefined transitions.
2. At each state, the agent sees only the tools relevant to that step — nothing more.
3. The agent does its work (calling tools, reasoning about results) and then decides which of the allowed transitions to take.
4. The workflow engine enforces that only valid transitions are possible. The agent cannot jump to an arbitrary state.
5. Back-edges are first-class citizens — the agent can decide to go back and revalidate, re-collect data, or retry a step if the situation warrants it.

This gives you something none of the current frameworks offer cleanly: **deterministic workflow structure with agentic decision-making at every step.**

## Why Not Just Use LangGraph or Semantic Kernel?

I want to be fair to these frameworks — they're powerful and well-engineered. But they're solving a different problem.

**LangGraph** models workflows as graphs of nodes, each with its own LLM call and tool set. This is close to what I need, but it's fundamentally a multi-node architecture. Each node is a separate function call with separate state management. Building a single agent with continuous reasoning that navigates a workflow graph requires fighting the framework's grain. LangGraph's mental model is "graph of processing steps," not "one agent navigating a constrained map."

**Multi-agent frameworks** (CrewAI, AutoGen, Semantic Kernel Agent Framework) solve coordination between multiple specialized agents. But I don't want multiple agents with different personas and separate histories. I want one agent, one identity, one reasoning context — just with different tools available at different stages.

**The "adapt it" argument.** Yes, you can build this pattern on top of any framework with enough custom code. I've done it. The resulting code is fragile, hard to test, and the framework fights you at every turn. When you're spending more time working around the framework than working with it, it's time to build the right abstraction.

## Three Design Decisions That Change Everything

Building this framework forced me to confront three architectural questions that most agent builders don't think carefully about. The answers fundamentally shape the framework's character.

**1. Stateless between states, agentic within states.**

Most agent frameworks accumulate a growing conversation history — every tool call, every result, every intermediate response. By state 7 of a workflow, the LLM is attending to 40+ messages of context from earlier states it no longer needs. This wastes tokens, dilutes attention, and makes states implicitly coupled.

Instead, I treat each state transition as a clean boundary. The state store (a typed data structure) carries forward only the relevant outputs from previous states. Each state gets a fresh, constructed context — not an appended conversation. This is inspired by DSPy's philosophy of treating each LLM call as a typed, stateless function: defined inputs, defined outputs, no accumulated baggage.

Within a state, the agent runs a standard tool-calling loop — call tools, see results, reason, call more tools if needed. That's genuine agency. But between states, the context is surgically constructed, not blindly accumulated.

The result: each state is independently testable, token-efficient, and cleanly isolated. You can test state 5 without constructing the full history of states 1 through 4.

**2. The agent navigates; the workflow constrains.**

In existing frameworks, either the code decides the next step (conditional edges, router functions) or the LLM decides freely (full agent autonomy). Neither is right for enterprise workflows.

My approach: the workflow definition declares which transitions are possible from each state. The agent sees these transitions (with semantic descriptions) and chooses which one to take based on its reasoning and tool results. The workflow engine validates the choice — if the agent picks a transition that doesn't exist in the graph, it's rejected.

This means the agent can decide to go back and revalidate data, escalate to a human, or skip an optional step — but only if the workflow graph explicitly allows that path. The agent has navigational autonomy within defined constraints.

**3. No event sourcing. Just checkpoints.**

I've worked extensively with Temporal for workflow orchestration, and I respect what it does. But Temporal's event-sourcing model — recording every activity dispatch, start, and completion — is overkill for agent workflows. It adds latency, storage overhead, and fundamentally can't handle streaming (which matters for real-time agent interactions).

Instead, the framework checkpoints the state store between steps. If a step fails, you resume from the last checkpoint. Simple, efficient, sufficient. Think "save game" vs "record every keystroke."

## What I'm Building

The framework (working title: **FlowAgent**) is a Python library with these core principles:

- **Single agent, single identity.** One LLM, one system prompt, one orchestrator.
- **Workflow as directed graph.** States, transitions, and back-edges defined declaratively.
- **Per-state tool scoping.** The agent only sees tools relevant to the current state. No more, no less.
- **Agent-driven navigation.** The agent chooses transitions based on tool results and reasoning. The engine enforces the graph constraints.
- **DSPy-style state isolation.** Each state gets typed context from the state store, not accumulated chat history.
- **Checkpoint-based persistence.** Snapshot and resume, not event source.
- **LLM-provider agnostic.** Anthropic, OpenAI, Azure, local models — all via pluggable adapters.
- **Minimal dependencies.** Core framework depends only on Pydantic. Everything else is optional.

I'm building this as an open-source side project, informed by real enterprise requirements from my work in applied AI and distributed systems.

## Who This Is For

If you're building AI-powered automation for business processes that have defined steps, compliance requirements, and auditability needs — this is for you. Think:

- Financial document processing with validation gates
- Customer onboarding with required verification steps
- IT operations with approval workflows
- Healthcare data processing with regulatory checkpoints
- Any workflow where "the AI skipped a step" is unacceptable

## Current Status and What's Next

The detailed specification is complete. I'm now implementing the core runtime — the state executor, workflow engine, and the first LLM adapter (Anthropic Claude). I expect a working MVP with a complete example workflow within the next few weeks.

I'll be sharing progress, design decisions, and technical deep-dives as I build. If this problem resonates with you — if you've felt the friction of trying to make existing agent frameworks behave deterministically — I'd love to hear from you.

What enterprise workflow would you want to automate with a framework like this?

---

_I'm a Principal Software Architect specializing in applied AI and enterprise software platforms. I work on large-scale distributed systems and integrating generative AI into software development workflows. This framework is a side project born from real-world friction with existing agentic tools._

#AgenticAI #AIAgents #EnterpriseAI #SoftwareArchitecture #LLM #OpenSource #Python #AIFramework #WorkflowAutomation
