"""FlowAgent: State-Gated Agentic Workflow Framework."""

from flowagent.adapters import LLMAdapter
from flowagent.engine import WorkflowEngine
from flowagent.errors import (
    CheckpointError,
    CheckpointLoadError,
    CheckpointSaveError,
    ConfigurationError,
    ContextOverflowError,
    FlowAgentError,
    InvalidTransitionError,
    LLMAdapterError,
    MalformedResponseError,
    MaxToolCallsExceeded,
    NoTransitionDeclaredError,
    RateLimitError,
    StateExecutionError,
    ToolExecutionError,
    ToolNotAllowedError,
    WorkflowValidationError,
)
from flowagent.state_executor import StateExecutor
from flowagent.state_store import StateStore, StateStoreSnapshot
from flowagent.tool import ToolDef, tool
from flowagent.transition import (
    TRANSITION_TOOL_NAME,
    TransitionParams,
    make_transition_tool,
)
from flowagent.types import (
    Checkpoint,
    ErrorPolicy,
    LLMResponse,
    StateDef,
    StateExecutionResult,
    StateMode,
    ToolCall,
    ToolCallRecord,
    ToolResult,
    TransitionDef,
    WorkflowDefinition,
    WorkflowExecutionResult,
)

__all__ = [
    # Enums
    "StateMode",
    # Graph primitives
    "TransitionDef",
    "StateDef",
    "WorkflowDefinition",
    # Tool
    "ToolDef",
    "tool",
    # State store
    "StateStore",
    "StateStoreSnapshot",
    # Transition
    "TRANSITION_TOOL_NAME",
    "TransitionParams",
    "make_transition_tool",
    # LLM types
    "ToolCall",
    "ToolResult",
    "LLMResponse",
    # Execution records
    "ToolCallRecord",
    "StateExecutionResult",
    "WorkflowExecutionResult",
    # Runtime
    "LLMAdapter",
    "StateExecutor",
    "WorkflowEngine",
    # Config
    "ErrorPolicy",
    "Checkpoint",
    # Errors
    "FlowAgentError",
    "WorkflowValidationError",
    "StateExecutionError",
    "ToolNotAllowedError",
    "InvalidTransitionError",
    "NoTransitionDeclaredError",
    "MaxToolCallsExceeded",
    "ToolExecutionError",
    "LLMAdapterError",
    "RateLimitError",
    "ContextOverflowError",
    "MalformedResponseError",
    "CheckpointError",
    "CheckpointSaveError",
    "CheckpointLoadError",
    "ConfigurationError",
]
