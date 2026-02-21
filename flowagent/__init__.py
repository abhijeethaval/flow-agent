"""FlowAgent: State-Gated Agentic Workflow Framework."""

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
from flowagent.tool import ToolDef, tool
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
    # LLM types
    "ToolCall",
    "ToolResult",
    "LLMResponse",
    # Execution records
    "ToolCallRecord",
    "StateExecutionResult",
    "WorkflowExecutionResult",
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
