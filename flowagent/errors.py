"""FlowAgent error hierarchy."""


class FlowAgentError(Exception):
    """Base exception for all FlowAgent errors."""


# --- Workflow Validation ---


class WorkflowValidationError(FlowAgentError):
    """Graph integrity issues detected during validation."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        msg = "Workflow validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(msg)


# --- State Execution ---


class StateExecutionError(FlowAgentError):
    """Base for errors during state execution."""


class ToolNotAllowedError(StateExecutionError):
    """Agent called a tool not in the current state's allowed set."""


class InvalidTransitionError(StateExecutionError):
    """Agent chose a transition not in the current state's allowed set."""


class NoTransitionDeclaredError(StateExecutionError):
    """Agent returned a final response without declaring a transition."""


class MaxToolCallsExceeded(StateExecutionError):
    """Tool-calling loop exceeded the per-state bound without a transition."""


class ToolExecutionError(StateExecutionError):
    """A tool raised an exception during execution."""

    def __init__(self, tool_name: str, original_error: Exception):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}' failed: {original_error}")


# --- LLM Adapter ---


class LLMAdapterError(FlowAgentError):
    """Base for LLM adapter errors."""


class RateLimitError(LLMAdapterError):
    """LLM provider rate limit exceeded."""


class ContextOverflowError(LLMAdapterError):
    """Message context exceeds the LLM's context window."""


class MalformedResponseError(LLMAdapterError):
    """LLM returned a response that could not be parsed."""


# --- Checkpoint ---


class CheckpointError(FlowAgentError):
    """Base for checkpoint persistence errors."""


class CheckpointSaveError(CheckpointError):
    """Failed to save a checkpoint."""


class CheckpointLoadError(CheckpointError):
    """Failed to load a checkpoint."""


# --- Configuration ---


class ConfigurationError(FlowAgentError):
    """Misconfigured state or workflow definition."""
