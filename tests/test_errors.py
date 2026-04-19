"""Unit tests for :mod:`flowagent.errors`."""

from __future__ import annotations

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


def test_hierarchy():
    # Top-level
    for cls in (
        WorkflowValidationError,
        StateExecutionError,
        LLMAdapterError,
        CheckpointError,
        ConfigurationError,
    ):
        assert issubclass(cls, FlowAgentError)

    # State execution subclasses
    for cls in (
        ToolNotAllowedError,
        InvalidTransitionError,
        NoTransitionDeclaredError,
        MaxToolCallsExceeded,
        ToolExecutionError,
    ):
        assert issubclass(cls, StateExecutionError)

    # LLM subclasses
    for cls in (RateLimitError, ContextOverflowError, MalformedResponseError):
        assert issubclass(cls, LLMAdapterError)

    # Checkpoint subclasses
    for cls in (CheckpointSaveError, CheckpointLoadError):
        assert issubclass(cls, CheckpointError)


def test_tool_execution_error_captures_cause():
    cause = ValueError("boom")
    exc = ToolExecutionError("extract_fields", cause)
    assert exc.tool_name == "extract_fields"
    assert exc.original_error is cause
    assert "extract_fields" in str(exc)
    assert "boom" in str(exc)
