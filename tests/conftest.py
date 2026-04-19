"""Shared test fixtures — in particular, a scripted ``MockLLMAdapter``."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import pytest

from flowagent.adapters.base import LLMAdapter
from flowagent.tool import ToolDef
from flowagent.types import LLMResponse, ToolCall


ScriptStep = Union[
    # A list of tool calls to return from this LLM turn.
    Sequence["StepToolCall"],
    # A ready-made LLMResponse (for testing exotic paths).
    LLMResponse,
    # A callable that produces either of the above given the inputs.
    Callable[..., Any],
]


@dataclass
class StepToolCall:
    """Scripted tool call for :class:`MockLLMAdapter`."""

    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None


@dataclass
class MockCallRecord:
    """Snapshot of arguments passed into ``MockLLMAdapter.call``."""

    system_prompt: str
    messages: List[Dict[str, Any]]
    tool_names: List[str]
    available_transitions: Dict[str, str]


class MockLLMAdapter(LLMAdapter):
    """Scripted adapter for unit tests.

    The script is a list of steps. On each ``call`` the adapter pops the next
    step and converts it to an :class:`LLMResponse`. When the script is
    exhausted the adapter raises ``AssertionError`` so tests fail loudly rather
    than hang in the tool-calling loop.
    """

    def __init__(self, script: Sequence[ScriptStep]) -> None:
        self._script: List[ScriptStep] = list(script)
        self.calls: List[MockCallRecord] = []

    def call(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: List[ToolDef],
        available_transitions: Dict[str, str],
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.calls.append(
            MockCallRecord(
                system_prompt=system_prompt,
                messages=[dict(m) for m in messages],
                tool_names=[t.name for t in tools],
                available_transitions=dict(available_transitions),
            )
        )

        if not self._script:
            raise AssertionError(
                "MockLLMAdapter script exhausted — test did not supply enough steps"
            )

        step = self._script.pop(0)

        if callable(step) and not isinstance(step, LLMResponse):
            step = step(
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                available_transitions=available_transitions,
            )

        if isinstance(step, LLMResponse):
            return step

        tool_calls = [
            ToolCall(
                id=spec.id or f"call_{uuid.uuid4().hex[:8]}",
                tool_name=spec.tool_name,
                arguments=dict(spec.arguments),
            )
            for spec in step
        ]
        return LLMResponse(
            tool_calls=tool_calls,
            text_content=None,
            transition=None,
            reasoning=None,
            raw_response=None,
        )


@pytest.fixture
def make_mock_adapter():
    """Factory fixture: ``make_mock_adapter([step1, step2, ...])``."""

    def _factory(script: Sequence[ScriptStep]) -> MockLLMAdapter:
        return MockLLMAdapter(script)

    return _factory
