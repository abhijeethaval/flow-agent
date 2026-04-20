"""Abstract LLM adapter interface.

Concrete implementations convert :class:`~flowagent.tool.ToolDef` lists to the
provider's native tool format, perform the API call, and parse the response
into a unified :class:`~flowagent.types.LLMResponse`.

Phase 1 defines the contract so the :class:`~flowagent.state_executor.StateExecutor`
can depend on an abstract surface; concrete adapters ship in Phase 2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from flowagent.tool import ToolDef
from flowagent.types import LLMResponse


class LLMAdapter(ABC):
    """Sync + async interface for LLM providers with tool-use support."""

    @abstractmethod
    def call(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: List[ToolDef],
        available_transitions: Dict[str, str],
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Make a synchronous LLM call and return a parsed response."""

    async def acall(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: List[ToolDef],
        available_transitions: Dict[str, str],
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Async counterpart to :meth:`call`.

        The default implementation raises ``NotImplementedError``; adapters
        that support async must override it. Phase 5 wires async through the
        engine.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement async call()"
        )
