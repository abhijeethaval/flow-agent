"""LLM adapter package.

Phase 1 only ships :class:`LLMAdapter`, the abstract base class used by the
state executor. Concrete adapters (Anthropic, OpenAI, ...) land in Phase 2.
"""

from flowagent.adapters.base import LLMAdapter

__all__ = ["LLMAdapter"]
