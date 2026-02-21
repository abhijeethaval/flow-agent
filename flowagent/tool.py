"""@tool decorator and ToolDef for FlowAgent."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel


@dataclass
class ToolDef:
    """Metadata for a registered tool."""

    name: str
    description: str
    parameters_schema: Optional[Type[BaseModel]]  # Pydantic model for input validation
    return_schema: Optional[Type[BaseModel]]  # Pydantic model for output typing
    function: Callable  # The actual implementation
    is_async: bool = False


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    params: Optional[Type[BaseModel]] = None,
    returns: Optional[Type[BaseModel]] = None,
) -> Callable[..., ToolDef]:
    """Decorator to register a function as a FlowAgent tool.

    Usage::

        class ValidateParams(BaseModel):
            data: dict
            strict: bool = True

        class ValidateResult(BaseModel):
            valid: bool
            errors: list[str] = []

        @tool(
            name="validate_schema",
            description="Validate data against the required schema",
            params=ValidateParams,
            returns=ValidateResult,
        )
        def validate_schema(data: dict, strict: bool = True) -> dict:
            ...
            return {"valid": True, "errors": []}
    """

    def decorator(fn: Callable) -> ToolDef:
        tool_name = name or fn.__name__
        tool_desc = description or fn.__doc__ or ""

        tool_def = ToolDef(
            name=tool_name,
            description=tool_desc,
            parameters_schema=params,
            return_schema=returns,
            function=fn,
            is_async=asyncio.iscoroutinefunction(fn),
        )
        # Preserve original function for introspection
        tool_def.__wrapped__ = fn  # type: ignore[attr-defined]
        return tool_def

    return decorator
