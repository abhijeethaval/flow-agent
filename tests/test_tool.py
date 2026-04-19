"""Unit tests for :mod:`flowagent.tool`."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from flowagent.tool import ToolDef, tool


class Params(BaseModel):
    x: int
    y: int = 1


class Returns(BaseModel):
    total: int


def test_tool_decorator_basic_metadata():
    @tool(
        name="add",
        description="Add two ints",
        params=Params,
        returns=Returns,
    )
    def add(x: int, y: int = 1) -> dict:
        return {"total": x + y}

    assert isinstance(add, ToolDef)
    assert add.name == "add"
    assert add.description == "Add two ints"
    assert add.parameters_schema is Params
    assert add.return_schema is Returns
    assert add.is_async is False
    assert add.function(x=3, y=4) == {"total": 7}
    assert add.__wrapped__(3, 4) == {"total": 7}


def test_tool_defaults_from_function():
    @tool()
    def my_func(x: int) -> dict:
        """Docstring describing the tool."""
        return {"x": x}

    assert my_func.name == "my_func"
    assert my_func.description == "Docstring describing the tool."


def test_tool_detects_async():
    @tool(name="ado", description="async")
    async def ado(x: int) -> dict:
        return {"x": x}

    assert ado.is_async is True
    result = asyncio.run(ado.function(x=7))
    assert result == {"x": 7}


def test_tool_without_schemas():
    @tool(name="bare", description="no schemas")
    def bare() -> dict:
        return {}

    assert bare.parameters_schema is None
    assert bare.return_schema is None
