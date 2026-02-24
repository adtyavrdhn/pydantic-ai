"""Tests for before/after tool call hooks on ToolManager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from pydantic_ai import FunctionToolset, ToolCallPart
from pydantic_ai._run_context import RunContext
from pydantic_ai._tool_manager import ToolManager
from pydantic_ai.exceptions import ApprovalRequired, ToolCallDenied
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import AfterToolCallHook, BeforeToolCallHook, ToolDefinition
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio


@dataclass
class Deps:
    pass


def build_run_context(deps: Any = None, run_step: int = 0) -> RunContext[Any]:
    return RunContext(
        deps=deps or Deps(),
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=run_step,
    )


async def _make_tool_manager(
    *,
    before_hooks: list[BeforeToolCallHook[Any]] | None = None,
    after_hooks: list[AfterToolCallHook[Any]] | None = None,
) -> ToolManager[Any]:
    """Helper to create a ToolManager with an `add` tool and optional hooks."""
    toolset: FunctionToolset[Any] = FunctionToolset()

    @toolset.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    ctx = build_run_context()
    tm = ToolManager(
        toolset=toolset,
        before_tool_call_hooks=list(before_hooks or []),
        after_tool_call_hooks=list(after_hooks or []),
    )
    return await tm.for_run_step(ctx)


# --- Before hook tests ---


async def test_before_hook_returning_none_proceeds():
    async def noop_hook(ctx: RunContext[Any], tool_def: ToolDefinition, tool_args: dict[str, Any]) -> None:
        return None

    tm = await _make_tool_manager(before_hooks=[noop_hook])
    result = await tm.handle_call(ToolCallPart(tool_name='add', args={'a': 1, 'b': 2}))
    assert result == 3


async def test_before_hook_raising_tool_call_denied():
    async def deny_hook(ctx: RunContext[Any], tool_def: ToolDefinition, tool_args: dict[str, Any]) -> None:
        raise ToolCallDenied('Not allowed')

    tm = await _make_tool_manager(before_hooks=[deny_hook])
    # ToolCallDenied propagates out of handle_call; the graph layer catches it.
    with pytest.raises(ToolCallDenied, match='Not allowed'):
        await tm.handle_call(ToolCallPart(tool_name='add', args={'a': 1, 'b': 2}))


async def test_before_hook_raising_approval_required():
    async def ask_hook(ctx: RunContext[Any], tool_def: ToolDefinition, tool_args: dict[str, Any]) -> None:
        raise ApprovalRequired()

    tm = await _make_tool_manager(before_hooks=[ask_hook])
    with pytest.raises(ApprovalRequired):
        await tm.handle_call(ToolCallPart(tool_name='add', args={'a': 1, 'b': 2}))


async def test_before_hook_returning_dict_modifies_args():
    async def modify_hook(
        ctx: RunContext[Any], tool_def: ToolDefinition, tool_args: dict[str, Any]
    ) -> dict[str, Any]:
        return {'a': 10, 'b': 20}

    tm = await _make_tool_manager(before_hooks=[modify_hook])
    result = await tm.handle_call(ToolCallPart(tool_name='add', args={'a': 1, 'b': 2}))
    assert result == 30


async def test_multiple_before_hooks_run_in_order():
    call_order: list[str] = []

    async def hook_a(ctx: RunContext[Any], tool_def: ToolDefinition, tool_args: dict[str, Any]) -> None:
        call_order.append('a')
        return None

    async def hook_b(ctx: RunContext[Any], tool_def: ToolDefinition, tool_args: dict[str, Any]) -> None:
        call_order.append('b')
        return None

    tm = await _make_tool_manager(before_hooks=[hook_a, hook_b])
    await tm.handle_call(ToolCallPart(tool_name='add', args={'a': 1, 'b': 2}))
    assert call_order == ['a', 'b']


async def test_multiple_before_hooks_first_deny_wins():
    call_order: list[str] = []

    async def allow_hook(ctx: RunContext[Any], tool_def: ToolDefinition, tool_args: dict[str, Any]) -> None:
        call_order.append('allow')
        return None

    async def deny_hook(ctx: RunContext[Any], tool_def: ToolDefinition, tool_args: dict[str, Any]) -> None:
        call_order.append('deny')
        raise ToolCallDenied()

    async def unreachable_hook(ctx: RunContext[Any], tool_def: ToolDefinition, tool_args: dict[str, Any]) -> None:
        call_order.append('unreachable')
        return None

    tm = await _make_tool_manager(before_hooks=[allow_hook, deny_hook, unreachable_hook])
    with pytest.raises(ToolCallDenied):
        await tm.handle_call(ToolCallPart(tool_name='add', args={'a': 1, 'b': 2}))
    assert call_order == ['allow', 'deny']


async def test_before_hook_dict_chaining():
    """Multiple hooks that modify args: each sees the args from the previous hook."""

    async def double_a(
        ctx: RunContext[Any], tool_def: ToolDefinition, tool_args: dict[str, Any]
    ) -> dict[str, Any]:
        return {**tool_args, 'a': tool_args['a'] * 2}

    async def double_b(
        ctx: RunContext[Any], tool_def: ToolDefinition, tool_args: dict[str, Any]
    ) -> dict[str, Any]:
        return {**tool_args, 'b': tool_args['b'] * 2}

    tm = await _make_tool_manager(before_hooks=[double_a, double_b])
    # a=3 -> 6, b=4 -> 8, result = 14
    result = await tm.handle_call(ToolCallPart(tool_name='add', args={'a': 3, 'b': 4}))
    assert result == 14


# --- After hook tests ---


async def test_after_hook_can_transform_result():
    async def double_result(
        ctx: RunContext[Any], tool_def: ToolDefinition, tool_args: dict[str, Any], result: Any
    ) -> Any:
        return result * 2

    tm = await _make_tool_manager(after_hooks=[double_result])
    result = await tm.handle_call(ToolCallPart(tool_name='add', args={'a': 1, 'b': 2}))
    assert result == 6


async def test_multiple_after_hooks_chain():
    async def add_ten(
        ctx: RunContext[Any], tool_def: ToolDefinition, tool_args: dict[str, Any], result: Any
    ) -> Any:
        return result + 10

    async def double_result(
        ctx: RunContext[Any], tool_def: ToolDefinition, tool_args: dict[str, Any], result: Any
    ) -> Any:
        return result * 2

    tm = await _make_tool_manager(after_hooks=[add_ten, double_result])
    # add(1,2) = 3, +10 = 13, *2 = 26
    result = await tm.handle_call(ToolCallPart(tool_name='add', args={'a': 1, 'b': 2}))
    assert result == 26


async def test_after_hook_receives_tool_def_and_args():
    captured: dict[str, Any] = {}

    async def capture_hook(
        ctx: RunContext[Any], tool_def: ToolDefinition, tool_args: dict[str, Any], result: Any
    ) -> Any:
        captured['tool_name'] = tool_def.name
        captured['tool_args'] = tool_args
        captured['result'] = result
        return result

    tm = await _make_tool_manager(after_hooks=[capture_hook])
    await tm.handle_call(ToolCallPart(tool_name='add', args={'a': 7, 'b': 8}))
    assert captured['tool_name'] == 'add'
    assert captured['tool_args'] == {'a': 7, 'b': 8}
    assert captured['result'] == 15


# --- Hook propagation ---


async def test_hooks_propagate_through_for_run_step():
    call_count = 0

    async def counting_hook(ctx: RunContext[Any], tool_def: ToolDefinition, tool_args: dict[str, Any]) -> None:
        nonlocal call_count
        call_count += 1
        return None

    toolset: FunctionToolset[Any] = FunctionToolset()

    @toolset.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    ctx = build_run_context()
    tm = ToolManager(toolset=toolset, before_tool_call_hooks=[counting_hook])
    tm1 = await tm.for_run_step(ctx)
    await tm1.handle_call(ToolCallPart(tool_name='add', args={'a': 1, 'b': 2}))
    assert call_count == 1

    # Propagate to next step
    ctx2 = build_run_context(run_step=1)
    tm2 = await tm1.for_run_step(ctx2)
    await tm2.handle_call(ToolCallPart(tool_name='add', args={'a': 1, 'b': 2}))
    assert call_count == 2


# --- Before hook receives correct context ---


async def test_before_hook_receives_tool_def():
    captured_name: list[str] = []

    async def capture_hook(ctx: RunContext[Any], tool_def: ToolDefinition, tool_args: dict[str, Any]) -> None:
        captured_name.append(tool_def.name)
        return None

    tm = await _make_tool_manager(before_hooks=[capture_hook])
    await tm.handle_call(ToolCallPart(tool_name='add', args={'a': 1, 'b': 2}))
    assert captured_name == ['add']
