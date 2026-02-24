"""Tests for permission evaluation and permission_hook factory."""

from __future__ import annotations

from typing import Any

import pytest

from pydantic_ai import FunctionToolset, ToolCallPart
from pydantic_ai._permissions import (
    PermissionRule,
    ToolPermission,
    default_permission_key,
    evaluate_permission,
    permission_hook,
)
from pydantic_ai._run_context import RunContext
from pydantic_ai._tool_manager import ToolManager
from pydantic_ai.exceptions import ApprovalRequired, ToolCallDenied
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio


def build_run_context(deps: Any = None, run_step: int = 0) -> RunContext[Any]:
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=run_step,
    )


# --- evaluate_permission ---


def test_evaluate_permission_first_match_wins():
    rules = [
        PermissionRule('foo', ToolPermission.DENY),
        PermissionRule('foo', ToolPermission.ALLOW),
    ]
    assert evaluate_permission('foo', rules) is ToolPermission.DENY


def test_evaluate_permission_glob_pattern():
    rules = [
        PermissionRule('Shell(git *)', ToolPermission.ALLOW),
        PermissionRule('Shell(rm *)', ToolPermission.DENY),
        PermissionRule('Shell(*)', ToolPermission.ASK),
    ]
    assert evaluate_permission('Shell(git status)', rules) is ToolPermission.ALLOW
    assert evaluate_permission('Shell(rm -rf /)', rules) is ToolPermission.DENY
    assert evaluate_permission('Shell(echo hello)', rules) is ToolPermission.ASK


def test_evaluate_permission_default_when_no_match():
    rules = [
        PermissionRule('foo', ToolPermission.ALLOW),
    ]
    assert evaluate_permission('bar', rules) is ToolPermission.ASK
    assert evaluate_permission('bar', rules, default=ToolPermission.DENY) is ToolPermission.DENY


def test_evaluate_permission_empty_rules_returns_default():
    assert evaluate_permission('anything', []) is ToolPermission.ASK
    assert evaluate_permission('anything', [], default=ToolPermission.ALLOW) is ToolPermission.ALLOW


def test_evaluate_permission_star_matches_all():
    rules = [
        PermissionRule('*', ToolPermission.ALLOW),
    ]
    assert evaluate_permission('anything', rules) is ToolPermission.ALLOW
    assert evaluate_permission('something_else', rules) is ToolPermission.ALLOW


# --- default_permission_key ---


def test_default_permission_key_returns_tool_name():
    tool_def = ToolDefinition(name='my_tool', description='A tool')
    assert default_permission_key(tool_def, {'arg': 'val'}) == 'my_tool'


# --- permission_hook factory ---


async def test_permission_hook_allow_returns_none():
    rules = [PermissionRule('add', ToolPermission.ALLOW)]
    hook = permission_hook(rules)
    result = await hook(build_run_context(), ToolDefinition(name='add'), {'a': 1})
    assert result is None


async def test_permission_hook_deny_raises_tool_call_denied():
    rules = [PermissionRule('add', ToolPermission.DENY)]
    hook = permission_hook(rules)
    with pytest.raises(ToolCallDenied):
        await hook(build_run_context(), ToolDefinition(name='add'), {'a': 1})


async def test_permission_hook_ask_raises_approval_required():
    rules = [PermissionRule('add', ToolPermission.ASK)]
    hook = permission_hook(rules)
    with pytest.raises(ApprovalRequired):
        await hook(build_run_context(), ToolDefinition(name='add'), {'a': 1})


async def test_permission_hook_custom_key_func():
    rules = [
        PermissionRule('Shell(git *)', ToolPermission.ALLOW),
        PermissionRule('Shell(*)', ToolPermission.DENY),
    ]

    def shell_key(tool_def: ToolDefinition, tool_args: dict[str, Any]) -> str:
        return f'Shell({tool_args.get("command", "")})'

    hook = permission_hook(rules, key_func=shell_key)

    result = await hook(
        build_run_context(),
        ToolDefinition(name='shell'),
        {'command': 'git status'},
    )
    assert result is None

    with pytest.raises(ToolCallDenied):
        await hook(
            build_run_context(),
            ToolDefinition(name='shell'),
            {'command': 'rm -rf /'},
        )


# --- End-to-end: permission_hook on ToolManager ---


async def test_permission_hook_end_to_end_allow():
    toolset: FunctionToolset[Any] = FunctionToolset()

    @toolset.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    rules = [PermissionRule('add', ToolPermission.ALLOW)]
    hook = permission_hook(rules)

    ctx = build_run_context()
    tm = ToolManager(toolset=toolset, before_tool_call_hooks=[hook])
    tm = await tm.for_run_step(ctx)

    result = await tm.handle_call(ToolCallPart(tool_name='add', args={'a': 3, 'b': 4}))
    assert result == 7


async def test_permission_hook_end_to_end_deny():
    toolset: FunctionToolset[Any] = FunctionToolset()

    @toolset.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    rules = [PermissionRule('add', ToolPermission.DENY)]
    hook = permission_hook(rules)

    ctx = build_run_context()
    tm = ToolManager(toolset=toolset, before_tool_call_hooks=[hook])
    tm = await tm.for_run_step(ctx)

    with pytest.raises(ToolCallDenied):
        await tm.handle_call(ToolCallPart(tool_name='add', args={'a': 3, 'b': 4}))


async def test_permission_hook_end_to_end_ask():
    toolset: FunctionToolset[Any] = FunctionToolset()

    @toolset.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    rules = [PermissionRule('add', ToolPermission.ASK)]
    hook = permission_hook(rules)

    ctx = build_run_context()
    tm = ToolManager(toolset=toolset, before_tool_call_hooks=[hook])
    tm = await tm.for_run_step(ctx)

    with pytest.raises(ApprovalRequired):
        await tm.handle_call(ToolCallPart(tool_name='add', args={'a': 3, 'b': 4}))


async def test_permission_hook_end_to_end_default_ask():
    """No rules match, default is ASK."""
    toolset: FunctionToolset[Any] = FunctionToolset()

    @toolset.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    rules = [PermissionRule('other_tool', ToolPermission.ALLOW)]
    hook = permission_hook(rules)

    ctx = build_run_context()
    tm = ToolManager(toolset=toolset, before_tool_call_hooks=[hook])
    tm = await tm.for_run_step(ctx)

    with pytest.raises(ApprovalRequired):
        await tm.handle_call(ToolCallPart(tool_name='add', args={'a': 3, 'b': 4}))
