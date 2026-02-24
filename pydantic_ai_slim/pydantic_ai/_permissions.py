"""Permission evaluation for before-tool-call hooks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatch
from typing import Any

from ._run_context import AgentDepsT, RunContext
from .exceptions import ApprovalRequired, ToolCallDenied
from .tools import ToolDefinition

__all__ = (
    'ToolPermission',
    'PermissionRule',
    'evaluate_permission',
    'default_permission_key',
    'permission_hook',
)

PermissionKeyFunc = Callable[[ToolDefinition, dict[str, Any]], str]
"""A function that derives a permission key from a tool definition and its arguments."""


class ToolPermission(Enum):
    """Decision for a permission rule."""

    ALLOW = 'allow'
    """Allow the tool call to proceed."""
    DENY = 'deny'
    """Deny the tool call and return a denial message to the model."""
    ASK = 'ask'
    """Defer the tool call for human-in-the-loop approval."""


@dataclass(frozen=True)
class PermissionRule:
    """A glob-based permission rule.

    Attributes:
        pattern: A glob pattern matched against the permission key (e.g. `"Shell(git status*)"`).
        decision: The `ToolPermission` to return when this rule matches.
    """

    pattern: str
    decision: ToolPermission


def evaluate_permission(
    permission_key: str,
    rules: Sequence[PermissionRule],
    *,
    default: ToolPermission = ToolPermission.ASK,
) -> ToolPermission:
    """Evaluate permission rules against a key using first-match-wins semantics.

    Args:
        permission_key: The key to match rules against.
        rules: Ordered sequence of rules; the first matching rule wins.
        default: The decision when no rule matches.

    Returns:
        The `ToolPermission` from the first matching rule, or `default`.
    """
    for rule in rules:
        if fnmatch(permission_key, rule.pattern):
            return rule.decision
    return default


def default_permission_key(tool_def: ToolDefinition, tool_args: dict[str, Any]) -> str:
    """Return `tool_def.name` as the permission key."""
    return tool_def.name


def permission_hook(
    rules: Sequence[PermissionRule],
    *,
    key_func: PermissionKeyFunc = default_permission_key,
    default: ToolPermission = ToolPermission.ASK,
) -> Callable[..., Any]:
    """Factory that creates a `BeforeToolCallHook` from permission rules.

    The returned hook raises `ToolCallDenied` for denied calls and
    `ApprovalRequired` for calls that need human approval.

    Args:
        rules: Ordered sequence of permission rules.
        key_func: Function to derive the permission key from tool definition and args.
        default: The decision when no rule matches.

    Returns:
        An async callable suitable for use as a `BeforeToolCallHook`.
    """

    async def _hook(
        ctx: RunContext[AgentDepsT],
        tool_def: ToolDefinition,
        tool_args: dict[str, Any],
    ) -> dict[str, Any] | None:
        key = key_func(tool_def, tool_args)
        decision = evaluate_permission(key, rules, default=default)
        if decision is ToolPermission.DENY:
            raise ToolCallDenied()
        elif decision is ToolPermission.ASK:
            raise ApprovalRequired()
        return None

    return _hook
