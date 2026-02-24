"""Pydantic AI PR Review Bot — replaces Claude Code in bots.yml with ~50 lines of agent definition.

This script demonstrates the new features on the `jack` branch:
- ExecutionEnvironmentToolset + LocalEnvironment (shell, file reading, grep, glob)
- web_fetch_tool (fetching web pages with SSRF protection)
- TodoToolset (structured task tracking for the review)
- ObservationMaskingCapability (compaction for long conversations)
- PermissionRule + permission_hook (controlling which shell commands are allowed)

Usage:
    python scripts/review_bot.py <pr_number> [repo]

Example:
    python scripts/review_bot.py 4303 pydantic/pydantic-ai
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import logfire

logfire.configure()

from pydantic_ai import Agent
from pydantic_ai._permissions import PermissionRule, ToolPermission, permission_hook
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.common_tools.todos import TodoToolset
from pydantic_ai.common_tools.web_fetch import web_fetch_tool
from pydantic_ai.capabilities.compaction import ObservationMaskingCapability
from pydantic_ai.environments.local import LocalEnvironment
from pydantic_ai.toolsets.execution_environment import ExecutionEnvironmentToolset

REVIEW_INSTRUCTIONS = """\
You are a code review bot for the pydantic/pydantic-ai repository.

## Tools

- Shell: `gh` CLI for PR context and posting comments, `git` for history
- File system: `read_file`, `glob`, `grep` to explore the checked-out code
- Web fetch: look up linked issues, docs, or API references
- Todos: track each review finding — every todo = one review comment to post

## Task

Review PR #{pr_number} in {repo}.

## Workflow

1. Gather PR context:
   - `shell("gh pr view {pr_number} --repo {repo}")` for title and description
   - `shell("gh pr diff {pr_number} --repo {repo} --name-only")` for changed files
   - `shell("gh pr diff {pr_number} --repo {repo}")` for the full diff
2. Read CLAUDE.md for repository standards.
3. Review each changed file against the coding guidelines.
4. For each problem found, create a todo with the finding (file, line, issue, suggestion).
5. After reviewing all files, post your findings as a single PR comment summarizing all issues:
   ```
   shell("gh pr comment {pr_number} --repo {repo} --body '<markdown review body>'")
   ```
   The review body should list all findings with file paths, line numbers, and suggestions.
   This works on both open AND merged PRs.
6. Mark each todo as completed after the comment is posted.

## What to look for

- Public API design issues (missing keyword-only `*`, leaked internals, unclear naming)
- Type safety problems (`Any`, `cast`, missing type annotations, `# type: ignore`)
- Missing or insufficient test coverage
- Missing docstrings on public API
- Broad exception handlers (catching `Exception` instead of specific types)
- Backward-incompatible changes
- Violations of coding guidelines in CLAUDE.md

## Important

- Every todo must be a concrete, actionable review finding — NOT a planning step.
- Do not create todos like "gather context" or "read standards". Just do those things.
- Be concise: 1-3 paragraphs per comment. Include a concrete suggestion when possible.
- Do not comment on things that are fine. Only flag problems.
- Call multiple tools in parallel whenever possible. For example, fetch the PR view, \
  diff, and changed file list in a single step with parallel tool calls.
- You MUST post your review as a PR comment using `gh pr comment`. Do not just return \
  text output — the review must be visible on the PR itself. This works on merged PRs too.
"""

# Permission rules: allow read-only git/gh commands, deny everything else
SHELL_PERMISSIONS = [
    # Read-only GitHub CLI
    PermissionRule('shell(gh pr *)', ToolPermission.ALLOW),
    PermissionRule('shell(gh issue *)', ToolPermission.ALLOW),
    # GitHub API (read + post review comments)
    PermissionRule('shell(gh api *)', ToolPermission.ALLOW),
    # Read-only git
    PermissionRule('shell(git log *)', ToolPermission.ALLOW),
    PermissionRule('shell(git diff *)', ToolPermission.ALLOW),
    PermissionRule('shell(git show *)', ToolPermission.ALLOW),
    PermissionRule('shell(git status*)', ToolPermission.ALLOW),
    PermissionRule('shell(jq *)', ToolPermission.ALLOW),
    # Deny everything else (rm, git push, pip install, etc.)
    PermissionRule('shell(*)', ToolPermission.DENY),
    # Allow all non-shell tools unconditionally
    PermissionRule('*', ToolPermission.ALLOW),
]


def _shell_permission_key(tool_def: ToolDefinition, tool_args: dict[str, Any]) -> str:
    """Build a permission key that includes the command for shell calls."""
    if tool_def.name == 'shell':
        cmd = tool_args.get('command', '')
        return f'shell({cmd})'
    return tool_def.name


def build_review_agent(pr_number: int, repo: str) -> tuple[Agent, LocalEnvironment]:
    """Build the review agent with all the new features composed together."""
    env = LocalEnvironment(root_dir=Path.cwd())

    agent = Agent(
        'gateway/anthropic:claude-sonnet-4-20250514',
        instructions=REVIEW_INSTRUCTIONS.format(pr_number=pr_number, repo=repo),
        toolsets=[
            # Shell + file system tools backed by LocalEnvironment
            ExecutionEnvironmentToolset(
                env,
                include=['shell', 'read_file', 'glob', 'grep'],
            ),
            # Structured task tracking
            TodoToolset(),
        ],
        tools=[
            # Web fetching with SSRF protection
            web_fetch_tool(),
        ],
        capabilities=[
            # Compaction: mask old tool returns when approaching context limit
            ObservationMaskingCapability(keep_last=15, trigger_ratio=0.6),
        ],
        before_tool_call_hooks=[
            # Permission hook: only allow read-only shell commands, deny rm/git push/etc.
            permission_hook(
                SHELL_PERMISSIONS,
                key_func=_shell_permission_key,
                default=ToolPermission.ALLOW,
            ),
        ],
        instrument=True,
    )

    return agent, env


async def main(pr_number: int, repo: str) -> None:
    agent, env = build_review_agent(pr_number, repo)

    print(f'Reviewing PR #{pr_number} in {repo}...\n')

    async with env:
        result = await agent.run(
            f'Please review PR #{pr_number} in {repo}. Start by gathering context, then do a thorough review.'
        )

    print('\n--- Review Result ---\n')
    print(result.output)
    print('\n--- Usage ---')
    print(f'Requests: {result.usage().requests}')
    print(f'Input tokens: {result.usage().input_tokens}')
    print(f'Output tokens: {result.usage().output_tokens}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/review_bot.py <pr_number> [repo]')
        sys.exit(1)

    pr_number = int(sys.argv[1])
    repo = sys.argv[2] if len(sys.argv) > 2 else 'pydantic/pydantic-ai'

    asyncio.run(main(pr_number, repo))
