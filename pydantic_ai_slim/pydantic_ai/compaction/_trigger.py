"""Shared trigger logic for compaction processors."""

from __future__ import annotations

from pydantic_ai.usage import RunUsage


def should_compact(
    usage: RunUsage,
    *,
    context_window: int | None,
    trigger_ratio: float,
) -> bool:
    """Check if compaction should trigger.

    Uses the same cumulative token accounting as UsageLimits.
    Triggers when total_tokens / context_window > trigger_ratio.
    """
    if context_window is not None and usage.total_tokens > 0:
        return usage.total_tokens / context_window > trigger_ratio
    return False
