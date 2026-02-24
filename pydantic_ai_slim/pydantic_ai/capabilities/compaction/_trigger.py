"""Shared trigger logic for compaction capabilities."""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelResponse


def last_input_tokens(messages: list[ModelMessage]) -> int | None:
    """Get input_tokens from the most recent ModelResponse."""
    for message in reversed(messages):
        if isinstance(message, ModelResponse) and message.usage.input_tokens > 0:
            return message.usage.input_tokens
    return None


def should_compact(
    messages: list[ModelMessage],
    *,
    context_window: int | None,
    trigger_ratio: float,
    fallback_threshold: int,
) -> bool:
    """Check if compaction should trigger.

    When context_window is known: triggers if input_tokens/context_window > trigger_ratio.
    Otherwise: falls back to message count > fallback_threshold.
    """
    if context_window is not None:
        input_tokens = last_input_tokens(messages)
        if input_tokens is not None:
            return input_tokens / context_window > trigger_ratio
    return len(messages) > fallback_threshold
