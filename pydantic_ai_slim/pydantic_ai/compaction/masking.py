"""History processor that masks old tool return content to reduce token usage."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from pydantic_ai._run_context import RunContext
from pydantic_ai.compaction._trigger import should_compact
from pydantic_ai.messages import ModelMessage, ModelRequest, RetryPromptPart, ToolReturnPart


@dataclass
class ObservationMaskingProcessor:
    """A history processor that replaces old tool return content with a placeholder.

    Keeps the structure of the conversation intact (tool call/return pairs are preserved)
    but reduces token usage by masking the content of tool returns older than `keep_last`.

    When the model's context window is known, compaction triggers based on context window
    utilization (input tokens / context window > trigger_ratio). Otherwise, falls back to
    message count.

    Usage:
        agent = Agent('openai:gpt-5.2', history_processors=[ObservationMaskingProcessor()])
    """

    keep_last: int = 10
    """Number of recent messages to leave untouched."""

    placeholder: str = '[compacted]'
    """Replacement text for masked tool return content."""

    trigger_ratio: float = 0.7
    """Context window utilization ratio that triggers compaction.

    Only used when the model's context window size is known.
    """

    async def __call__(self, ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
        context_window = ctx.model.profile.context_window
        if not should_compact(
            messages, context_window=context_window, trigger_ratio=self.trigger_ratio, fallback_threshold=self.keep_last
        ):
            return messages

        cutoff = len(messages) - self.keep_last
        older = copy.deepcopy(messages[:cutoff])

        for message in older:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, (ToolReturnPart, RetryPromptPart)):
                        part.content = self.placeholder

        return [*older, *messages[cutoff:]]
