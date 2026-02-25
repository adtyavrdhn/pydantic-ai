"""Capability that masks old tool return content to reduce token usage."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.capabilities.compaction._trigger import should_compact
from pydantic_ai.messages import ModelMessage, ModelRequest, RetryPromptPart, ToolReturnPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext


@dataclass
class ObservationMaskingCapability(AbstractCapability[AgentDepsT]):
    """A capability that replaces old tool return content with a placeholder.

    Keeps the structure of the conversation intact (tool call/return pairs are preserved)
    but reduces token usage by masking the content of tool returns older than `keep_last`.

    Compaction triggers based on cumulative token usage relative to the context window
    (total_tokens / context_window > trigger_ratio), using the same accounting as UsageLimits.

    Usage:
        agent = Agent('openai:gpt-5.2', capabilities=[ObservationMaskingCapability()])
    """

    keep_last: int = 1
    """Number of recent messages to leave untouched."""

    placeholder: str = '[compacted]'
    """Replacement text for masked tool return content."""

    trigger_ratio: float = 0.7
    """Context window utilization ratio that triggers compaction.

    Only used when the model's context window size is known.
    """

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        messages: list[ModelMessage],
        model_settings: ModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[list[ModelMessage], ModelSettings, ModelRequestParameters]:
        messages = self._mask_messages(ctx, messages)
        return messages, model_settings, model_request_parameters

    def _mask_messages(self, ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
        context_window = ctx.model.profile.context_window
        if not should_compact(ctx.usage, context_window=context_window, trigger_ratio=self.trigger_ratio):
            return messages

        cutoff = len(messages) - self.keep_last
        older = copy.deepcopy(messages[:cutoff])

        masked_count = 0
        for message in older:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, (ToolReturnPart, RetryPromptPart)):
                        part.content = self.placeholder
                        masked_count += 1

        with self.span(
            ctx,
            'capability compaction/masking',
            {
                'pydantic_ai.capability': 'compaction/masking',
                'pydantic_ai.compaction.trigger_ratio': self.trigger_ratio,
                'pydantic_ai.compaction.context_utilization': ctx.usage.total_tokens / context_window
                if context_window
                else 0,
                'pydantic_ai.compaction.total_tokens': ctx.usage.total_tokens,
                'pydantic_ai.compaction.context_window': context_window or 0,
                'pydantic_ai.compaction.messages_total': len(messages),
                'pydantic_ai.compaction.messages_kept': len(messages) - cutoff,
                'pydantic_ai.compaction.tool_returns_masked': masked_count,
            },
        ):
            pass

        return [*older, *messages[cutoff:]]
