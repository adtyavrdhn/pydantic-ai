"""Capability that masks tool returns then summarizes old messages using an LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities.compaction._trigger import should_compact
from pydantic_ai.capabilities.compaction.masking import ObservationMaskingCapability
from pydantic_ai.capabilities.compaction.utils import format_messages
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    SystemPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext

DEFAULT_SUMMARY_PROMPT = """\
Summarize the following conversation concisely, preserving key facts, decisions, \
and context that would be needed to continue the conversation. \
Focus on what was asked, what was decided, and any important results from tool calls.

Conversation:
{conversation}
"""


@dataclass(kw_only=True)
class MaskedSummarizationCapability(ObservationMaskingCapability[AgentDepsT]):
    """A capability that masks tool returns then summarizes old messages using a separate LLM call.

    When context window utilization exceeds `trigger_ratio` (or message count exceeds
    `trigger_threshold` when context window is unknown), messages older than `keep_last`
    are first masked (tool returns replaced with placeholders), then summarized and replaced
    with a single system prompt containing the summary.

    Usage:
        agent = Agent(
            'openai:gpt-5.2',
            capabilities=[MaskedSummarizationCapability(agent=Agent('openai:gpt-4.1-mini'))],
        )
    """

    agent: Agent[None]
    """Agent to use for generating summaries. Typically a cheap/fast model."""

    trigger_threshold: int = 20
    """Minimum message count before summarization kicks in (fallback when context window is unknown)."""

    keep_last: int = 10
    """Number of recent messages to preserve unchanged."""

    summary_prompt: str = DEFAULT_SUMMARY_PROMPT
    """Prompt template for the summarization model. Must contain {conversation}."""

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        messages: list[ModelMessage],
        model_settings: ModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[list[ModelMessage], ModelSettings, ModelRequestParameters]:
        # First mask tool returns
        messages = self._mask_messages(ctx, messages)
        # Then summarize if needed
        messages = await self._summarize_messages(ctx, messages)
        return messages, model_settings, model_request_parameters

    async def _summarize_messages(self, ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
        context_window = ctx.model.profile.context_window
        if not should_compact(
            messages,
            context_window=context_window,
            trigger_ratio=self.trigger_ratio,
            fallback_threshold=self.trigger_threshold,
        ):
            return messages

        cutoff = len(messages) - self.keep_last
        older = messages[:cutoff]
        recent = messages[cutoff:]

        conversation_text = format_messages(older)
        prompt = self.summary_prompt.format(conversation=conversation_text)
        result = await self.agent.run(prompt)
        summary = result.output

        summary_message = ModelRequest(parts=[SystemPromptPart(content=f'Summary of prior conversation:\n{summary}')])
        return [summary_message, *recent]
