"""History processor that summarizes old messages using an LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai._run_context import RunContext
from pydantic_ai.compaction._trigger import should_compact
from pydantic_ai.compaction.utils import format_messages
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    SystemPromptPart,
)

DEFAULT_SUMMARY_PROMPT = """\
Summarize the following conversation concisely, preserving key facts, decisions, \
and context that would be needed to continue the conversation. \
Focus on what was asked, what was decided, and any important results from tool calls.

Conversation:
{conversation}
"""


@dataclass
class SummarizationProcessor:
    """A history processor that summarizes old messages using a separate LLM call.

    When context window utilization exceeds `trigger_ratio`, messages older than `keep_last`
    are summarized and replaced with a single system prompt containing the summary.

    Usage:
        agent = Agent(
            'openai:gpt-5.2',
            history_processors=[SummarizationProcessor(model='openai:gpt-4.1-mini')],
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

    trigger_ratio: float = 0.7
    """Context window utilization ratio that triggers compaction.

    Only used when the model's context window size is known.
    """

    async def __call__(self, ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
        context_window = ctx.model.profile.context_window
        if not should_compact(ctx.usage, context_window=context_window, trigger_ratio=self.trigger_ratio):
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
