"""Tests for token-based compaction triggers and context_window auto-population."""

from __future__ import annotations

import pytest

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart, capture_run_messages
from pydantic_ai.capabilities.compaction._trigger import should_compact
from pydantic_ai.capabilities.compaction.masking import ObservationMaskingCapability
from pydantic_ai.messages import RetryPromptPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile, lookup_context_window
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage, RunUsage

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsStr

pytestmark = [pytest.mark.anyio]


# --- lookup_context_window tests ---


def test_lookup_context_window_known_model():
    """Known model returns a context window value."""
    result = lookup_context_window('openai', 'gpt-4o')
    assert result is not None
    assert result > 0


def test_lookup_context_window_unknown_model():
    """Unknown model returns None."""
    assert lookup_context_window('openai', 'totally-fake-model-xyz') is None


def test_lookup_context_window_unknown_provider():
    """Unknown provider returns None."""
    assert lookup_context_window('totally-fake-provider', 'some-model') is None


# --- context_window auto-population on Model.profile ---


def test_context_window_auto_populated():
    """context_window is auto-populated from genai-prices for known models."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test-key'))
    assert model.profile.context_window is not None
    assert model.profile.context_window > 0


def test_context_window_user_override():
    """User-specified context_window takes precedence over genai-prices."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    profile = ModelProfile(context_window=42_000)
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test-key'), profile=profile)
    assert model.profile.context_window == snapshot(42_000)


# --- should_compact tests ---


def test_should_compact_token_triggers():
    """90k total / 128k context = 70.3% > 0.7 threshold → should trigger."""
    usage = RunUsage(input_tokens=85_000, output_tokens=5_000)
    assert should_compact(usage, context_window=128_000, trigger_ratio=0.7) == snapshot(True)


def test_should_compact_token_no_trigger():
    """50k total / 128k context = 39% < 0.7 threshold → should not trigger."""
    usage = RunUsage(input_tokens=45_000, output_tokens=5_000)
    assert should_compact(usage, context_window=128_000, trigger_ratio=0.7) == snapshot(False)


def test_should_compact_custom_ratio():
    """50k total / 128k context = 39% > 0.3 custom threshold → should trigger."""
    usage = RunUsage(input_tokens=45_000, output_tokens=5_000)
    assert should_compact(usage, context_window=128_000, trigger_ratio=0.3) == snapshot(True)


def test_should_compact_zero_tokens():
    """Zero tokens → should not trigger."""
    usage = RunUsage()
    assert should_compact(usage, context_window=128_000, trigger_ratio=0.7) == snapshot(False)


def test_should_compact_no_context_window():
    """When context_window is None → should not trigger."""
    usage = RunUsage(input_tokens=100_000, output_tokens=10_000)
    assert should_compact(usage, context_window=None, trigger_ratio=0.7) == snapshot(False)


# --- ObservationMaskingCapability with token-based trigger ---


async def test_masking_capability_token_trigger():
    """ObservationMaskingCapability triggers based on cumulative token usage.

    Needs 2 rounds of tool calls so the first round's tool return falls outside keep_last
    and can be masked when compaction triggers on the 3rd model request.
    """
    profile = ModelProfile(context_window=1_000)

    call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Round 1: tool call
            return ModelResponse(
                parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='1')],
                usage=RequestUsage(input_tokens=300, output_tokens=50),
            )
        if call_count == 2:
            # Round 2: another tool call (cumulative total now > 400)
            return ModelResponse(
                parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='2')],
                usage=RequestUsage(input_tokens=300, output_tokens=50),
            )
        # Round 3: final answer
        return ModelResponse(
            parts=[TextPart(content='done')],
            usage=RequestUsage(input_tokens=400, output_tokens=50),
        )

    def my_tool(ctx: RunContext[None]) -> str:
        """A tool that returns data."""
        return 'long tool output data'

    model = FunctionModel(model_fn, profile=profile)
    # trigger_ratio=0.4 → triggers when total_tokens/1000 > 0.4 (i.e. > 400 tokens)
    # After round 1: total=350. After round 2: total=700 → 70% > 40% → triggers on request 3
    # keep_last=2 → only last 2 messages untouched, first tool return gets masked
    processor = ObservationMaskingCapability(keep_last=2, trigger_ratio=0.4)
    agent = Agent(model, tools=[my_tool], capabilities=[processor])

    with capture_run_messages() as captured:
        result = await agent.run('call the tool')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='call the tool', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='1')],
                usage=RequestUsage(input_tokens=300, output_tokens=50),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='my_tool', content='[compacted]', tool_call_id='1', timestamp=IsDatetime())
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='2')],
                usage=RequestUsage(input_tokens=300, output_tokens=50),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='my_tool', content='long tool output data', tool_call_id='2', timestamp=IsDatetime()
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=400, output_tokens=50),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )


async def test_masking_capability_no_trigger_below_ratio():
    """ObservationMaskingCapability does NOT trigger when below ratio threshold."""
    profile = ModelProfile(context_window=1_000_000)

    call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='1')],
                usage=RequestUsage(input_tokens=500, output_tokens=50),
            )
        return ModelResponse(
            parts=[TextPart(content='done')],
            usage=RequestUsage(input_tokens=600, output_tokens=50),
        )

    def my_tool(ctx: RunContext[None]) -> str:
        """A tool that returns data."""
        return 'long tool output data'

    model = FunctionModel(model_fn, profile=profile)
    # trigger_ratio=0.7 → needs 700k tokens. Usage is ~550 total → no trigger
    processor = ObservationMaskingCapability(keep_last=2, trigger_ratio=0.7)
    agent = Agent(model, tools=[my_tool], capabilities=[processor])

    with capture_run_messages() as captured:
        result = await agent.run('call the tool')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='call the tool', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='1')],
                usage=RequestUsage(input_tokens=500, output_tokens=50),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='my_tool', content='long tool output data', tool_call_id='1', timestamp=IsDatetime()
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=600, output_tokens=50),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )
