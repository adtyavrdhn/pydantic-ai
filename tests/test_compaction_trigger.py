"""Tests for token-based compaction triggers and context_window auto-population."""

from __future__ import annotations

import pytest

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart, capture_run_messages
from pydantic_ai.compaction._trigger import last_input_tokens, should_compact
from pydantic_ai.compaction.masking import ObservationMaskingProcessor
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile, lookup_context_window
from pydantic_ai.usage import RequestUsage

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


# --- last_input_tokens tests ---


def test_last_input_tokens_from_response():
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(parts=[TextPart(content='hi')], usage=RequestUsage(input_tokens=100, output_tokens=10)),
    ]
    assert last_input_tokens(messages) == snapshot(100)


def test_last_input_tokens_multiple_responses():
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(parts=[TextPart(content='hi')], usage=RequestUsage(input_tokens=50, output_tokens=5)),
        ModelRequest(parts=[UserPromptPart(content='more')]),
        ModelResponse(parts=[TextPart(content='ok')], usage=RequestUsage(input_tokens=200, output_tokens=20)),
    ]
    assert last_input_tokens(messages) == snapshot(200)


def test_last_input_tokens_no_response():
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
    ]
    assert last_input_tokens(messages) is None


def test_last_input_tokens_zero_tokens():
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(parts=[TextPart(content='hi')], usage=RequestUsage(input_tokens=0, output_tokens=10)),
    ]
    assert last_input_tokens(messages) is None


# --- should_compact tests ---


def test_should_compact_token_triggers():
    """90k/128k = 70.3% > 0.7 threshold → should trigger."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(parts=[TextPart(content='hi')], usage=RequestUsage(input_tokens=90_000, output_tokens=100)),
    ]
    assert should_compact(messages, context_window=128_000, trigger_ratio=0.7, fallback_threshold=10) is True


def test_should_compact_token_no_trigger():
    """50k/128k = 39% < 0.7 threshold → should not trigger."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(parts=[TextPart(content='hi')], usage=RequestUsage(input_tokens=50_000, output_tokens=100)),
    ]
    assert should_compact(messages, context_window=128_000, trigger_ratio=0.7, fallback_threshold=10) is False


def test_should_compact_custom_ratio():
    """50k/128k = 39% > 0.3 custom threshold → should trigger."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(parts=[TextPart(content='hi')], usage=RequestUsage(input_tokens=50_000, output_tokens=100)),
    ]
    assert should_compact(messages, context_window=128_000, trigger_ratio=0.3, fallback_threshold=10) is True


def test_should_compact_fallback_to_message_count():
    """When context_window is None, falls back to message count."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=f'msg {i}')]) for i in range(15)
    ]
    assert should_compact(messages, context_window=None, trigger_ratio=0.7, fallback_threshold=10) is True
    assert should_compact(messages, context_window=None, trigger_ratio=0.7, fallback_threshold=20) is False


def test_should_compact_no_response_falls_back():
    """First turn (no ModelResponse) with context_window set → falls back to message count."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
    ]
    assert should_compact(messages, context_window=128_000, trigger_ratio=0.7, fallback_threshold=10) is False


# --- ObservationMaskingProcessor with token-based trigger ---


async def test_masking_processor_token_trigger():
    """ObservationMaskingProcessor triggers based on token usage when context_window is known."""
    profile = ModelProfile(context_window=128_000)

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content='response')],
            usage=RequestUsage(input_tokens=90_000, output_tokens=500),
        )

    model = FunctionModel(model_fn, profile=profile)
    processor = ObservationMaskingProcessor(keep_last=2, trigger_ratio=0.7)
    agent = Agent(model, history_processors=[processor])

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='call tool')]),
        ModelResponse(
            parts=[TextPart(content='tool result')],
            usage=RequestUsage(input_tokens=90_000, output_tokens=100),
        ),
        ModelRequest(parts=[ToolReturnPart(tool_name='my_tool', content='long tool output data', tool_call_id='1')]),
        ModelResponse(
            parts=[TextPart(content='processed')],
            usage=RequestUsage(input_tokens=95_000, output_tokens=200),
        ),
    ]

    with capture_run_messages() as captured:
        result = await agent.run('next question', message_history=message_history)

    # 95k/128k > 0.7 → tool return should be masked
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='call tool', timestamp=IsDatetime())],
            ),
            ModelResponse(
                parts=[TextPart(content='tool result')],
                usage=RequestUsage(input_tokens=90_000, output_tokens=100),
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[ToolReturnPart(tool_name='my_tool', content='[compacted]', tool_call_id='1', timestamp=IsDatetime())],
            ),
            ModelResponse(
                parts=[TextPart(content='processed')],
                usage=RequestUsage(input_tokens=95_000, output_tokens=200),
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='next question', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='response')],
                usage=RequestUsage(input_tokens=90_000, output_tokens=500),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )


async def test_masking_processor_no_trigger_below_ratio():
    """ObservationMaskingProcessor does NOT trigger when below ratio threshold."""
    profile = ModelProfile(context_window=1_000_000)  # 1M context — usage is well below 70%

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content='response')],
            usage=RequestUsage(input_tokens=90_000, output_tokens=500),
        )

    model = FunctionModel(model_fn, profile=profile)
    processor = ObservationMaskingProcessor(keep_last=2, trigger_ratio=0.7)
    agent = Agent(model, history_processors=[processor])

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='call tool')]),
        ModelResponse(
            parts=[TextPart(content='tool result')],
            usage=RequestUsage(input_tokens=90_000, output_tokens=100),
        ),
        ModelRequest(parts=[ToolReturnPart(tool_name='my_tool', content='long tool output data', tool_call_id='1')]),
        ModelResponse(
            parts=[TextPart(content='processed')],
            usage=RequestUsage(input_tokens=95_000, output_tokens=200),
        ),
    ]

    with capture_run_messages() as captured:
        result = await agent.run('next question', message_history=message_history)

    # 95k/1M = 9.5% < 0.7 → tool return should NOT be masked
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='call tool', timestamp=IsDatetime())],
            ),
            ModelResponse(
                parts=[TextPart(content='tool result')],
                usage=RequestUsage(input_tokens=90_000, output_tokens=100),
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[ToolReturnPart(tool_name='my_tool', content='long tool output data', tool_call_id='1', timestamp=IsDatetime())],
            ),
            ModelResponse(
                parts=[TextPart(content='processed')],
                usage=RequestUsage(input_tokens=95_000, output_tokens=200),
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='next question', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='response')],
                usage=RequestUsage(input_tokens=90_000, output_tokens=500),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )


async def test_masking_processor_fallback_message_count():
    """ObservationMaskingProcessor falls back to message count when context_window is None."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='response')])

    model = FunctionModel(model_fn)
    processor = ObservationMaskingProcessor(keep_last=2)
    agent = Agent(model, history_processors=[processor])

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[ToolReturnPart(tool_name='t', content='old data', tool_call_id='1')]),
        ModelResponse(parts=[TextPart(content='r1')]),
        ModelRequest(parts=[UserPromptPart(content='q2')]),
        ModelResponse(parts=[TextPart(content='r2')]),
    ]

    with capture_run_messages() as captured:
        result = await agent.run('q3', message_history=message_history)

    # 4 messages > keep_last=2 → first message's ToolReturnPart should be masked
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[ToolReturnPart(tool_name='t', content='[compacted]', tool_call_id='1', timestamp=IsDatetime())],
            ),
            ModelResponse(
                parts=[TextPart(content='r1')],
                usage=RequestUsage(),
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='q2', timestamp=IsDatetime())],
            ),
            ModelResponse(
                parts=[TextPart(content='r2')],
                usage=RequestUsage(),
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='q3', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='response')],
                usage=RequestUsage(input_tokens=53, output_tokens=3),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )
