"""Tests for HTTP response caching with hishel."""

from __future__ import annotations

import json

import httpx
import pytest

from .conftest import try_import

with try_import() as imports_successful:
    import hishel
    from hishel.httpx import AsyncCacheTransport

    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='hishel or openai not installed'),
    pytest.mark.anyio,
]


def create_mock_openai_response(content: str = 'Hello!') -> httpx.Response:
    """Create a mock OpenAI chat completion response."""
    response_json = {
        'id': 'chatcmpl-123',
        'object': 'chat.completion',
        'created': 1704067200,
        'model': 'gpt-4o',
        'choices': [
            {
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': content,
                },
                'finish_reason': 'stop',
            }
        ],
        'usage': {
            'prompt_tokens': 10,
            'completion_tokens': 5,
            'total_tokens': 15,
        },
    }
    return httpx.Response(
        200,
        content=json.dumps(response_json),
        headers={
            'content-type': 'application/json',
            'cache-control': 'max-age=3600',
        },
    )


class CountingTransport(httpx.AsyncBaseTransport):
    """A transport that counts requests and returns mock responses."""

    def __init__(self) -> None:
        self.request_count = 0
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.request_count += 1
        self.requests.append(request)
        return create_mock_openai_response()


async def test_hishel_caching_basic(allow_model_requests: None):
    """Test that identical requests are served from cache."""
    counting_transport = CountingTransport()

    # Use FilterPolicy for LLM API caching - SpecificationPolicy follows RFC 9111
    # which mandates that POST requests always go to origin (never served from cache)
    policy = hishel.FilterPolicy()
    policy.use_body_key = True  # Different prompts = different cache keys

    # Wrap with hishel caching
    cache_transport = AsyncCacheTransport(
        next_transport=counting_transport,
        storage=hishel.AsyncSqliteStorage(),
        policy=policy,
    )

    async with httpx.AsyncClient(transport=cache_transport) as http_client:
        provider = OpenAIProvider(http_client=http_client, api_key='test-key')
        model = OpenAIChatModel('gpt-4o', provider=provider)
        agent = Agent(model)

        # First call - should hit the counting transport
        result1 = await agent.run('Hello')
        assert result1.output == 'Hello!'
        assert counting_transport.request_count == 1

        # Second identical call - should be served from cache
        result2 = await agent.run('Hello')
        assert result2.output == 'Hello!'
        assert counting_transport.request_count == 1  # Still 1, not 2!

        # Different prompt - should hit transport again (cache miss)
        result3 = await agent.run('Goodbye')
        assert result3.output == 'Hello!'  # Same mock response
        assert counting_transport.request_count == 2