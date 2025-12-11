# HTTP Response Caching

Pydantic AI supports caching HTTP responses from model providers through custom HTTP clients.
This allows identical requests to return cached responses instantly, reducing latency and API costs.

!!! note "HTTP-level caching"
    This is caching at the HTTP request/response level, not at the agent run level.
    If the same HTTP request is made to a model provider, the cached response is returned.
    This is particularly useful when you have repeated identical prompts or message histories.

## Overview

The caching functionality is built on top of the [hishel](https://hishel.com/) library and integrates
seamlessly with httpx clients. You can configure caching behavior for any provider that accepts a custom HTTP client.

## Installation

To use caching, you need to install `hishel`, which you can do via the `caching` dependency group:

```bash
pip/uv-add 'pydantic-ai-slim[caching]'
```

## Usage Example

Here's an example of adding caching with SQLite storage:

```python {title="basic_cache_example.py"}
import hishel
import httpx
from hishel.httpx import AsyncCacheTransport

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


def create_caching_client() -> httpx.AsyncClient:
    """Create a client with HTTP response caching for LLM APIs."""
    # Use FilterPolicy for LLM API caching - it bypasses RFC 9111 restrictions
    # that would prevent POST requests from being served from cache
    policy = hishel.FilterPolicy()
    policy.use_body_key = True  # Different prompts = different cache keys

    transport = AsyncCacheTransport(
        next_transport=httpx.AsyncHTTPTransport(),
        storage=hishel.AsyncSqliteStorage(),
        policy=policy,
    )
    return httpx.AsyncClient(transport=transport)


# Use the caching client with a model
client = create_caching_client()
model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(http_client=client))
agent = Agent(model)
```

## Why FilterPolicy?

Hishel provides two caching policies:

- **`SpecificationPolicy`** (default): Follows HTTP caching RFC 9111 strictly. This means POST requests
  are *never* served from cache—they're always forwarded to the origin server. This is correct HTTP
  behavior but not useful for LLM API caching.

- **`FilterPolicy`**: Bypasses RFC 9111 restrictions and caches all requests. This is what you need
  for LLM APIs which use POST requests.

The `use_body_key = True` setting ensures that different request bodies (different prompts) get
different cache keys, so only truly identical requests return cached responses.

## Storage

Hishel uses SQLite for cache storage:

```python {title="sqlite_storage_example.py"}
import hishel

# Uses default path
storage = hishel.AsyncSqliteStorage()

# Or for synchronous usage
sync_storage = hishel.SyncSqliteStorage()
```

## Using with Different Providers

The caching transport works with any provider that accepts a custom HTTP client:

### OpenAI

```python {title="openai_with_cache.py" requires="basic_cache_example.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from basic_cache_example import create_caching_client

client = create_caching_client()
model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(http_client=client))
agent = Agent(model)
```

### Anthropic

```python {title="anthropic_with_cache.py" requires="basic_cache_example.py"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from basic_cache_example import create_caching_client

client = create_caching_client()
model = AnthropicModel('claude-sonnet-4-5-20250929', provider=AnthropicProvider(http_client=client))
agent = Agent(model)
```

### Groq

```python {title="groq_with_cache.py" requires="basic_cache_example.py"}
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

from basic_cache_example import create_caching_client

client = create_caching_client()
model = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(http_client=client))
agent = Agent(model)
```

## Cache Configuration

### Using Request Filters

You can filter which requests get cached using request filters:

```python {title="cache_config_example.py"}
import hishel
import httpx
from hishel import Request
from hishel.httpx import AsyncCacheTransport


class OnlyPostFilter(hishel.BaseFilter[Request]):
    """Only cache POST requests."""

    def needs_body(self) -> bool:
        return False

    def apply(self, item: Request, body: bytes | None) -> bool:
        return item.method.upper() == 'POST'


def create_configured_cache_client() -> httpx.AsyncClient:
    """Create a client with custom cache configuration."""
    policy = hishel.FilterPolicy(
        request_filters=[OnlyPostFilter()],
    )
    policy.use_body_key = True

    transport = AsyncCacheTransport(
        next_transport=httpx.AsyncHTTPTransport(),
        storage=hishel.AsyncSqliteStorage(),
        policy=policy,
    )
    return httpx.AsyncClient(transport=transport)
```

### Using Response Filters

You can also filter based on response properties:

```python {title="response_filter_example.py"}
import hishel
import httpx
from hishel import Response
from hishel.httpx import AsyncCacheTransport


class OnlySuccessFilter(hishel.BaseFilter[Response]):
    """Only cache successful responses."""

    def needs_body(self) -> bool:
        return False

    def apply(self, item: Response, body: bytes | None) -> bool:
        return item.status_code == 200


def create_success_only_cache_client() -> httpx.AsyncClient:
    """Create a client that only caches successful responses."""
    policy = hishel.FilterPolicy(
        response_filters=[OnlySuccessFilter()],
    )
    policy.use_body_key = True

    transport = AsyncCacheTransport(
        next_transport=httpx.AsyncHTTPTransport(),
        storage=hishel.AsyncSqliteStorage(),
        policy=policy,
    )
    return httpx.AsyncClient(transport=transport)
```

## Best Practices

1. **Use `FilterPolicy`**: The default `SpecificationPolicy` follows RFC 9111 which prevents POST
   requests from being served from cache. LLM APIs use POST requests.

2. **Enable `use_body_key`**: Set `policy.use_body_key = True` so different prompts get different
   cache keys.

3. **Consider cache invalidation**: Cached responses may become stale. Clear the cache database
   when needed.

4. **Monitor cache hits**: Track cache hit rates to measure effectiveness.

## Combining with Retries

You can combine caching with retry functionality by chaining transports:

```python {title="cache_with_retries.py"}
import hishel
import httpx
from hishel.httpx import AsyncCacheTransport


def create_cached_retrying_client() -> httpx.AsyncClient:
    """Create a client with both caching and retry functionality."""
    # Configure caching policy for LLM APIs
    policy = hishel.FilterPolicy()
    policy.use_body_key = True

    # Cache transport wraps the base transport
    cache_transport = AsyncCacheTransport(
        next_transport=httpx.AsyncHTTPTransport(),
        storage=hishel.AsyncSqliteStorage(),
        policy=policy,
    )

    # Use httpx's built-in transport with retry support
    return httpx.AsyncClient(transport=cache_transport)
```

For more advanced configurations, refer to the [hishel documentation](https://hishel.com/).
