from __future__ import annotations

from dataclasses import dataclass

import httpx
from typing_extensions import Any, TypedDict

from pydantic_ai.tools import Tool

try:
    from bs4 import BeautifulSoup
    from markdownify import markdownify
except ImportError as _import_error:
    raise ImportError(
        'Please install `beautifulsoup4` and `markdownify` to use the web fetch tool, '
        'you can use the `web-fetch` optional group — `pip install "pydantic-ai-slim[web-fetch]"`'
    ) from _import_error

__all__ = ('web_fetch_tool',)

_DEFAULT_MAX_CONTENT_LENGTH = 50_000


class WebFetchResult(TypedDict):
    """The result of fetching a web page."""

    url: str
    """The final URL after any redirects."""
    status_code: int
    """The HTTP status code."""
    content_type: str
    """The content type of the response."""
    content: str
    """The content of the response, converted to markdown for HTML pages."""


@dataclass
class WebFetchTool:
    """A tool that fetches web page content."""

    client: httpx.AsyncClient
    """The HTTP client to use for requests."""

    max_content_length: int
    """The maximum length of content to return."""

    async def __call__(self, url: str, head_only: bool = False) -> WebFetchResult | dict[str, Any]:
        """Fetches the content of a web page.

        For HTML pages, the content is converted to markdown for easier reading.

        Args:
            url: The URL to fetch.
            head_only: If True, only fetch headers (HEAD request) to check metadata without downloading content.

        Returns:
            The fetched content or metadata.
        """
        try:
            if head_only:
                response = await self.client.head(url, follow_redirects=True)
                return {
                    'url': str(response.url),
                    'status_code': response.status_code,
                    'content_type': response.headers.get('content-type', ''),
                    'content_length': response.headers.get('content-length'),
                }

            response = await self.client.get(url, follow_redirects=True)
        except httpx.HTTPError as e:
            return {'error': f'{type(e).__name__}: {e}'}

        content_type = response.headers.get('content-type', '')
        text = response.text

        if 'html' in content_type:
            soup = BeautifulSoup(text, 'html.parser')
            for tag in soup(['script', 'style', 'img']):
                tag.decompose()
            text = markdownify(str(soup))

        if len(text) > self.max_content_length:
            text = (
                text[: self.max_content_length]
                + f'\n\n... Content truncated at {self.max_content_length:,} characters.'
            )

        return WebFetchResult(
            url=str(response.url),
            status_code=response.status_code,
            content_type=content_type,
            content=text,
        )


def web_fetch_tool(
    *,
    max_content_length: int = _DEFAULT_MAX_CONTENT_LENGTH,
    httpx_client: httpx.AsyncClient | None = None,
) -> Tool[Any]:
    """Creates a web fetch tool that can retrieve and read web page content.

    HTML content is automatically converted to markdown for easier consumption by LLMs.

    Args:
        max_content_length: The maximum length of content to return. Defaults to 50,000 characters.
        httpx_client: An optional custom `httpx.AsyncClient` to use for requests.
    """
    client = httpx_client or httpx.AsyncClient()
    return Tool[Any](
        WebFetchTool(client=client, max_content_length=max_content_length).__call__,
        name='web_fetch',
        description='Fetches the content of a web page. For HTML pages, content is converted to markdown.',
    )
