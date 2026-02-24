from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

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

_BLOCKED_HOSTNAMES = frozenset(
    {
        'localhost',
        'metadata.google.internal',
    }
)


def _is_private_ip(host: str) -> bool:
    """Check if a host string is a private/reserved IP address."""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return addr.is_private or addr.is_reserved or addr.is_loopback or addr.is_link_local


def _validate_url(url: str) -> str | None:
    """Validate that a URL doesn't target internal/private resources.

    Returns an error message if the URL is blocked, or None if it's safe.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return 'Invalid URL'

    if parsed.scheme not in ('http', 'https'):
        return f'URL scheme {parsed.scheme!r} is not allowed, only http and https are supported'

    hostname = parsed.hostname
    if not hostname:
        return 'URL has no hostname'

    if hostname in _BLOCKED_HOSTNAMES:
        return f'Fetching {hostname!r} is not allowed'

    if _is_private_ip(hostname):
        return f'Fetching private/internal IP address {hostname!r} is not allowed'

    return None


def _resolve_and_validate_host(hostname: str) -> str | None:
    """Resolve a hostname via DNS and validate that it doesn't point to a private IP.

    Returns an error message if blocked, or None if safe.
    """
    try:
        results = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        # Let httpx handle DNS failures naturally
        return None

    for _, _, _, _, sockaddr in results:
        ip = str(sockaddr[0])
        if _is_private_ip(ip):
            return f'Hostname {hostname!r} resolves to private/internal IP address {ip!r} and is not allowed'

    return None


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

    allow_private_urls: bool
    """Whether to allow fetching private/internal URLs."""

    async def __call__(self, url: str, head_only: bool = False) -> WebFetchResult | dict[str, Any]:
        """Fetches the content of a web page.

        For HTML pages, the content is converted to markdown for easier reading.

        Args:
            url: The URL to fetch.
            head_only: If True, only fetch headers (HEAD request) to check metadata without downloading content.

        Returns:
            The fetched content or metadata.
        """
        if not self.allow_private_urls:
            if error := _validate_url(url):
                return {'error': error}

            hostname = urlparse(url).hostname
            if hostname and not _is_private_ip(hostname):
                if error := _resolve_and_validate_host(hostname):
                    return {'error': error}

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
    allow_private_urls: bool = False,
) -> Tool[Any]:
    """Creates a web fetch tool that can retrieve and read web page content.

    HTML content is automatically converted to markdown for easier consumption by LLMs.

    By default, requests to private/internal IP addresses and hostnames (e.g. `localhost`,
    `127.0.0.1`, `169.254.169.254`) are blocked to prevent SSRF attacks. Set
    `allow_private_urls=True` to disable this protection.

    Args:
        max_content_length: The maximum length of content to return. Defaults to 50,000 characters.
        httpx_client: An optional custom `httpx.AsyncClient` to use for requests.
        allow_private_urls: Whether to allow fetching private/internal URLs. Defaults to `False`.
    """
    client = httpx_client or httpx.AsyncClient()
    return Tool[Any](
        WebFetchTool(
            client=client, max_content_length=max_content_length, allow_private_urls=allow_private_urls
        ).__call__,
        name='web_fetch',
        description='Fetches the content of a web page. For HTML pages, content is converted to markdown.',
    )
