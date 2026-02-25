"""Tests for common tools."""

from collections.abc import Callable
from typing import TypeAlias
from unittest.mock import patch

import httpx
import pytest

from pydantic_ai.common_tools.web_fetch import WebFetchTool, web_fetch_tool

_Handler: TypeAlias = Callable[[httpx.Request], httpx.Response]


def _html_response(body: str, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, html=body)


def _text_response(body: str, content_type: str = 'text/plain', status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, text=body, headers={'content-type': content_type})


def _make_tool(handler: _Handler, *, max_content_length: int = 50_000, allow_private_urls: bool = True) -> WebFetchTool:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return WebFetchTool(client=client, max_content_length=max_content_length, allow_private_urls=allow_private_urls)


class TestWebFetchTool:
    async def test_html_to_markdown(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return _html_response('<html><body><h1>Hello</h1><p>World</p></body></html>')

        tool = _make_tool(handler)
        result = await tool('http://example.com')

        assert isinstance(result, dict)
        assert result['status_code'] == 200
        assert 'Hello' in result['content']
        assert 'World' in result['content']
        assert 'text/html' in result['content_type']

    async def test_content_truncation(self):
        long_content = '<html><body>' + 'a' * 1000 + '</body></html>'

        def handler(request: httpx.Request) -> httpx.Response:
            return _html_response(long_content)

        tool = _make_tool(handler, max_content_length=100)
        result = await tool('http://example.com')

        assert isinstance(result, dict)
        assert result['content'].endswith('... Content truncated at 100 characters.')
        assert len(result['content']) < len(long_content)

    async def test_head_only(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == 'HEAD'
            return httpx.Response(200, headers={'content-type': 'text/html', 'content-length': '12345'})

        tool = _make_tool(handler)
        result = await tool('http://example.com', head_only=True)

        assert isinstance(result, dict)
        assert result.get('status_code') == 200
        assert result.get('content_type') == 'text/html'
        assert result.get('content_length') == '12345'
        assert 'content' not in result

    async def test_plain_text(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return _text_response('Just plain text.')

        tool = _make_tool(handler)
        result = await tool('http://example.com')

        assert isinstance(result, dict)
        assert result['content'] == 'Just plain text.'
        assert result['content_type'] == 'text/plain'

    async def test_json_content(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={'key': 'value'})

        tool = _make_tool(handler)
        result = await tool('http://example.com/api')

        assert isinstance(result, dict)
        assert result['status_code'] == 200
        assert 'json' in result['content_type']
        assert 'key' in result['content']

    async def test_http_error_status(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return _text_response('Not Found', status_code=404)

        tool = _make_tool(handler)
        result = await tool('http://example.com/missing')

        assert isinstance(result, dict)
        assert result['status_code'] == 404
        assert result['content'] == 'Not Found'

    async def test_connection_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError('Connection refused')

        tool = _make_tool(handler)
        result = await tool('http://unreachable.example.com')

        assert isinstance(result, dict)
        assert 'error' in result
        assert 'ConnectError' in result['error']

    async def test_custom_httpx_client(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers.get('x-custom') == 'test'
            return _text_response('OK')

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), headers={'x-custom': 'test'})
        tool = web_fetch_tool(httpx_client=client)

        assert tool.name == 'web_fetch'

    async def test_factory_defaults(self):
        tool = web_fetch_tool()
        assert tool.name == 'web_fetch'
        assert (
            tool.description == 'Fetches the content of a web page. For HTML pages, content is converted to markdown.'
        )

    async def test_follow_redirects(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url) == 'http://example.com/old':
                return httpx.Response(301, headers={'location': 'http://example.com/new'})
            return _text_response('Redirected content')

        tool = _make_tool(handler)
        result = await tool('http://example.com/old')

        assert isinstance(result, dict)
        assert result['url'] == 'http://example.com/new'
        assert result['content'] == 'Redirected content'

    async def test_html_strips_scripts_and_styles(self):
        html = '<html><head><style>body{color:red}</style></head><body><script>alert(1)</script><p>Content</p></body></html>'

        def handler(request: httpx.Request) -> httpx.Response:
            return _html_response(html)

        tool = _make_tool(handler)
        result = await tool('http://example.com')

        assert isinstance(result, dict)
        assert 'alert' not in result['content']
        assert 'color:red' not in result['content']
        assert 'Content' in result['content']


class TestSSRFProtection:
    """Tests for SSRF (Server-Side Request Forgery) protection."""

    def _noop_handler(self, request: httpx.Request) -> httpx.Response:
        raise AssertionError('Request should have been blocked before reaching the transport')

    def _make_protected_tool(self) -> WebFetchTool:
        return _make_tool(self._noop_handler, allow_private_urls=False)

    @pytest.mark.parametrize(
        'url',
        [
            'http://localhost/secret',
            'http://localhost:8080/admin',
            'http://127.0.0.1/secret',
            'http://127.0.0.1:3000/',
            'http://0.0.0.0/',
            'http://[::1]/',
            'http://10.0.0.1/internal',
            'http://172.16.0.1/internal',
            'http://192.168.1.1/admin',
            'http://169.254.169.254/latest/meta-data/',
            'http://metadata.google.internal/computeMetadata/v1/',
        ],
    )
    async def test_blocks_private_urls(self, url: str):
        tool = self._make_protected_tool()
        result = await tool(url)

        assert isinstance(result, dict)
        assert 'error' in result
        assert 'not allowed' in result['error']

    async def test_blocks_ftp_scheme(self):
        tool = self._make_protected_tool()
        result = await tool('ftp://example.com/file')

        assert isinstance(result, dict)
        assert 'error' in result
        assert 'not allowed' in result['error']
        assert 'ftp' in result['error']

    async def test_blocks_file_scheme(self):
        tool = self._make_protected_tool()
        result = await tool('file:///etc/passwd')

        assert isinstance(result, dict)
        assert 'error' in result
        assert 'not allowed' in result['error']

    async def test_allows_public_urls(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return _text_response('public content')

        tool = _make_tool(handler, allow_private_urls=False)

        with patch('pydantic_ai.common_tools.web_fetch.socket.getaddrinfo') as mock_dns:
            mock_dns.return_value = [(2, 1, 6, '', ('93.184.216.34', 0))]
            result = await tool('http://example.com')

        assert isinstance(result, dict)
        assert result['content'] == 'public content'

    async def test_blocks_dns_rebinding_to_private_ip(self):
        """A hostname that resolves to a private IP should be blocked."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError('Request should have been blocked')

        tool = _make_tool(handler, allow_private_urls=False)

        with patch('pydantic_ai.common_tools.web_fetch.socket.getaddrinfo') as mock_dns:
            mock_dns.return_value = [(2, 1, 6, '', ('127.0.0.1', 0))]
            result = await tool('http://evil.example.com')

        assert isinstance(result, dict)
        assert 'error' in result
        assert 'private/internal IP' in result['error']

    async def test_blocks_dns_rebinding_to_metadata_ip(self):
        """A hostname that resolves to a cloud metadata IP should be blocked."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError('Request should have been blocked')

        tool = _make_tool(handler, allow_private_urls=False)

        with patch('pydantic_ai.common_tools.web_fetch.socket.getaddrinfo') as mock_dns:
            mock_dns.return_value = [(2, 1, 6, '', ('169.254.169.254', 0))]
            result = await tool('http://sneaky.example.com')

        assert isinstance(result, dict)
        assert 'error' in result
        assert 'private/internal IP' in result['error']

    async def test_allow_private_urls_flag(self):
        """When allow_private_urls=True, private URLs should be permitted."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _text_response('internal content')

        tool = _make_tool(handler, allow_private_urls=True)
        result = await tool('http://localhost/admin')

        assert isinstance(result, dict)
        assert result['content'] == 'internal content'

    async def test_factory_default_blocks_private(self):
        """The factory function should block private URLs by default."""
        protected_tool = _make_tool(self._noop_handler, allow_private_urls=False)
        result = await protected_tool('http://127.0.0.1/')

        assert isinstance(result, dict)
        assert 'error' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
