"""Tests for common tools."""

import httpx
import pytest

from pydantic_ai.common_tools.web_fetch import WebFetchTool, web_fetch_tool


def _mock_transport(handler):
    """Create an httpx.MockTransport from a handler function."""
    return httpx.MockTransport(handler)


def _html_response(body: str, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, html=body)


def _text_response(body: str, content_type: str = 'text/plain', status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, text=body, headers={'content-type': content_type})


class TestWebFetchTool:
    async def test_html_to_markdown(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return _html_response('<html><body><h1>Hello</h1><p>World</p></body></html>')

        client = httpx.AsyncClient(transport=_mock_transport(handler))
        tool = WebFetchTool(client=client, max_content_length=50_000)
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

        client = httpx.AsyncClient(transport=_mock_transport(handler))
        tool = WebFetchTool(client=client, max_content_length=100)
        result = await tool('http://example.com')

        assert isinstance(result, dict)
        assert result['content'].endswith('... Content truncated at 100 characters.')
        assert len(result['content']) < len(long_content)

    async def test_head_only(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == 'HEAD'
            return httpx.Response(200, headers={'content-type': 'text/html', 'content-length': '12345'})

        client = httpx.AsyncClient(transport=_mock_transport(handler))
        tool = WebFetchTool(client=client, max_content_length=50_000)
        result = await tool('http://example.com', head_only=True)

        assert isinstance(result, dict)
        assert result['status_code'] == 200
        assert result['content_type'] == 'text/html'
        assert result['content_length'] == '12345'
        assert 'content' not in result

    async def test_plain_text(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return _text_response('Just plain text.')

        client = httpx.AsyncClient(transport=_mock_transport(handler))
        tool = WebFetchTool(client=client, max_content_length=50_000)
        result = await tool('http://example.com')

        assert isinstance(result, dict)
        assert result['content'] == 'Just plain text.'
        assert result['content_type'] == 'text/plain'

    async def test_json_content(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={'key': 'value'})

        client = httpx.AsyncClient(transport=_mock_transport(handler))
        tool = WebFetchTool(client=client, max_content_length=50_000)
        result = await tool('http://example.com/api')

        assert isinstance(result, dict)
        assert result['status_code'] == 200
        assert 'json' in result['content_type']
        assert 'key' in result['content']

    async def test_http_error_status(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return _text_response('Not Found', status_code=404)

        client = httpx.AsyncClient(transport=_mock_transport(handler))
        tool = WebFetchTool(client=client, max_content_length=50_000)
        result = await tool('http://example.com/missing')

        assert isinstance(result, dict)
        assert result['status_code'] == 404
        assert result['content'] == 'Not Found'

    async def test_connection_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError('Connection refused')

        client = httpx.AsyncClient(transport=_mock_transport(handler))
        tool = WebFetchTool(client=client, max_content_length=50_000)
        result = await tool('http://unreachable.example.com')

        assert isinstance(result, dict)
        assert 'error' in result
        assert 'ConnectError' in result['error']

    async def test_custom_httpx_client(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers.get('x-custom') == 'test'
            return _text_response('OK')

        client = httpx.AsyncClient(transport=_mock_transport(handler), headers={'x-custom': 'test'})
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

        client = httpx.AsyncClient(transport=_mock_transport(handler))
        tool = WebFetchTool(client=client, max_content_length=50_000)
        result = await tool('http://example.com/old')

        assert isinstance(result, dict)
        assert result['url'] == 'http://example.com/new'
        assert result['content'] == 'Redirected content'

    async def test_html_strips_scripts_and_styles(self):
        html = '<html><head><style>body{color:red}</style></head><body><script>alert(1)</script><p>Content</p></body></html>'

        def handler(request: httpx.Request) -> httpx.Response:
            return _html_response(html)

        client = httpx.AsyncClient(transport=_mock_transport(handler))
        tool = WebFetchTool(client=client, max_content_length=50_000)
        result = await tool('http://example.com')

        assert isinstance(result, dict)
        assert 'alert' not in result['content']
        assert 'color:red' not in result['content']
        assert 'Content' in result['content']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
