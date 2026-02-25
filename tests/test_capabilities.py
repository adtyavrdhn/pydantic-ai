import pytest

from pydantic_ai.agent import Agent
from pydantic_ai.capabilities import (
    CAPABILITY_TYPES,
    ExecutionEnvironment,
    HistoryProcessorCapability,
    Instructions,
    ModelSettingsCapability,
    Thinking,
    Toolset,
    WebSearch,
)
from pydantic_ai.environments.memory import MemoryEnvironment
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsStr

pytestmark = [
    pytest.mark.anyio,
]


def test_capability_types() -> None:
    assert CAPABILITY_TYPES == snapshot(
        {
            'execution_environment': ExecutionEnvironment,
            'history_processor': HistoryProcessorCapability,
            'instructions': Instructions,
            'model_settings': ModelSettingsCapability,
            'thinking': Thinking,
            'toolset': Toolset,
            'web_search': WebSearch,
        }
    )


@pytest.fixture
def memory_environment() -> MemoryEnvironment:
    return MemoryEnvironment(files={'hello.py': 'print("hello")'})


async def test_agent(anthropic_api_key: str, allow_model_requests: None, memory_environment: MemoryEnvironment):
    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[
            Instructions("You are 'JAK of all trades', a General Agent for Knowledge built with Pydantic AI."),
            Thinking(),
            ExecutionEnvironment('local', include=['ls', 'read_file']),
            WebSearch(),
        ],
    )
    result = await agent.run('Whats in lines 27-34 of the readme')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Whats in lines 27-34 of the readme', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                instructions="You are 'JAK of all trades', a General Agent for Knowledge built with Pydantic AI.",
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user wants to read lines 27-34 of a readme file. Let me look for it in the current directory first.',
                        signature=IsStr(),
                        provider_name='anthropic',
                    ),
                    TextPart(content='Sure! Let me look for the README file first!'),
                    ToolCallPart(tool_name='ls', args={}, tool_call_id=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=2506,
                    output_tokens=88,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 2506,
                        'output_tokens': 88,
                    },
                ),
                model_name='claude-sonnet-4-6',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id=IsStr(),
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='ls',
                        content="""\
.cache/
.claude/
.coverage/
.env (2538 bytes)
.gemini/
.git/
.github/
.gitignore (432 bytes)
.logfire/
.pre-commit-config.yaml (2082 bytes)
.pytest_cache/
.ruff_cache/
.venv/
.vscode/
AGENTS.md (20683 bytes)
CLAUDE.md (20683 bytes)
LICENSE (1100 bytes)
Makefile (5396 bytes)
README.md (11667 bytes)
clai/
docs/
docs-site/
examples/
mkdocs.yml (12666 bytes)
pydantic_ai_slim/
pydantic_evals/
pydantic_graph/
pyproject.toml (12355 bytes)
scratch/
scripts/
tests/
uv.lock (1917730 bytes)\
""",
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="You are 'JAK of all trades', a General Agent for Knowledge built with Pydantic AI.",
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(content='Found it! Now let me read lines 27–34 of `README.md`.'),
                    ToolCallPart(
                        tool_name='read_file',
                        args={'path': 'README.md', 'offset': 26, 'limit': 8},
                        tool_call_id='toolu_01NLst65QFMKKaRdFEvE7GSq',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=2824,
                    output_tokens=109,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 2824,
                        'output_tokens': 109,
                    },
                ),
                model_name='claude-sonnet-4-6',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id=IsStr(),
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='read_file',
                        content="""\
    27	### <em>Pydantic AI is a Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with Generative AI.</em>
    28
    29
    30	FastAPI revolutionized web development by offering an innovative and ergonomic design, built on the foundation of [Pydantic Validation](https://docs.pydantic.dev) and modern Python features like type hints.
    31
    32	Yet despite virtually every Python agent framework and LLM library using Pydantic Validation, when we began to use LLMs in [Pydantic Logfire](https://pydantic.dev/logfire), we couldn't find anything that gave us the same feeling.
    33
    34	We built Pydantic AI with one simple aim: to bring that FastAPI feeling to GenAI app and agent development.
... (167 more lines. Use offset=34 to continue reading.)
""",
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="You are 'JAK of all trades', a General Agent for Knowledge built with Pydantic AI.",
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
Here are lines **27–34** of `README.md`:

- **Line 27:** A tagline in italics — *"Pydantic AI is a Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with Generative AI."*
- **Line 28–29:** Blank lines.
- **Line 30:** A comparison to FastAPI — noting how it revolutionized web development through its ergonomic design, built on [Pydantic Validation](https://docs.pydantic.dev) and modern Python features like type hints.
- **Line 32:** An observation that despite virtually every Python agent framework using Pydantic Validation, the team couldn't find anything that gave them the same feeling when building [Pydantic Logfire](https://pydantic.dev/logfire).
- **Line 34:** The mission statement — *"We built Pydantic AI with one simple aim: to bring that FastAPI feeling to GenAI app and agent development."*\
"""
                    )
                ],
                usage=RequestUsage(
                    input_tokens=3170,
                    output_tokens=236,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 3170,
                        'output_tokens': 236,
                    },
                ),
                model_name='claude-sonnet-4-6',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


# async def test_agent_from_spec(
#     anthropic_api_key: str, allow_model_requests: None, memory_environment: MemoryEnvironment
# ):
#     agent = Agent.from_spec(
#         {
#             'model': 'anthropic:claude-sonnet-4-6',
#             'capabilities': [
#                 {'instructions': "You are 'JAK of all trades', a General Agent for Knowledge built with Pydantic AI."},
#                 'thinking',
#                 {'execution_environment': {'environment': 'local', 'include': ['ls', 'read_file']}},
#                 'web_search',
#             ],
#         }
#     )
