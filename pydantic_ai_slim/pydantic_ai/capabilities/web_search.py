from collections.abc import Sequence
from dataclasses import replace

from pydantic_ai.builtin_tools import AbstractBuiltinTool, WebSearchTool
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.tools import AgentDepsT, BuiltinToolFunc
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

from .abstract import CAPABILITY_TYPES, AbstractCapability

_BUILTIN_WEB_SEARCH_TOOL = WebSearchTool()


class WebSearch(AbstractCapability[AgentDepsT]):
    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        return FunctionToolset([duckduckgo_search_tool()]).prepared(
            lambda ctx, tool_defs: [
                replace(tool_def, prefers_builtin=_BUILTIN_WEB_SEARCH_TOOL.unique_id) for tool_def in tool_defs
            ],
        )

    def get_builtin_tools(self) -> Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]]:
        return [_BUILTIN_WEB_SEARCH_TOOL]


CAPABILITY_TYPES['web_search'] = WebSearch
