from dataclasses import dataclass

from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

from .abstract import CAPABILITY_TYPES, AbstractCapability


@dataclass
class Toolset(AbstractCapability[AgentDepsT]):
    toolset: AbstractToolset[AgentDepsT]

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        return self.toolset


CAPABILITY_TYPES['toolset'] = Toolset
