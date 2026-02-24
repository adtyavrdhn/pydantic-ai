from dataclasses import dataclass

from pydantic_ai import _instructions
from pydantic_ai.capabilities.abstract import CAPABILITY_TYPES, AbstractCapability
from pydantic_ai.tools import AgentDepsT


@dataclass
class Instructions(AbstractCapability[AgentDepsT]):
    instructions: _instructions.Instructions[AgentDepsT]

    def get_instructions(self) -> _instructions.Instructions[AgentDepsT] | None:
        return self.instructions


CAPABILITY_TYPES['instructions'] = Instructions
