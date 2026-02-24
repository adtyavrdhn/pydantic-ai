from typing import cast

from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT

from .abstract import CAPABILITY_TYPES
from .model_settings import ModelSettingsCapability


class Thinking(ModelSettingsCapability[AgentDepsT]):
    def __init__(self):
        super().__init__(
            cast(
                ModelSettings,
                {
                    'openai_reasoning_effort': 'high',
                    'anthropic_thinking': {'type': 'adaptive'},
                    # etc
                },
            ),
        )


CAPABILITY_TYPES['thinking'] = Thinking
