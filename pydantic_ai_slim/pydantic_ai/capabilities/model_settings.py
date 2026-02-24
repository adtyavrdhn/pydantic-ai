from dataclasses import dataclass

from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings, merge_model_settings
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import CAPABILITY_TYPES, AbstractCapability


@dataclass
class ModelSettingsCapability(AbstractCapability[AgentDepsT]):
    settings: ModelSettings

    # TODO: Restore get_model_settings() method

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        messages: list[ModelMessage],
        model_settings: ModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[list[ModelMessage], ModelSettings, ModelRequestParameters]:
        model_settings = merge_model_settings(model_settings, self.settings) or self.settings
        return messages, model_settings, model_request_parameters


CAPABILITY_TYPES['model_settings'] = ModelSettingsCapability
