from typing import TypedDict

from pydantic import JsonValue

ModelID = str  # needs to have :, validate against KnownModelName?

# class InstructionsCapabilitySpec(TypedDict):
#     instructions: str

# class ThinkingCapabilitySpecWithLevel(TypedDict):
#     thinking: Literal['low', 'medium', 'high']

# ThinkingCapabilitySpec = Literal['thinking'] | ThinkingCapabilitySpecWithLevel

# execution_environment_capability_spec = TypedDict(
#     'execution_environment_capability_spec',
#     {'execution_environment': ExecutionEnvironmentSpec},  # pyright: ignore[reportInvalidTypeForm]
# )

# CapabilitySpec = InstructionsCapabilitySpec | ThinkingCapabilitySpec | execution_environment_capability_spec


class AgentSpec(TypedDict):
    model: ModelID
    # capabilities: list[CapabilitySpec]
    output: JsonValue
