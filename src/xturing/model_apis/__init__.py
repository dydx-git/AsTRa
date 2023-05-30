from .ai21 import AI21TextGenerationAPI
from .ai21 import J2Grande as AI21J2Grande
from .base import BaseParent, TextGenerationAPI
from .cohere import CohereTextGenerationAPI
from .cohere import Medium as CohereMedium
from .openai import ChatGPT as OpenAIChatGPT
from .openai import Davinci as OpenAIDavinci
from .openai import OpenAITextGenerationAPI

BaseParent.add_to_registry(OpenAITextGenerationAPI.config_name, OpenAITextGenerationAPI)
BaseParent.add_to_registry(CohereTextGenerationAPI.config_name, CohereTextGenerationAPI)
BaseParent.add_to_registry(AI21TextGenerationAPI.config_name, AI21TextGenerationAPI)
BaseParent.add_to_registry(OpenAIDavinci.config_name, OpenAIDavinci)
BaseParent.add_to_registry(OpenAIChatGPT.config_name, OpenAIChatGPT)
BaseParent.add_to_registry(CohereMedium.config_name, CohereMedium)
BaseParent.add_to_registry(AI21J2Grande.config_name, AI21J2Grande)
