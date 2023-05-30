from abc import ABC, abstractmethod
from xturing.registry import BaseParent

class TextGenerationAPI(ABC):
    config_name = "text_generation"

    def __init__(self, engine, api_key, request_batch_size=1, organization=None):
        self.engine = engine
        self.api_key = api_key
        self.organization = organization
        self.request_batch_size = request_batch_size

    @abstractmethod
    def generate_text(self, *args, **kwargs):
        raise NotImplementedError(
            "generate_text method should be implemented in the child class, use specific api"
        )
