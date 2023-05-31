from typing import Optional
from xturing.engines.t5_engine import FlanLargeLoraEngine, FlanSmallLoraEngine, FlanUl2LoraEngine
from .causal import CausalLoraModel

class FlanSmallLora(CausalLoraModel):
    config_name: str = "google/flan-t5-small"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(FlanSmallLoraEngine.config_name, weights_path)

class FlanLargeLora(CausalLoraModel):
    config_name: str = "google/flan-t5-xxl"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(FlanLargeLoraEngine.config_name, weights_path)

class FlanUl2Lora(CausalLoraModel):
    config_name: str = "google/flan-ul2"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(FlanUl2LoraEngine.config_name, weights_path)