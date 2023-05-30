from typing import Optional
from xturing.engines.t5_engine import FlanSmallLoraEngine
from .causal import CausalLoraModel

class FlanLora(CausalLoraModel):
    config_name: str = "flan_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(FlanSmallLoraEngine.config_name, weights_path)