from typing import Optional
from xturing.engines.mpt_engine import MPTEngine
from .causal import CausalModel

class Mpt(CausalModel):
    config_name: str = "mpt"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(MPTEngine.config_name, weights_path)