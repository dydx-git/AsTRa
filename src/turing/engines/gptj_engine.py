from pathlib import Path
from typing import Optional, Union

from turing.engines.causal import CausalEngine, CausalLoraEngine


class GPTJEngine(CausalEngine):
    config_name: str = "gptj_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__("EleutherAI/gpt-j-6B", weights_path)


class GPTJLoraEngine(CausalLoraEngine):
    config_name: str = "gptj_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__("EleutherAI/gpt-j-6B", weights_path)
