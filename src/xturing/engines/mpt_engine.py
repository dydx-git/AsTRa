import os
from pathlib import Path
from typing import Optional, Union

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from xturing.engines.causal import CausalEngine, CausalLoraEngine
from xturing.engines.gptj_utils.gptj import GPTJAttention
from xturing.engines.lora_engine import prepare_model_for_int8_training


class MPTEngine(CausalEngine):
    config_name: str = "mpt_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="k0t1k/mosaicml-mpt-7b-instruct-lora", weights_path=weights_path, trust_remote_code = True
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token