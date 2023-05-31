from pathlib import Path
from typing import Optional, Union

from xturing.engines.seq2seq import Seq2SeqLoraEngine

class FlanSmallLoraEngine(Seq2SeqLoraEngine):
    config_name: str = "google/flan-t5-small_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="google/flan-t5-small",
            weights_path=weights_path,
            target_modules=["q", "v"],
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

class FlanLargeLoraEngine(Seq2SeqLoraEngine):
    config_name: str = "google/flan-t5-xxl_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="google/flan-t5-xxl",
            weights_path=weights_path,
            target_modules=["q", "v"],
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

class FlanUl2LoraEngine(Seq2SeqLoraEngine):
    config_name: str = "google/flan-ul2"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="google/flan-ul2",
            weights_path=weights_path,
            target_modules=["q", "v"],
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token