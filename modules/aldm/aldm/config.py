from __future__ import annotations

from sd_pipeline_typing.types import Config


class ALDMConfig(Config):
    def __init__(
        self,
        *,
        prompt: str,
        neg_prompt: str,
        num_samples: int,
        ddim_steps: int,
        cfg_scale: float,
        seed: int,
        random_seed: bool = False,
        model: str = "cityscapes",
    ):
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.num_samples = num_samples
        self.ddim_steps = ddim_steps
        self.cfg_scale = cfg_scale
        self.seed = seed
        self.random_seed = random_seed
        self.model = model
