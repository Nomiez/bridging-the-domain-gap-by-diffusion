from sd_pipeline_typing.types import Config


class ALDMConfig(Config):
    def __init__(self, *, prompt: str, neg: bool, num_samples: int, ddim_steps: int, cfg_scale: float, seed: int, random_seed: bool = False, model: str = "cityscapes"):
        self.prompt = prompt
        self.neg = neg
        self.num_samples = num_samples
        self.ddim_steps = ddim_steps
        self.cfg_scale = cfg_scale
        self.seed = seed
        self.random_seed = random_seed
        self.model = model

