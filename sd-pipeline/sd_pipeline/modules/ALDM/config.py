from ...components.config import Config


class ALDMConfig(Config):
    def __init__(self, *, prompt: str, neg: bool, output_dir: str, num_samples: int, ddim_steps: int, cfg_scale: float, seed: int):
        self.prompt = prompt
        self.neg = neg
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.ddim_steps = ddim_steps
        self.cfg_scale = cfg_scale
        self.seed = seed
