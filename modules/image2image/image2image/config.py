from sd_pipeline_typing.types import Config


class I2IConfig(Config):
    def __init__(self, *, prompt: str):
        self.prompt = prompt
