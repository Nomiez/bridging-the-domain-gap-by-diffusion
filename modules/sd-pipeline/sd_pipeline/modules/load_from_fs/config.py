from sd_pipeline_typing.types import Config


class LFFSConfig(Config):
    def __init__(self, *, input_dir: str):
        self.input_dir = input_dir
