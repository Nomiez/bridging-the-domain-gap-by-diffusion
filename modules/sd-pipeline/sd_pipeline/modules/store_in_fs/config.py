from sd_pipeline_typing.types import Config

class SIFSConfig(Config):
    def __init__(self, *, output_dir: str):
        self.output_dir = output_dir

