from __future__ import annotations

from sd_pipeline_typing.types import Config


class C2CConfig(Config):
    def __init__(self, *, output_dir_json):
        self.output_dir_json = output_dir_json
