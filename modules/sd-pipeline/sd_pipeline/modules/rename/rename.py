from typing import Dict

from PIL.Image import Image

from sd_pipeline_typing.types import Module
from .config import RenameConfig


class Rename(Module):
    config: RenameConfig

    def __init__(self, *, config: RenameConfig):
        self.config = config

    def run(self, input_data: Dict[str, str | Image], _) -> Dict[str, str | Image]:
        name = input_data["name"]
        new_name = self.config.rename_function(name)
        input_data["name"] = new_name

        return input_data
