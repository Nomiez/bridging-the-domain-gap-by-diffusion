from typing import Callable

from sd_pipeline_typing.types import Config


class RenameConfig(Config):
    def __init__(self, *, rename_function: Callable):
        self.rename_function = rename_function
