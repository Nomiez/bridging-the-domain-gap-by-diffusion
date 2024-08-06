from __future__ import annotations

import sys

from PIL import Image
from pathlib import Path
import importlib.util

from sd_pipeline_typing.types import Module

from .config import CarlaConfig


class Carla(Module):
    def __init__(self, *, config: CarlaConfig):
        self.config = config

    @staticmethod
    def _import_class_from_file(file_path: str, class_name: str) -> type:
        file_path = Path(file_path)
        module_name = file_path.stem

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        return getattr(module, class_name)

    def run(
        self, input_data: dict[str, str | Image.Image], pipeline_config
    ) -> dict[str, str | Image.Image]:
        path = Path(self.config.scripts_dir)

        for script in path.glob("*.py"):
            module_name = script.stem
            class_ = Carla._import_class_from_file(module_name, "CarlaScript")

            class_instance = class_(config=self.config)
            class_instance.pre()
            res = class_instance.run_script(input_data=input_data, pipeline_config=pipeline_config)
            class_instance.post()
            return res
