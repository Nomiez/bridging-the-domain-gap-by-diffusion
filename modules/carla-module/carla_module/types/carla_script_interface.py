# Source: https://pythonprogramming.net/control-camera-sensor-self-driving-autonomous-cars-carla-python/

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import carla

from carla_module import CarlaConfig
from sd_pipeline_typing.types import PipelineConfig
from PIL.Image import Image


class CarlaScriptInterface(ABC):
    config: CarlaConfig
    actor_list: list[carla.Actor]
    state: dict[str, Any]

    def __init__(self, *, config: CarlaConfig):
        self.config = config
        self.actor_list = []
        self.state = {}

    def pre(self) -> None:
        try:
            client = carla.Client(self.config.hostname, self.config.port)
        except Exception as e:
            raise RuntimeError(f"Connection to Carla Server failed. Reason: {e}")

        client.set_timeout(2.0)

        world = client.get_world()

        blueprint_library = world.get_blueprint_library()

        self.state["client"] = client
        self.state["world"] = world
        self.state["blueprint_library"] = blueprint_library

    def post(self) -> None:
        for actor in self.actor_list:
            actor.destroy()

    @abstractmethod
    def run_script(
        self, input_data, pipeline_config: PipelineConfig
    ) -> dict[str, str | Image] | tuple[dict[str, str | Image]]:
        raise NotImplementedError("This method is not implemented!")
