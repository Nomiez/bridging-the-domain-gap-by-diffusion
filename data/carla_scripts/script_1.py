from __future__ import annotations

import random
import time

import numpy as np
from PIL import Image

import carla
from carla_module import CarlaConfig
from carla_module.types import CarlaScriptInterface
from sd_pipeline_typing.types import PipelineConfig

IM_WIDTH = 640
IM_HEIGHT = 480


def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    image = Image.fromarray(i2)
    return image


class CarlaScript(CarlaScriptInterface):
    def __init__(self, *, config: CarlaConfig):
        super().__init__(config=config)
        self.config = config

    def run_script(self, input_data, pipeline_config: PipelineConfig) -> dict[str, str | Image] | tuple[
        dict[str, str | Image]]:

        world = self.state["world"]
        blueprint_library = self.state["blueprint_library"]

        bp = blueprint_library.filter('model3')[0]
        print(bp)

        spawn_point = random.choice(world.get_map().get_spawn_points())

        vehicle = world.spawn_actor(bp, spawn_point)
        vehicle.set_autopilot(True)

        self.actor_list.append(vehicle)

        blueprints = []
        if self.config.generate_images:
            blueprint = blueprint_library.find('sensor.camera.rgb')
            blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
            blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
            blueprint.set_attribute('fov', '110')
            blueprints.append(blueprint)

        if self.config.generate_segmentations:
            # Get Segmentation camera sensor
            blueprint = blueprint_library.find('sensor.camera.semantic_segmentation')
            blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
            blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
            blueprint.set_attribute('fov', '110')
            blueprints.append(blueprint)

        if self.config.generate_depths:
            # Get Depth camera sensor
            blueprint = blueprint_library.find('sensor.camera.depth')
            blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
            blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
            blueprint.set_attribute('fov', '110')
            blueprints.append(blueprint)

        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        sensor = world.spawn_actor(blueprints, spawn_point, attach_to=vehicle)
        self.actor_list.append(sensor)

        # camera.listen(lambda image: save_image(image, output_dir))
        time.sleep(10)

        return {}
