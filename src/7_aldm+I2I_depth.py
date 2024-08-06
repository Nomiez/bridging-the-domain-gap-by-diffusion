import os.path
import PIL
import numpy as np

import PIL.Image
from sd_pipeline.components import Pipeline, PipelineConfig, SubPipeline
from sd_pipeline.modules.load_from_fs import LFFS, LFFSConfig
from sd_pipeline.modules.store_in_fs import SIFS, SIFSConfig
from sd_pipeline.modules.rename import Rename, RenameConfig
from sd_pipeline.modules.resize import Resize, ResizeConfig
from aldm import ALDM, ALDMConfig
from image2image.depth import I2IDepth, I2IDepthConfig
from convert_depth_format.denormalizer import Denormalizer, DenormalizerConfig
from convert_depth_format.inverter import Inverter, InverterConfig


"""
This script provides a short demo of the image2image depth module inside the pipeline.
"""

if __name__ == "__main__":
    main_theme = "sunny scene"
    # Init configs
    pipeline_config = PipelineConfig()
    aldm_config = ALDMConfig(
        prompt=main_theme,
        neg_prompt="red, ugly, deformed, disfigured, poor details",
        num_samples=1,
        ddim_steps=25,
        cfg_scale=7.5,
        seed=23,
        random_seed=True,
    )

    lffs_config_images = LFFSConfig(input_dir=os.path.join(os.getcwd(), "data/input_7/images"))

    lffs_config_depth = LFFSConfig(input_dir=os.path.join(os.getcwd(), "data/input_7/dmaps"))

    sifs_config = SIFSConfig(
        output_dir="data/output_7/",
    )

    result = (
        Pipeline.init(pipeline_config=pipeline_config)
        .parallel(
            SubPipeline.init()
            .step(LFFS(config=lffs_config_images))
            .for_each(SubPipeline.init().step(ALDM(config=aldm_config)))
            .for_each(SubPipeline.init().step(Resize(config=ResizeConfig(W=1920, H=1080)))),
            SubPipeline.init()
            .step(LFFS(config=lffs_config_depth))
            .for_each(
                SubPipeline.init().step(
                    Rename(config=RenameConfig(rename_function=lambda x: x + "_depth"))
                )
            )
            .for_each(
                SubPipeline.init().step(
                    Denormalizer(
                        config=DenormalizerConfig(
                            denormalize_function=lambda depth: np.exp(
                                ((depth[depth > 0] / 255) - 1) * 5.70378
                            )
                            * 1000
                        )
                    )
                )
            )
            .for_each(SubPipeline.init().step(Inverter(config=InverterConfig()))),
        )
        .prepare(I2IDepth.format_stream(name_of_depth_contains="_depth"))
        .for_each(
            SubPipeline.init().step(
                I2IDepth(
                    config=I2IDepthConfig(
                        prompt=f"{main_theme}, photorealistic, cars",
                        negative_prompt="ugly, deformed, disfigured, poor details",
                        strength=0.7,
                    )
                )
            )
        )
        .step(SIFS(config=sifs_config))
        .run()
    )
