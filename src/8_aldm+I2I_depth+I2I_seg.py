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
from image2image.combine import I2ICombined, I2ICombinedConfig
from convert_depth_format.denormalizer import Denormalizer, DenormalizerConfig
from convert_depth_format.inverter import Inverter, InverterConfig
from convert_seg_format.to_ade import ToAde, ToAdeConfig
from pathlib import Path

"""
This script provides a short demo for combining segmentation and depth image2image.
"""

if __name__ == "__main__":
    main_theme = "sunny scene"
    # Init configs
    pipeline_config = PipelineConfig()
    aldm_config = ALDMConfig(
        prompt=main_theme,
        neg=False,
        num_samples=1,
        ddim_steps=25,
        cfg_scale=7.5,
        seed=23,
        random_seed=True,
    )

    lffs_config_images = LFFSConfig(input_dir=os.path.join(os.getcwd(), "data/input_7/images"))

    lffs_config_depth = LFFSConfig(input_dir=os.path.join(os.getcwd(), "data/input_7/dmaps"))

    lffs_config_seg = LFFSConfig(input_dir=os.path.join(os.getcwd(), "data/input_7/images"))

    sifs_config = SIFSConfig(
        output_dir="data/output_9/",
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
                    Rename(
                        config=RenameConfig(
                            rename_function=lambda name: Path(name).stem
                            + "_depth"
                            + Path(name).suffix
                        )
                    )
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
            SubPipeline.init()
            .step(LFFS(config=lffs_config_seg))
            .for_each(SubPipeline.init().step(ToAde(config=ToAdeConfig())))
            .for_each(
                SubPipeline.init().step(
                    Rename(
                        config=RenameConfig(
                            rename_function=lambda name: Path(name).stem
                            + "_segmentation"
                            + Path(name).suffix
                        )
                    )
                )
            ),
        )
        .prepare(
            I2ICombined.format_stream(
                name_of_depth_contains="_depth", name_of_seg_contains="_segmentation"
            )
        )
        .for_each(
            SubPipeline.init().step(
                I2ICombined(
                    config=I2ICombinedConfig(
                        prompt=f"{main_theme}, photorealistic, cars", strength=0.7
                    )
                )
            )
        )
        .step(SIFS(config=sifs_config))
        .run()
    )
