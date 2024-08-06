import os.path

import numpy as np

from sd_pipeline.components import Pipeline, PipelineConfig, SubPipeline
from sd_pipeline.modules.load_from_fs import LFFS, LFFSConfig
from sd_pipeline.modules.store_in_fs import SIFS, SIFSConfig
from sd_pipeline.modules.rename import Rename, RenameConfig
from sd_pipeline.modules.resize import Resize, ResizeConfig
from sd_pipeline.modules.sort import Sort, SortConfig
from sd_pipeline.modules.group import Group, GroupConfig
from convert_seg_format.to_ade import ToAde, ToAdeConfig
from convert_depth_format.denormalizer import Denormalizer, DenormalizerConfig
from convert_depth_format.inverter import Inverter, InverterConfig
from image2image.combine import I2ICombined, I2ICombinedConfig
from pathlib import Path

"""
This script generates the output for the temporal and spacial consistency tests guided by segmentation & depth.
"""

if __name__ == "__main__":
    input_seg_maps = "data/input_8_100/Scene5/seg"
    input_dep_maps = "data/input_8_100/Scene5/dep"
    current_image = "data/input_8_100/Scene5/current"
    generated_image = "data/input_8_100/Scene5/output_seg_dep_med"
    debug_output = "data/input_8_100/Scene5/debug"
    prompt = "foggy scene, cars, vehicles, grey, realistic"
    negative_prompt = "red, ugly, deformed, disfigured, poor details"

    (
        Pipeline.init(pipeline_config=PipelineConfig())
        .parallel(
            SubPipeline.init()
            .step(LFFS(config=LFFSConfig(input_dir=input_seg_maps)))
            .for_each(SubPipeline.init().step(ToAde(config=ToAdeConfig())))
            .step(SIFS(config=SIFSConfig(output_dir=debug_output)))
            .for_each(
                SubPipeline.init().step(
                    Rename(config=RenameConfig(rename_function=lambda x: x + "_segmentation"))
                )
            )
            .for_each(SubPipeline.init().step(Resize(config=ResizeConfig(W=1920, H=1080))))
            .step(Sort(config=SortConfig(key_function=lambda el: el["name"]))),
            SubPipeline.init()
            .step(LFFS(config=LFFSConfig(input_dir=input_dep_maps)))
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
            .for_each(SubPipeline.init().step(Inverter(config=InverterConfig())))
            .step(Sort(config=SortConfig(key_function=lambda el: el["name"]))),
        )
        .step(Group(config=GroupConfig()))
        .for_each(
            SubPipeline.init()
            .parallel(
                SubPipeline.init()
                .step(LFFS(config=LFFSConfig(input_dir=current_image, clear_dir=True)))
                .step(Resize(config=ResizeConfig(W=1920, H=1080)))
                .flatten(depth=-1),
                SubPipeline.init(),  # nullop
            )
            .flatten(depth=1)
            .prepare(
                I2ICombined.format_stream(
                    name_of_seg_contains="_segmentation", name_of_depth_contains="_depth"
                )
            )
            .for_each(
                SubPipeline.init().step(
                    I2ICombined(
                        config=I2ICombinedConfig(
                            prompt=prompt, negative_prompt=negative_prompt, strength=0.6
                        )
                    )
                )
            )
            .step(
                Rename(
                    config=RenameConfig(
                        rename_function=lambda name: f"{int(Path(name).stem.split(sep='_')[0]) + 1}{Path(name).suffix}"
                    )
                )
            )
            .step(SIFS(config=SIFSConfig(output_dir=current_image)))
            .step(SIFS(config=SIFSConfig(output_dir=generated_image)))
        )
        .run()
    )
