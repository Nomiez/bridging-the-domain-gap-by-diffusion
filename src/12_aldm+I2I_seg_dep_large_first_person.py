import os
import os.path

import numpy as np

from pathlib import Path

from sd_pipeline.components import Pipeline, PipelineConfig, SubPipeline
from sd_pipeline.modules.load_from_fs import LFFS, LFFSConfig
from sd_pipeline.modules.store_in_fs import SIFS, SIFSConfig
from sd_pipeline.modules.rename import Rename, RenameConfig
from sd_pipeline.modules.resize import Resize, ResizeConfig
from sd_pipeline.modules.filter import Filter, FilterConfig
from sd_pipeline.modules.multiply import Multiply, MultiplyConfig
from sd_pipeline.modules.sort import Sort, SortConfig
from convert_seg_format.to_ade import ToAde, ToAdeConfig
from image2image.combine import I2ICombined, I2ICombinedConfig
from convert_depth_format.denormalizer import Denormalizer, DenormalizerConfig
from convert_depth_format.inverter import Inverter, InverterConfig

"""
This script was used to generate the first person image2image segmentation + depth dataset, which is based on 
MUAD. Therefor all necessary transformations to the depth images had to be done. 
The output from 9_aldm_large.py is also used here.
"""

if __name__ == "__main__":
    lffs_config = LFFSConfig(
        input_dir=os.path.join(os.getcwd(), "data/output_1_large/first_person/img")
    )

    sifs_config = SIFSConfig(
        output_dir="data/output_8_large/first_person/img_new",
    )

    lffs_config_seg = LFFSConfig(
        input_dir=os.path.join(os.getcwd(), "data/input_1_large/first_person/seg")
    )

    lffs_config_depth = LFFSConfig(
        input_dir=os.path.join(os.getcwd(), "data/first_person_dataset/leftDep8bit")
    )

    main_theme = "daytime scene, cars, vehicles, realistic"
    # Init configs
    pipeline_config = PipelineConfig()
    result = (
        Pipeline.init(pipeline_config=pipeline_config)
        .parallel(
            SubPipeline.init()
            .step(LFFS(config=lffs_config))
            .step(Filter(config=FilterConfig(filter_function=lambda x: "daytime" in x["name"])))
            .step(Sort(config=SortConfig(key_function=lambda el: int(el["name"].split("_")[0]))))
            .for_each(SubPipeline.init().step(Resize(config=ResizeConfig(W=2048, H=1024)))),
            SubPipeline.init()
            .step(LFFS(config=lffs_config_seg))
            .for_each(
                SubPipeline.init()
                .step(Resize(config=ResizeConfig(W=1024, H=512)))
                .step(Resize(config=ResizeConfig(W=2048, H=1024)))
            )
            .for_each(SubPipeline.init().step(ToAde(config=ToAdeConfig())))
            .for_each(SubPipeline.init().step(Multiply(config=MultiplyConfig(factor=2))))
            .for_each(
                SubPipeline.init().step(
                    Rename(config=RenameConfig(rename_function=lambda x: x + "_segmentation"))
                )
            )
            .step(Sort(config=SortConfig(key_function=lambda el: int(el["name"].split("_")[0])))),
            SubPipeline.init()
            .step(LFFS(config=lffs_config_depth))
            .for_each(
                SubPipeline.init().step(
                    Rename(config=RenameConfig(rename_function=lambda x: x + "_depth"))
                )
            )
            .for_each(SubPipeline.init().step(Multiply(config=MultiplyConfig(factor=2))))
            .step(Sort(config=SortConfig(key_function=lambda el: int(el["name"].split("_")[0])))),
        )
        .prepare(
            I2ICombined.format_stream(
                name_of_seg_contains="_segmentation", name_of_depth_contains="_depth"
            )
        )
        .for_each(
            SubPipeline.init()
            .step(
                I2ICombined(
                    config=I2ICombinedConfig(
                        prompt=main_theme,
                        negative_prompt="painting, ugly, deformed, disfigured, poor details",
                        strength=0.65,
                    )
                )
            )
            .step(SIFS(config=sifs_config))
        )
        .run()
    )

    main_theme = "dawn scene, cars, vehicles, realistic"
    # Init configs
    pipeline_config = PipelineConfig()
    result = (
        Pipeline.init(pipeline_config=pipeline_config)
        .parallel(
            SubPipeline.init()
            .step(LFFS(config=lffs_config))
            .step(Filter(config=FilterConfig(filter_function=lambda x: "dawn" in x["name"])))
            .step(Sort(config=SortConfig(key_function=lambda el: int(el["name"].split("_")[0]))))
            .for_each(SubPipeline.init().step(Resize(config=ResizeConfig(W=2048, H=1024)))),
            SubPipeline.init()
            .step(LFFS(config=lffs_config_seg))
            .for_each(
                SubPipeline.init()
                .step(Resize(config=ResizeConfig(W=1024, H=512)))
                .step(Resize(config=ResizeConfig(W=2048, H=1024)))
            )
            .for_each(SubPipeline.init().step(ToAde(config=ToAdeConfig())))
            .for_each(SubPipeline.init().step(Multiply(config=MultiplyConfig(factor=2))))
            .for_each(
                SubPipeline.init().step(
                    Rename(config=RenameConfig(rename_function=lambda x: x + "_segmentation"))
                )
            )
            .step(Sort(config=SortConfig(key_function=lambda el: int(el["name"].split("_")[0])))),
            SubPipeline.init()
            .step(LFFS(config=lffs_config_depth))
            .for_each(
                SubPipeline.init().step(
                    Rename(config=RenameConfig(rename_function=lambda x: x + "_depth"))
                )
            )
            .for_each(SubPipeline.init().step(Multiply(config=MultiplyConfig(factor=2))))
            .step(Sort(config=SortConfig(key_function=lambda el: int(el["name"].split("_")[0])))),
        )
        .prepare(
            I2ICombined.format_stream(
                name_of_seg_contains="_segmentation", name_of_depth_contains="_depth"
            )
        )
        .for_each(
            SubPipeline.init()
            .step(
                I2ICombined(
                    config=I2ICombinedConfig(
                        prompt=main_theme,
                        negative_prompt="painting, ugly, deformed, disfigured, poor details",
                        strength=0.65,
                    )
                )
            )
            .step(SIFS(config=sifs_config))
        )
        .run()
    )

    main_theme = "rainy scene, cars, vehicles, realistic"
    # Init configs
    pipeline_config = PipelineConfig()
    result = (
        Pipeline.init(pipeline_config=pipeline_config)
        .parallel(
            SubPipeline.init()
            .step(LFFS(config=lffs_config))
            .step(Filter(config=FilterConfig(filter_function=lambda x: "rainy" in x["name"])))
            .step(Sort(config=SortConfig(key_function=lambda el: int(el["name"].split("_")[0]))))
            .for_each(SubPipeline.init().step(Resize(config=ResizeConfig(W=2048, H=1024)))),
            SubPipeline.init()
            .step(LFFS(config=lffs_config_seg))
            .for_each(
                SubPipeline.init()
                .step(Resize(config=ResizeConfig(W=1024, H=512)))
                .step(Resize(config=ResizeConfig(W=2048, H=1024)))
            )
            .for_each(SubPipeline.init().step(ToAde(config=ToAdeConfig())))
            .for_each(SubPipeline.init().step(Multiply(config=MultiplyConfig(factor=2))))
            .for_each(
                SubPipeline.init().step(
                    Rename(config=RenameConfig(rename_function=lambda x: x + "_segmentation"))
                )
            )
            .step(Sort(config=SortConfig(key_function=lambda el: int(el["name"].split("_")[0])))),
            SubPipeline.init()
            .step(LFFS(config=lffs_config_depth))
            .for_each(
                SubPipeline.init().step(
                    Rename(config=RenameConfig(rename_function=lambda x: x + "_depth"))
                )
            )
            .for_each(SubPipeline.init().step(Multiply(config=MultiplyConfig(factor=2))))
            .step(Sort(config=SortConfig(key_function=lambda el: int(el["name"].split("_")[0])))),
        )
        .prepare(
            I2ICombined.format_stream(
                name_of_seg_contains="_segmentation", name_of_depth_contains="_depth"
            )
        )
        .for_each(
            SubPipeline.init()
            .step(
                I2ICombined(
                    config=I2ICombinedConfig(
                        prompt=main_theme,
                        negative_prompt="painting, ugly, deformed, disfigured, poor details",
                        strength=0.65,
                    )
                )
            )
            .step(SIFS(config=sifs_config))
        )
        .run()
    )

    main_theme = "night scene, cars, vehicles, realistic"
    # Init configs
    pipeline_config = PipelineConfig()
    result = (
        Pipeline.init(pipeline_config=pipeline_config)
        .parallel(
            SubPipeline.init()
            .step(LFFS(config=lffs_config))
            .step(Filter(config=FilterConfig(filter_function=lambda x: "night" in x["name"])))
            .step(Sort(config=SortConfig(key_function=lambda el: int(el["name"].split("_")[0]))))
            .for_each(SubPipeline.init().step(Resize(config=ResizeConfig(W=2048, H=1024)))),
            SubPipeline.init()
            .step(LFFS(config=lffs_config_seg))
            .for_each(
                SubPipeline.init()
                .step(Resize(config=ResizeConfig(W=1024, H=512)))
                .step(Resize(config=ResizeConfig(W=2048, H=1024)))
            )
            .for_each(SubPipeline.init().step(ToAde(config=ToAdeConfig())))
            .for_each(SubPipeline.init().step(Multiply(config=MultiplyConfig(factor=2))))
            .for_each(
                SubPipeline.init().step(
                    Rename(config=RenameConfig(rename_function=lambda x: x + "_segmentation"))
                )
            )
            .step(Sort(config=SortConfig(key_function=lambda el: int(el["name"].split("_")[0])))),
            SubPipeline.init()
            .step(LFFS(config=lffs_config_depth))
            .for_each(
                SubPipeline.init().step(
                    Rename(config=RenameConfig(rename_function=lambda x: x + "_depth"))
                )
            )
            .for_each(SubPipeline.init().step(Multiply(config=MultiplyConfig(factor=2))))
            .step(Sort(config=SortConfig(key_function=lambda el: int(el["name"].split("_")[0])))),
        )
        .prepare(
            I2ICombined.format_stream(
                name_of_seg_contains="_segmentation", name_of_depth_contains="_depth"
            )
        )
        .for_each(
            SubPipeline.init()
            .step(
                I2ICombined(
                    config=I2ICombinedConfig(
                        prompt=main_theme,
                        negative_prompt="painting, ugly, deformed, disfigured, poor details",
                        strength=0.65,
                    )
                )
            )
            .step(SIFS(config=sifs_config))
        )
        .run()
    )
