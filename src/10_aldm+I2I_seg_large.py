import os
import os.path

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
from image2image import I2I, I2IConfig

"""
This script was used to generate the image2image segmentation datasets, which were tested in the thesis. The output from
9_aldm_large.py is used here.
"""

if __name__ == "__main__":
    lffs_config = LFFSConfig(
        input_dir=os.path.join(os.getcwd(), "data/output_1_large/first_person/img")
    )

    sifs_config = SIFSConfig(
        output_dir="data/output_2_large/first_person/img",
    )

    lffs_config_seg = LFFSConfig(
        input_dir=os.path.join(os.getcwd(), "data/input_1_large/first_person/seg")
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
        )
        .prepare(I2I.format_stream(name_of_seg_contains="_segmentation"))
        .for_each(
            SubPipeline.init()
            .step(
                I2I(
                    config=I2IConfig(
                        prompt=main_theme,
                        negative_prompt="painting, ugly, deformed, disfigured, poor details",
                        strength=0.4,
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
        )
        .prepare(I2I.format_stream(name_of_seg_contains="_segmentation"))
        .for_each(
            SubPipeline.init().step(
                I2I(
                    config=I2IConfig(
                        prompt=main_theme,
                        negative_prompt="ugly, deformed, disfigured, poor details",
                        strength=0.4,
                    )
                )
            )
        )
        .step(SIFS(config=sifs_config))
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
        )
        .prepare(I2I.format_stream(name_of_seg_contains="_segmentation"))
        .for_each(
            SubPipeline.init().step(
                I2I(
                    config=I2IConfig(
                        prompt=main_theme,
                        negative_prompt="ugly, deformed, disfigured, poor details",
                        strength=0.4,
                    )
                )
            )
        )
        .step(SIFS(config=sifs_config))
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
        )
        .prepare(I2I.format_stream(name_of_seg_contains="_segmentation"))
        .for_each(
            SubPipeline.init().step(
                I2I(
                    config=I2IConfig(
                        prompt=main_theme,
                        negative_prompt="ugly, deformed, disfigured, poor details",
                        strength=0.4,
                    )
                )
            )
        )
        .step(SIFS(config=sifs_config))
        .run()
    )
