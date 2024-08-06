import os.path

from sd_pipeline.components import Pipeline, PipelineConfig, SubPipeline
from sd_pipeline.modules.load_from_fs import LFFS, LFFSConfig
from sd_pipeline.modules.store_in_fs import SIFS, SIFSConfig
from sd_pipeline.modules.rename import Rename, RenameConfig
from sd_pipeline.modules.resize import Resize, ResizeConfig
from convert_seg_format.to_ade import ToAde, ToAdeConfig
from aldm import ALDM, ALDMConfig
from image2image import I2I, I2IConfig

"""
This script provides a short demo of the image2image segmentation module inside the pipeline.
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

    lffs_config = LFFSConfig(input_dir=os.path.join(os.getcwd(), "data/input_7/images"))

    sifs_config = SIFSConfig(
        output_dir="data/output_2",
    )

    result = (
        Pipeline.init(pipeline_config=pipeline_config)
        .parallel(
            SubPipeline.init()
            .step(LFFS(config=lffs_config))
            .for_each(SubPipeline.init().step(ALDM(config=aldm_config)))
            .for_each(SubPipeline.init().step(Resize(config=ResizeConfig(W=1920, H=1080)))),
            SubPipeline.init()
            .step(LFFS(config=lffs_config))
            .for_each(SubPipeline.init().step(ToAde(config=ToAdeConfig())))
            .for_each(
                SubPipeline.init().step(
                    Rename(config=RenameConfig(rename_function=lambda x: x + "_segmentation"))
                )
            ),
        )
        .prepare(I2I.format_stream(name_of_seg_contains="_segmentation"))
        .for_each(
            SubPipeline.init().step(
                I2I(
                    config=I2IConfig(
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
