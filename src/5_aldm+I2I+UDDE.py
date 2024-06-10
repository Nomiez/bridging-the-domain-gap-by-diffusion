import os.path

from sd_pipeline.components import Pipeline, PipelineConfig, SubPipeline
from sd_pipeline.modules.load_from_fs import LFFS, LFFSConfig
from sd_pipeline.modules.store_in_fs import SIFS, SIFSConfig
from sd_pipeline.modules.rename import Rename, RenameConfig
from sd_pipeline.modules.resize import Resize, ResizeConfig
from upscale_downscale_details_enhancer import UDDE, UDDEConfig
from aldm import ALDM, ALDMConfig
from image2image import I2I, I2IConfig

if __name__ == "__main__":
    main_theme = "snowy scene"
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

    lffs_config = LFFSConfig(input_dir=os.path.join(os.getcwd(), "data/input_one"))

    sifs_config = SIFSConfig(
        output_dir="data/output_one",
    )

    result = (
        Pipeline.init(pipeline_config=pipeline_config)
        .parallel(
            SubPipeline.init()
            .step(LFFS(config=lffs_config))
            .for_each(SubPipeline.init().step(ALDM(config=aldm_config))),
            SubPipeline.init()
            .step(LFFS(config=lffs_config))
            .for_each(
                SubPipeline.init().step(
                    Rename(config=RenameConfig(rename_function=lambda x: x + "_segmentation"))
                )
            )
            .for_each(SubPipeline.init().step(Resize(config=ResizeConfig(W=1024, H=512)))),
        )
        .prepare(I2I.format_stream(name_of_seg_contains="_segmentation"))
        .for_each(
            SubPipeline.init().step(
                I2I(
                    config=I2IConfig(
                        prompt=f"{main_theme}, ultra realism, sharp, detailed, 8k", strength=0.5
                    )
                )
            )
        )
        .for_each(
            SubPipeline.init().step(
                UDDE(
                    config=UDDEConfig(
                        prompt=f"{main_theme}, ultra realism, sharp, detailed, 8k",
                        pre_H=400,
                        pre_W=800,
                    )
                ),
                iterations=10,
            )
        )
        .step(SIFS(config=sifs_config))
        .run()
    )
