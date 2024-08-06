import os.path

from pathlib import Path

from sd_pipeline.components import Pipeline, PipelineConfig, SubPipeline
from sd_pipeline.modules.load_from_fs import LFFS, LFFSConfig
from sd_pipeline.modules.store_in_fs import SIFS, SIFSConfig
from sd_pipeline.modules.rename import Rename, RenameConfig
from aldm import ALDM, ALDMConfig

"""
This script was used to generate the aldm datasets, which were tested in the thesis.
"""

if __name__ == "__main__":
    lffs_config = LFFSConfig(
        input_dir=os.path.join(os.getcwd(), "data/input_1_large/first_person/seg")
    )

    sifs_config = SIFSConfig(
        output_dir="data/output_1_large/first_person/img",
    )

    main_theme = "daytime scene, cars, vehicles, realistic"
    # Init configs
    pipeline_config = PipelineConfig()
    aldm_config = ALDMConfig(
        prompt=main_theme,
        neg_prompt="ugly, deformed, disfigured, poor details",
        num_samples=1,
        ddim_steps=25,
        cfg_scale=7.5,
        seed=23,
        random_seed=True,
    )

    result = (
        Pipeline.init(pipeline_config=pipeline_config)
        .step(LFFS(config=lffs_config))
        .for_each(
            SubPipeline.init()
            .collect(
                SubPipeline.init()
                .step(ALDM(config=aldm_config))
                .step(
                    Rename(
                        config=RenameConfig(
                            rename_function=lambda name: Path(name).stem
                            + "_daytime"
                            + Path(name).suffix
                        )
                    )
                ),
                iterations=2,
            )
            .step(SIFS(config=sifs_config))
        )
        .run()
    )

    main_theme = "dawn scene, cars, vehicles, realistic"
    # Init configs
    pipeline_config = PipelineConfig()
    aldm_config = ALDMConfig(
        prompt=main_theme,
        neg_prompt="ugly, deformed, disfigured, poor details",
        num_samples=1,
        ddim_steps=25,
        cfg_scale=7.5,
        seed=23,
        random_seed=True,
    )

    result = (
        Pipeline.init(pipeline_config=pipeline_config)
        .step(LFFS(config=lffs_config))
        .for_each(
            SubPipeline.init()
            .collect(
                SubPipeline.init()
                .step(ALDM(config=aldm_config))
                .step(
                    Rename(
                        config=RenameConfig(
                            rename_function=lambda name: Path(name).stem
                            + "_dawn"
                            + Path(name).suffix
                        )
                    )
                ),
                iterations=2,
            )
            .step(SIFS(config=sifs_config))
        )
        .run()
    )

    main_theme = "rainy scene, cars, vehicles, realistic"
    # Init configs
    pipeline_config = PipelineConfig()
    aldm_config = ALDMConfig(
        prompt=main_theme,
        neg_prompt="ugly, deformed, disfigured, poor details",
        num_samples=1,
        ddim_steps=25,
        cfg_scale=7.5,
        seed=23,
        random_seed=True,
    )

    result = (
        Pipeline.init(pipeline_config=pipeline_config)
        .step(LFFS(config=lffs_config))
        .for_each(
            SubPipeline.init()
            .collect(
                SubPipeline.init()
                .step(ALDM(config=aldm_config))
                .step(
                    Rename(
                        config=RenameConfig(
                            rename_function=lambda name: Path(name).stem
                            + "_rainy"
                            + Path(name).suffix
                        )
                    )
                ),
                iterations=2,
            )
            .step(SIFS(config=sifs_config))
        )
        .run()
    )

    main_theme = "night scene, cars, vehicles, realistic"
    # Init configs
    pipeline_config = PipelineConfig()
    aldm_config = ALDMConfig(
        prompt=main_theme,
        neg_prompt="ugly, deformed, disfigured, poor details",
        num_samples=1,
        ddim_steps=25,
        cfg_scale=7.5,
        seed=23,
        random_seed=True,
    )

    result = (
        Pipeline.init(pipeline_config=pipeline_config)
        .step(LFFS(config=lffs_config))
        .for_each(
            SubPipeline.init()
            .collect(
                SubPipeline.init()
                .step(ALDM(config=aldm_config))
                .step(
                    Rename(
                        config=RenameConfig(
                            rename_function=lambda name: Path(name).stem
                            + "_night"
                            + Path(name).suffix
                        )
                    )
                ),
                iterations=2,
            )
            .step(SIFS(config=sifs_config))
        )
        .run()
    )
