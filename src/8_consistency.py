import os.path

from sd_pipeline.components import Pipeline, PipelineConfig, SubPipeline
from sd_pipeline.modules.load_from_fs import LFFS, LFFSConfig
from sd_pipeline.modules.store_in_fs import SIFS, SIFSConfig
from sd_pipeline.modules.rename import Rename, RenameConfig
from sd_pipeline.modules.resize import Resize, ResizeConfig
from sd_pipeline.modules.sort import Sort, SortConfig
from convert_seg_format.to_ade import ToAde, ToAdeConfig
from image2image import I2I, I2IConfig
from pathlib import Path
from random import randint

if __name__ == "__main__":
    input_seg_maps = "data/cons/seg"
    current_image = "data/cons/current"
    generated_image = "data/cons/output"
    debug_output = "data/cons/debug"

    (
        Pipeline.init(pipeline_config=PipelineConfig())
        .step(LFFS(config=LFFSConfig(input_dir=input_seg_maps)))
        .for_each(SubPipeline.init().step(ToAde(config=ToAdeConfig())))
        .step(SIFS(config=SIFSConfig(output_dir=debug_output)))
        .for_each(
            SubPipeline.init().step(
                Rename(config=RenameConfig(rename_function=lambda x: x + "_segmentation"))
            )
        )
        .for_each(SubPipeline.init().step(Resize(config=ResizeConfig(W=1920, H=1080))))
        .step(Sort(config=SortConfig(key_function=lambda el: el["name"])))
        .for_each(
            SubPipeline.init()
            .parallel(
                SubPipeline.init().step(
                    LFFS(config=LFFSConfig(input_dir=current_image, clear_dir=True))
                ),
                SubPipeline.init(),  # nullop
            )
            .prepare(I2I.format_stream(name_of_seg_contains="_segmentation"))
            .for_each(
                SubPipeline.init().step(
                    I2I(
                        config=I2IConfig(
                            prompt="sunny scene, driving cars, ultra realism", strength=0.3
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
