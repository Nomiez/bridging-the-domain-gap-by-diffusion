from __future__ import annotations

import os.path
from pathlib import Path

from sd_pipeline.components import Pipeline, PipelineConfig, SubPipeline
from sd_pipeline.modules.load_from_fs import LFFS, LFFSConfig
from sd_pipeline.modules.store_in_fs import SIFS, SIFSConfig
from sd_pipeline.modules.load_bboxes_from_fs import LBFS, LBFSConfig
from sd_pipeline.modules.rename import Rename, RenameConfig
from sd_pipeline.modules.resize import Resize, ResizeConfig
from sd_pipeline.modules.sort import Sort, SortConfig
from convert2coco import C2C, C2CConfig

"""
This script provides a short demo for converting the outputs of ALDM into a coco dataset with the help of synthehicle 
annotations. Overall the pipeline could also used for other bounding box formats but is limited through the 
'load_bboxes_from_fs' module, which only can handle Synthehicle annotations as well as the transformation from 
(x_1, y_1) (x_2, y_2) to (x_1, y_1), height, width. For FINE annotations please take a look into the scripts folder.
"""

if __name__ == "__main__":
    # Init configs
    pipeline_config = PipelineConfig()

    lffs_config = LFFSConfig(input_dir=os.path.join(os.getcwd(), "data/input_synthehicle_raw"))

    lbfs_config = LBFSConfig(input_dir=os.path.join(os.getcwd(), "data/input_2_large/bbox"))

    sifs_config = SIFSConfig(output_dir="data/output/coco_datasets/third_person/tp_base/images")

    train_c2c_config = C2CConfig(
        output_dir_json="data/output/coco_datasets/third_person/tp_base/annotations/train_labels.json"
    )

    val_c2c_config = C2CConfig(
        output_dir_json="data/output/coco_datasets/third_person/tp_base/annotations/val_labels.json"
    )

    result = (
        Pipeline.init(pipeline_config=pipeline_config)
        .parallel(
            SubPipeline.init()
            .step(LFFS(config=lffs_config))
            .step(Sort(config=SortConfig(key_function=lambda el: int(el["name"].split(".")[0]))))
            .for_each(SubPipeline.init().step(Resize(config=ResizeConfig(W=1920, H=1080))))
            .step(SIFS(config=sifs_config)),
            SubPipeline.init()
            .step(LBFS(config=lbfs_config))
            .step(Sort(config=SortConfig(key_function=lambda el: int(el["name"].split(".")[0]))))
            .for_each(
                SubPipeline.init().step(
                    Rename(
                        config=RenameConfig(
                            rename_function=lambda name: Path(name).stem
                            + "_bbox"
                            + Path(name).suffix
                        )
                    )
                )
            ),
        )
        .prepare(C2C.format_stream(name_of_bbox_contains="_bbox"))
        .split(
            SubPipeline.init().step(C2C(config=train_c2c_config)),
            SubPipeline.init().step(C2C(config=val_c2c_config)),
            split_array=[240, 60],
            type="abs",
        )
        .run()
    )
