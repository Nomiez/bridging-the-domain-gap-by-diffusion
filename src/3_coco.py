import os.path

from sd_pipeline.components import Pipeline, PipelineConfig, SubPipeline
from sd_pipeline.modules.load_from_fs import LFFS, LFFSConfig
from sd_pipeline.modules.store_in_fs import SIFS, SIFSConfig
from sd_pipeline.modules.load_bboxes_from_fs import LBFS, LBFSConfig
from sd_pipeline.modules.rename import Rename, RenameConfig
from convert2coco import C2C, C2CConfig
if __name__ == '__main__':
    # Init configs
    pipeline_config = PipelineConfig()

    lffs_config = LFFSConfig(
        input_dir=os.path.join(os.getcwd(), "data/input")
    )

    lbfs_config = LBFSConfig(
        input_dir=os.path.join(os.getcwd(), "data/bbox")
    )

    result = (
        Pipeline.init(pipeline_config=pipeline_config)
                .parallel(
                    SubPipeline.init()
                        .step(LFFS(config=lffs_config))
                        .step(SIFS(config=SIFSConfig(output_dir="data/output/dataset/images"))),
                    SubPipeline.init()
                        .step(LBFS(config=lbfs_config))
                        .for_each(
                            SubPipeline.init()
                                .step(Rename(config=RenameConfig(rename_function=lambda x: x + "_bbox")))
                        )
                )
                .prepare(C2C.format_stream(name_of_bbox_contains="_bbox"))
                .split(
                    SubPipeline.init()
                        .step(C2C(config=C2CConfig(output_dir_json="data/output/dataset/annotations/train_labels.json"))),
                    SubPipeline.init()
                        .step(C2C(config=C2CConfig(output_dir_json="data/output/dataset/annotations/val_labels.json"))),
                    SubPipeline.init()
                        .step(C2C(config=C2CConfig(output_dir_json="data/output/dataset/annotations/test_labels.json"))),
                    split_array=[200, 50, 50], 
                    type="abs"
                )
                .run()
    )
