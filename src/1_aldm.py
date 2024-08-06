import os.path

from sd_pipeline.components import Pipeline, PipelineConfig, SubPipeline
from sd_pipeline.modules.load_from_fs import LFFS, LFFSConfig
from sd_pipeline.modules.store_in_fs import SIFS, SIFSConfig
from aldm import ALDM, ALDMConfig

"""
This script provides a short demo of the ALDM module inside the pipeline.
"""

if __name__ == "__main__":
    # Init configs
    pipeline_config = PipelineConfig()
    aldm_config = ALDMConfig(
        prompt="sunny scene, cars, vehicles, grey, realistic",
        neg_prompt="red, ugly, deformed, disfigured, poor details",
        num_samples=1,
        ddim_steps=25,
        cfg_scale=7.5,
        seed=13,
    )

    lffs_config = LFFSConfig(
        input_dir=os.path.join(os.getcwd(), "data/input_1_small/third_person/seg")
    )

    sifs_config = SIFSConfig(
        output_dir="data/output_1_small/third_person/seg",
    )

    result = (
        Pipeline.init(pipeline_config=pipeline_config)
        .step(LFFS(config=lffs_config))
        .for_each(SubPipeline.init().step(ALDM(config=aldm_config)).step(SIFS(config=sifs_config)))
        .run()
    )
