import os.path

from sd_pipeline.components import Pipeline, PipelineConfig, SubPipeline
from sd_pipeline.modules.load_from_fs import LFFS, LFFSConfig
from sd_pipeline.modules.store_in_fs import SIFS, SIFSConfig
from aldm import ALDM, ALDMConfig

if __name__ == '__main__':
    # Init configs
    pipeline_config = PipelineConfig(checkpoint_dir='data/checkpoints')
    aldm_config = ALDMConfig(prompt='rainy scene',
                             neg=False,
                             output_dir='output',
                             num_samples=1,
                             ddim_steps=25,
                             cfg_scale=7.5,
                             seed=23,
                             )

    lffs_config = LFFSConfig(
        input_dir=os.path.join(os.getcwd(), "data/input")
    )

    sifs_config = SIFSConfig(
        output_dir='data/output',
    )

    result = (
        Pipeline.init(pipeline_config=pipeline_config)
        .step(LFFS(config=lffs_config))
        .for_each(
            SubPipeline.init().collect(
                SubPipeline.init().step(ALDM(config=aldm_config)),
                iterations=2
            )
        )
        .step(SIFS(config=sifs_config))
        .run()
    )
