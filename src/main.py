from sd_pipeline.components import Pipeline, PipelineConfig
from sd_pipeline.modules.ALDM import ALDM, ALDMConfig

if __name__ == '__main__':
    # Init configs
    pipeline_config = PipelineConfig(checkpoint_dir='checkpoints')
    aldm_config = ALDMConfig(prompt='rainy scene',
                             neg=False,
                             output_dir='output',
                             num_samples=1,
                             ddim_steps=25,
                             cfg_scale=7.5,
                             seed=23,
                             )
    input_image = {
        'label_path': 'input/frankfurt_000000_000576_gtFine_labelIds.png'
    }

    Pipeline.init(input_data=input_image, pipeline_config=pipeline_config) \
        .step(ALDM(config=aldm_config))
