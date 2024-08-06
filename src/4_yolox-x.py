from sd_pipeline.components import Pipeline, PipelineConfig
from yolox_module import YoloX, YoloXConfig

"""
This script provides a short demo for the use of the yolox module. Important is, that to support custom datasets, 
yolox has to be modified and build by yourself. For more information see the main readme file.
"""

if __name__ == "__main__":
    # Init configs
    pipeline_config = PipelineConfig()

    yolox_config = YoloXConfig(
        exp_file="data/exps/yolox-x.py",
        devices=1,
        batch_size=8,
        fp16=True,
        occupy=True,
        ckpt="data/models/yolox_x.pth",
    )

    result = Pipeline.init(pipeline_config=pipeline_config).step(YoloX(config=yolox_config)).run()
