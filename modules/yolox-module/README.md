# YoloX Module

"The YoloX module is the main evaluator module providing object detection on COCO datasets. YoloX provides different 
models with different layer sizes, offering  large flexibility for training. The module wraps the 
[YoloX](https://github.com/Megvii-BaseDetection/YOLOX) package, allowing it to be integrated into pipelines 
seamlessly." ~Thesis

Please check out the original repository:
https://github.com/Megvii-BaseDetection/YOLOX


## Installation
This module can be used out of the box... If you have the right dataset structure. If you do not have
this, what most of the time is the case, please download YoloX from 
[source](https://github.com/Megvii-BaseDetection/YOLOX) and adjusting all files necessary. This most of the times
are files [here](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/yolox/evaluators).
After that write the exps configuration file, drop into the [exps](..%2F..%2Fdata%2Fexps)-folder, download the 
corresponding ccoc weights (if needed) from [here](https://github.com/Megvii-BaseDetection/YOLOX), paste it into
the [yolox-coco-pre-trained](..%2F..%2Fdata%2Fmodels%2Fyolox-coco-pre-trained)-folder and you should be ready to go.