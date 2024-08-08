# ALDM

"ALDM is the first step of the image generation process, allowing a segmentation map to be transformed into a realistic, 
detailed image with the help of a stable diffusion algorithm and a control net. To integrate the code into the pipeline, 
it had first been made installable as a package. Therefore, rewriting the original ALDM code and creating a wrapper 
around it to be called from my pipeline was necessary. Additionally, the configuration settings are extracted into a 
configuration class to make them accessible at the pipeline programming level." ~Thesis

Please check out the original repository:
https://github.com/boschresearch/ALDM

## Installation

For the installation of this module, please download 
[this folder](https://github.com/boschresearch/ALDM/tree/main/dataloader) and paste its content into the 
[info](..%2F..%2Fdata%2Finfo)-folder.

For the pretrained checkpoint please download the pretrained weights for the model you want to use as well as the model 
information stored in the [ALDM](https://github.com/boschresearch/ALDM) repository (in the models folder).
and copy it in the corresponding [models](..%2F..%2Fdata%2Fmodels)-folder.

After that ALDM should be ready for tasks.
