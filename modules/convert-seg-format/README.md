# Convert Seg Format

"To provide correctly formatted segmentation and depth information for the ControlNets in the image2image module, 
the segmentation maps and depth maps must be properly modified.
The convert-seg-format module offers methods to transform a semantic segmentation map from the Cityscapes dataset 
format to the ade20k dataset format and vice versa. This is essential because Sythehicle provides and ALDM 
consumes Cityscape-formatted segmentation maps, while the ControlNet requires ADE-formatted segmentation maps." ~Thesis


## Installation
This module can be used out of the box.