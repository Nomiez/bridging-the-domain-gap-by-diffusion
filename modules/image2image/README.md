# Image to Image

"The Image2Image module extends the ALDM processing by enabling further image manipulation after its initial creation. 
As described in the controlnet section of my thesis, Image to Image processes take an existing image as their base 
and generate a new one from it. This generation can be guided by ControlNets. This module implements multiple 
variations, including image-to-image without a ControlNet, the ControlNet trained on semantic segmentation 
by Lvmin Zhang, the ControlNet trained on depth by Lvmin Zhang, and a combination of both. All modules work with 
Stable Diffusion 1.5 as it was the only available stable diffusion checkpoint with a ControlNet trained on both
inputs." ~Thesis


## Installation
This module can be used out of the box.
