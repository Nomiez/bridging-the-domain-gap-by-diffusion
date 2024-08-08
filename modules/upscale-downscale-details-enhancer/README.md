# Upscale Downscale Details Enhancer
"The upscale downscale module tries to enrich image data with more details by first up-scaling the image with the help 
of a latent diffusion network and then down-scaling it with "Pillow". This should 
help with missing details and broken proportions by adding/repairing them during the upscale process and not losing 
them completely during the downscale. In combination with the loop method provided by the pipeline, which can be found 
in the functionality section of my thesis, this module could enhance the image multiple times before passing it to the 
next pipeline method. Its idea originates from the DLSS-Algorithm developed by NVIDIA for PC-Games, which also takes a 
low-resolution image, upscales, and afterward downscales it to the monitor resolution to provide better image 
quality." ~Thesis


## Installation
This module can be used out of the box.