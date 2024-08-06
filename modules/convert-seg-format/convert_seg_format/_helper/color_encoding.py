import numpy as np


# Source: ALDM
def get_color_map(color_map: str) -> dict:
    if color_map == "cityscapes":
        return np.array(
            [
                [128, 64, 128],
                [244, 35, 232],
                [70, 70, 70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170, 30],
                [220, 220, 0],
                [107, 142, 35],
                [152, 251, 152],
                [70, 130, 180],
                [220, 20, 60],
                [255, 0, 0],
                [0, 0, 142],
                [0, 0, 70],
                [0, 60, 100],
                [0, 80, 100],
                [0, 0, 230],
                [119, 11, 32],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        )
    elif color_map == "ade20k":
        return np.array(
            [
                [140, 140, 140],  # road
                [235, 255, 7],  # sidewalk
                [180, 120, 120],  # building
                [120, 120, 120],  # wall
                [255, 184, 6],  # fence
                [51, 0, 255],  # pole
                [41, 0, 255],  # traffic light
                [255, 5, 153],  # traffic sign
                [4, 200, 3],  # vegetation # use tree
                [4, 250, 7],  # terrain # use grass
                [6, 230, 230],  # sky
                [150, 5, 61],  # person
                [255, 225, 0],  # rider # mainly based on bicycle
                [0, 102, 200],  # car
                [255, 0, 20],  # truck
                [255, 0, 245],  # bus
                [255, 61, 6],  # train # use rail
                [163, 0, 255],  # motorcycle # motorbike
                [255, 245, 0],  # bicycle
                [0, 0, 0],
            ],
            dtype=np.uint8,
        )
