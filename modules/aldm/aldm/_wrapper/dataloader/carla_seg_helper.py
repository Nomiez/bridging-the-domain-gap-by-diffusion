import os

from PIL import Image
import numpy as np
import torch


def _check_for_fixes(rgb):
    if np.all(rgb == [81, 0, 81]):      # Parking lot
        return [128, 64, 128]       # Road
    elif np.all(rgb == [170, 120, 50]): # Rod
        return [190, 153, 153]      # Fence
    elif np.all(rgb == [110, 190, 160]): # Brick / Fountain
        return [70, 70, 70]         # Building
    elif np.all(rgb == [157, 234, 50]): # road lines
        return [128, 64, 128]       # Road
    elif np.all(rgb == [100, 40, 40]): # Electrical box
        return [70, 70, 70]         # Building
    elif np.all(rgb == [145, 170, 100]): # Grass
        return [152, 251, 152]      # Vegetation
    else:
        return rgb


def map_color_to_trainId(label: Image.Image, name: str,  label_color_map, W: int, H: int) -> Image.Image:
    # Convert color Image to label Image
    label = label.convert("RGB")
    label = label.resize((W, H), Image.NEAREST)
    label = np.array(label).astype(np.int64)

    # Convert class id to train id
    res = np.empty((512, 1024), dtype=np.uint8)

    colors = []
    for irow, row, in enumerate(label):
        for irgb, rgb in enumerate(row):
            rgb = _check_for_fixes(rgb)
            for i, el in enumerate(label_color_map):
                if np.all(rgb == el):
                    res[irow, irgb] = i
                    break
                elif i == len(label_color_map) - 1:
                    colors.append(str(rgb))
                    res[irow, irgb] = 255

    print(list(set(colors)))
    # Save as Image
    Image.fromarray(res).save(os.path.join("debug", f"segmentation-{name}.png"))
    label = torch.LongTensor(res)
    return label
