from PIL import Image
import numpy as np


def replace(image: Image.Image, colors_from: np.ndarray, colors_to: np.ndarray) -> Image:
    image = image.convert("RGB")
    img_array = np.array(image)

    for i in range(len(colors_from)):
        mask = np.all(img_array == colors_from[i], axis=-1)
        img_array[mask] = colors_to[i]

    return Image.fromarray(img_array)
