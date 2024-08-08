import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2  # noqa
import numpy as np  # noqa
from PIL import Image, ImageOps  # noqa


def exr_to_png(exr_path, png_path):
    depth = cv2.imread(exr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth = Image.fromarray(depth)
    depth = np.asarray(depth, dtype=np.float32)
    depth = 400 * (1 - depth)  # the depth in meters
    cv2.imwrite(png_path, depth)

    im = Image.open(png_path)
    im = ImageOps.invert(im)
    im = np.array(im)
    cv2.imwrite(png_path, im)

    im = cv2.imread(png_path)
    im[(im >= 255).sum(axis=2) == 3] = [0, 0, 0]
    cv2.imwrite(png_path, im)


# Example usage

print("testcd")
print(os.listdir("./"))
for image_name in os.listdir("./"):
    if image_name.endswith(".exr"):
        exr_to_png(image_name, f"./converted/{image_name.split('.')[0]}.png")
