from __future__ import annotations

import os
from typing import Dict, Tuple
import json


from sd_pipeline_typing.types import Module

from .config import LBFSConfig


def _load_bbox(bbox_name: str, input_dir: str) -> dict:
    # Open all images in the input directory
    output = {}
    image_path = os.path.join(input_dir, bbox_name)
    with open(image_path, "r") as bbox_file:
        output["bbox"] = json.loads(bbox_file.read())
        output["name"] = bbox_name

    return output


class LBFS(Module):
    def __init__(self, *, config: LBFSConfig):
        self.config = config

    def run(self, input_data: str, _) -> Dict[str, str] | Tuple[Dict[str, str]]:
        res = ()
        for bbox_names in os.listdir(self.config.input_dir):
            if bbox_names.endswith(".txt") or bbox_names.endswith(".json"):
                res += (_load_bbox(bbox_names, self.config.input_dir),)
        if len(res) == 1:
            res = res[0]
        return res
