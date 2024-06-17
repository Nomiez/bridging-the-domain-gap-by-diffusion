from __future__ import annotations

from sd_pipeline_typing.types import Config


class CarlaConfig(Config):
    def __init__(
        self,
        *,
        scripts_dir: str,
        hostname: str,
        port: int,
        generate_images: bool = True,
        generate_segmentations: bool = True,
        generate_depths: bool = True,
        annotate_images: str | None = None,
        annotate_segmentations: str | None = "_segmentation",
        annotate_depths: str | None = "_depth",
    ):
        self.scripts_dir = scripts_dir
        self.hostname = hostname
        self.port = port
        self.generate_images = generate_images
        self.generate_segmentations = generate_segmentations
        self.generate_depths = generate_depths
        self.annotate_images = annotate_images
        self.annotate_segmentations = annotate_segmentations
        self.annotate_depths = annotate_depths
