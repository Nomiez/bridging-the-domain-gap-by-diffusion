from abc import ABC, abstractmethod
from typing import Dict, Tuple
from PIL.Image import Image

from .config import Config
from .pipeline_config import PipelineConfig


class Module(ABC):
    @abstractmethod
    def __init__(self, *, config: Config | None = None): ...

    @abstractmethod
    def run(
        self,
        input_data: Dict[str, str | Image] | Tuple[Dict[str, str | Image]] | None,
        pipeline_config: PipelineConfig,
    ) -> Dict[str, str | Image] | Tuple[Dict[str, str | Image]] | Tuple[str]: ...
