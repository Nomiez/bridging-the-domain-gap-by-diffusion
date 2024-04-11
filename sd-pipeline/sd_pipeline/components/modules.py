from abc import ABC, abstractmethod
from typing import Dict, Tuple

from .config import Config
from .pipeline_config import PipelineConfig


class Module(ABC):
    @abstractmethod
    def __init__(self, config: Config):
        ...

    @abstractmethod
    def run(self, input_data: Dict | Tuple[Dict], pipeline_config: PipelineConfig) -> Dict | Tuple[Dict]:
        ...
