from abc import ABC, abstractmethod
from typing import Dict, Tuple

class Module(ABC):
    @abstractmethod
    def __init__(self, config: Config):
        ...

    @abstractmethod
    def run(self, input_data: Dict | Tuple[Dict]) -> Dict | Tuple[Dict]:
        ...