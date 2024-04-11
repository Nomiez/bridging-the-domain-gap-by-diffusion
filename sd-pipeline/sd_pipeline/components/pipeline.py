from __future__ import annotations

from typing import Dict, Tuple

from .pipeline_config import PipelineConfig
from .modules import Module


class Pipeline:

    def __init__(self, input_data: Dict | Tuple[Dict], pipeline_config: PipelineConfig):
        self.state: Dict | Tuple[Dict] = input_data
        self.config = pipeline_config

    @staticmethod
    def init(*, input_data: Dict | Tuple[Dict], pipeline_config: PipelineConfig):
        return Pipeline(input_data, pipeline_config)

    def step(self, step: Pipeline | Module, *, iterations: int = 1):
        if isinstance(step, Pipeline):
            self.state = step.state
        elif isinstance(step, Module):
            for _ in range(iterations):
                self.state = step.run(self.state, self.config)
        else:
            raise ValueError("Invalid step type")

        return self
