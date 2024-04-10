from __future__ import annotations
from .modules import Module
from typing import Dict, Tuple

class Pipeline:

    def __init__(self):
        self.state : Dict | Tuple[Dict] = {}

    @staticmethod
    def init():
        return Pipeline()
    
    def step(self, step: Pipeline | Module, *, iterations: int = 1):
        if isinstance(step, Pipeline):
            self.state = step.state
        elif isinstance(step, Module):
            for _ in range(iterations):
                self.state = step.run(self.state)
        else:
            raise ValueError("Invalid step type")

        return self

        