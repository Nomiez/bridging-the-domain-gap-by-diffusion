from __future__ import annotations

from typing import Dict, Tuple, Callable

from .pipeline_config import PipelineConfig
from .modules import Module


class Pipeline:

    def __init__(self, input_data: Dict | Tuple | None = None, pipeline_config: PipelineConfig | None = None,
                 functions: list[Callable] = None):
        self.input_data = input_data
        self.config = pipeline_config
        self.functions = functions or []

    @staticmethod
    def init(*, input_data: Dict | Tuple, pipeline_config: PipelineConfig) -> Pipeline:
        return Pipeline(input_data, pipeline_config)

    def step(self, module: Module, *, iterations: int = 1) -> Pipeline:
        for _ in range(iterations):
            self.functions.append(lambda input_data, config: module.run(input_data, config))
        return self

    def loop(self, pipeline: Pipeline, *, iterations: int) -> Pipeline:
        for _ in range(iterations):
            self.functions.extend(pipeline.functions)
        return self

    def collect(self, pipeline: Pipeline, *, iterations: int) -> Pipeline:
        def store_output_iter(input_data: Dict | Tuple[Dict], config: PipelineConfig) -> Tuple:
            output = ()
            for _ in range(iterations):
                output += (Pipeline(input_data, config, pipeline.functions).run(),)
            return output

        for _ in range(iterations):
            self.functions.extend(pipeline.functions)
        return self

    def parallel(self, *pipelines: Pipeline) -> Pipeline:
        def store_output_para(input_data: Dict | Tuple[Dict], config: PipelineConfig) -> Tuple:
            output = ()
            for pipeline in pipelines:
                output += (Pipeline(input_data, config, pipeline.functions).run(),)
            return output

        self.functions.append(store_output_para)

        return self

    def flatten(self) -> Pipeline:
        def flatten_output(input_data: Tuple) -> Dict:
            output = {}
            for idx, data in enumerate(input_data):
                output[str(idx)] = data
            return output

        self.functions.append(flatten_output)
        return self

    def run(self) -> Dict:
        input_data = self.input_data
        for function in self.functions:
            input_data = function(input_data, self.config)
        return input_data


class SubPipeline:
    @staticmethod
    def init() -> Pipeline:
        return Pipeline()
