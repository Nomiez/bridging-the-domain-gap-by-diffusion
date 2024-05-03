from __future__ import annotations

import os
from typing import Dict, Tuple, Callable, Optional

from PIL.Image import Image

from sd_pipeline_typing.types import PipelineConfig, Module


class Pipeline:

    def __init__(self, pipeline_config: PipelineConfig | None = None,
                 functions: list[Callable] = None):
        self.config = pipeline_config
        self.functions = functions or []

    @staticmethod
    def init(*, pipeline_config: PipelineConfig) -> Pipeline:

        # Create path /debug if not exists
        if not os.path.exists("debug"):
            os.makedirs("debug")
        return Pipeline(pipeline_config)

    def step(self, module: Module, *, iterations: int = 1) -> Pipeline:
        for _ in range(iterations):
            self.functions.append(lambda input_data, config: module.run(input_data, config))
        return self

    def loop(self, pipeline: Pipeline, *, iterations: int) -> Pipeline:
        for _ in range(iterations):
            self.functions.extend(pipeline.functions)
        return self

    def collect(self, pipeline: Pipeline, *, iterations: int) -> Pipeline:
        def store_output_iter(input_data: Dict[str, str | Image] | Tuple[Dict[str, str | Image]],
                              config: PipelineConfig) -> Tuple:
            output = ()
            for _ in range(iterations):
                res = (Pipeline(config, pipeline.functions.copy())._inject_image_data(input_data).run(),)
                # Check if res is a tuple
                if isinstance(res, tuple):
                    output += res
                else:
                    output += (res,)
            return output

        self.functions.append(store_output_iter)
        return self

    def for_each(self, pipeline: Pipeline) -> Pipeline:
        def store_output_for_each(input_data: Dict[str, str | Image] | Tuple[Dict[str, str | Image]],
                                  config: PipelineConfig) -> Dict[str, str | Image] | Tuple:
            if isinstance(input_data, dict):
                return Pipeline(config, pipeline.functions)._inject_image_data(input_data).run()
            else:
                output = ()
                
                for data in input_data:
                    res = Pipeline(config, pipeline.functions.copy())._inject_image_data(data).run()
                    # Check if res is a tuple
                    if isinstance(res, tuple):
                        output += res
                    else:
                        output += (res,)
            return output

        self.functions.append(store_output_for_each)

        return self

    def parallel(self, *pipelines: Pipeline) -> Pipeline:
        def store_output_para(input_data: Dict[str, str | Image] | Tuple[Dict[str, str | Image]],
                              config: PipelineConfig) -> Tuple:
            output = ()
            for pipeline in pipelines:
                res = (Pipeline(config, pipeline.functions.copy())._inject_image_data(input_data).run(),)
                # Check if res is a tuple
                if isinstance(res, tuple):
                    output += res
                else:
                    output += (res,)
            return output

        self.functions.append(store_output_para)

        return self

    def prepare(self, prepare: Callable) -> Pipeline:
        self.functions.append(prepare)
        return self

    def run(self) -> Dict[str, str | Image]:
        input_data = None
        for function in self.functions:
            input_data = function(input_data, self.config)
        return input_data

    def _inject_image_data(self, input_data: Dict[str, str | Image] | Tuple[Dict[str, str | Image]]) -> Pipeline:
        def load_image_data(_: None, __: PipelineConfig) -> Dict | Tuple:
            return input_data

        self.functions.insert(0, load_image_data)

        return self


class SubPipeline:
    @staticmethod
    def init() -> Pipeline:
        return Pipeline()
