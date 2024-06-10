from __future__ import annotations

import warnings
import random

import torch
from PIL import Image
from torch.backends import cudnn

from yolox.core import launch
from yolox.exp import Exp, get_exp
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices

from sd_pipeline_typing.types import Module

from .config import YoloXConfig


class YoloX(Module):
    def __init__(self, *, config: YoloXConfig):
        self.config = config

    def run(
        self, input_data: dict[str, str | Image.Image], pipeline_config
    ) -> dict[str, str | Image.Image]:
        def main(exp: Exp, args):
            if exp.seed is not None:
                random.seed(exp.seed)
                torch.manual_seed(exp.seed)
                cudnn.deterministic = True
                warnings.warn(
                    "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
                    "which can slow down your training considerably! You may see unexpected behavior "
                    "when restarting from checkpoints."
                )

            # set environment variables for distributed training
            configure_nccl()
            configure_omp()
            cudnn.benchmark = True

            trainer = exp.get_trainer(args)
            trainer.train()

        configure_module()
        args = self.config
        exp = get_exp(args.exp_file, args.name)
        exp.merge(args.opts)

        if not args.experiment_name:
            args.experiment_name = exp.exp_name

        num_gpu = get_num_devices() if args.devices is None else args.devices
        assert num_gpu <= get_num_devices()

        if args.cache is not None:
            exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)

        dist_url = "auto" if args.dist_url is None else args.dist_url
        launch(
            main,
            num_gpu,
            args.num_machines,
            args.machine_rank,
            backend=args.dist_backend,
            dist_url=dist_url,
            args=(exp, args),
        )
