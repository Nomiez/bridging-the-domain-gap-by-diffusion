from sd_pipeline_typing.types import Config


class YoloXConfig(Config):
    def __init__(
        self,
        *,
        experiment_name: str | None = None,
        name: str | None = None,
        dist_backend: str | None = "nccl",
        dist_url: str | None = None,
        batch_size: int | None = 64,
        devices: int | None = None,
        exp_file: str | None = None,
        resume: bool | None = False,
        ckpt: str | None = None,
        start_epoch: int | None = None,
        num_machines: int | None = 1,
        machine_rank: int | None = 0,
        fp16: bool | None = False,
        cache: str | None = None,
        occupy: bool | None = False,
        logger: str | None = "tensorboard",
        opts: list[str] | None = None,
    ):
        self.experiment_name = experiment_name
        self.name = name
        self.dist_backend = dist_backend
        self.dist_url = dist_url
        self.batch_size = batch_size
        self.devices = devices
        self.exp_file = exp_file
        self.resume = resume
        self.ckpt = ckpt
        self.start_epoch = start_epoch
        self.num_machines = num_machines
        self.machine_rank = machine_rank
        self.fp16 = fp16
        self.cache = cache
        self.occupy = occupy
        self.logger = logger
        self.opts = opts

    def __dict__(self):
        return {
            "experiment_name": self.experiment_name,
            "name": self.name,
            "dist_backend": self.dist_backend,
            "dist_url": self.dist_url,
            "batch_size": self.batch_size,
            "devices": self.devices,
            "exp_file": self.exp_file,
            "resume": self.resume,
            "ckpt": self.ckpt,
            "start_epoch": self.start_epoch,
            "num_machines": self.num_machines,
            "machine_rank": self.machine_rank,
            "fp16": self.fp16,
            "cache": self.cache,
            "occupy": self.occupy,
            "logger": self.logger,
            "opts": self.opts,
        }
