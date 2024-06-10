from sd_pipeline_typing.types import Config


class UDDEConfig(Config):
    def __init__(
        self,
        *,
        prompt: str,
        pre_H: int | None = None,
        pre_W: int | None = None,
        after_H: int | None = None,
        after_W: int | None = None,
    ):
        self.prompt = prompt
        self.pre_H = pre_H
        self.pre_W = pre_W
        self.after_H = after_H
        self.after_W = after_W
