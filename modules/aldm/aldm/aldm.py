import os
import random

from PIL import Image
import PIL
import numpy as np
import torch
import random

from einops import rearrange
from pytorch_lightning import seed_everything
from pathlib import Path

from sd_pipeline_typing.types import Module

from ._wrapper.cldm.model import create_model, load_state_dict
from ._wrapper.cldm.ddim_hacked import DDIMSampler
from ._wrapper.dataloader.carla_seg_helper import map_color_to_trainId
from ._wrapper.dataloader.cityscapes import CityscapesBaseInfo

from .config import ALDMConfig

H, W = 512, 1024
label_color_map = CityscapesBaseInfo().create_label_colormap()


def convert_labels(label, labels_info):
    lb_map = {el['id']: el['trainId'] for el in labels_info}
    for k, v in lb_map.items():
        label[label == k] = v
    return label


def read_label(label: Image.Image, labels_info):
    label = label.resize((W, H), PIL.Image.NEAREST)
    label = np.array(label).astype(np.int64)
    label = convert_labels(label, labels_info)
    label = torch.LongTensor(label)
    return label


def label_encode_color(label):
    # encode the mask using color coding
    # return label: Tensor [3,h,w], (-1,1)
    label_ = np.copy(label)
    label_[label_ == 255] = 19
    label_ = label_color_map[label_]
    return label_


def label_encode_id(label):
    # return label: Tensor [1,h,w]
    label_ = np.copy(label)
    label_[label_ == 255] = 19
    label_ = torch.from_numpy(label_)
    label_ = label_.unsqueeze(0)
    return label_


def convert_id_to_control(label_id, model):
    control = model.class_embedding_manager(label_id.unsqueeze(0))  # [bs, c, h, w]
    control = control.to(model.device)
    control = control.squeeze(0)
    return control


rand = random.Random()


class ALDM(Module):
    def __init__(self, *, config: ALDMConfig):
        self.config = config

        if self.config.random_seed and rand.seed is not None:
            rand.seed(self.config.seed)

    def run(self, input_data: dict[str, str | Image.Image], pipeline_config) -> dict[str, str | Image.Image]:
        config = self.config
        output = {}

        if config.random_seed:
            config.seed = rand.randint(0, 1000000)

        image = input_data["image"]
        img_name = Path(input_data["name"]).stem + f"_{config.seed}" + Path(input_data["name"]).suffix

        model_path = f'data/models/{config.model}'
        model_config = os.path.join(model_path, f"cldm_seg_{config.model}_multi_step_D.yaml")
        model_checkpoint = os.path.join(model_path, f"{config.model}_step9.ckpt")

        segmenter_config = None
        model = create_model(model_config, extra_segmenter_config=segmenter_config).cpu()

        model.load_state_dict(load_state_dict(model_checkpoint, location='cpu'), strict=False)
        model = model.cuda()
        ddim_sampler = DDIMSampler(model)

        if hasattr(model, 'model_attend'):
            model.model_attend.image_ratio = W / H
            model.model_attend.attention_store.image_ratio = W / H

        label = map_color_to_trainId(image, img_name, label_color_map, W, H)

        label_color = label_encode_color(label)  # (512, 1024, 3)
        Image.fromarray(label_color).save(os.path.join("debug", f"segmentation-{img_name}.png"))

        label_id = label_encode_id(label)  # (1, 512, 1024)
        control_cond = convert_id_to_control(label_id, model)  # (768, 128, 256)

        # Define the prompt here
        prompt = config.prompt if not config.neg else ''

        # # Define the negative prompt here
        n_prompt = config.prompt if config.neg else ''

        num_samples = config.num_samples
        ddim_steps = config.ddim_steps
        cfg_scale = config.cfg_scale
        seed = config.seed

        control = torch.stack([control_cond for _ in range(num_samples)], dim=0)
        control = control.clone().cuda()

        cond = {"c_concat": [control],
                "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)]}
        un_cond = {"c_concat": [control],
                   "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)
        seed_everything(seed)
        samples, intermediates = ddim_sampler.sample(
            ddim_steps, num_samples,
            shape, cond, verbose=False, eta=0.0,
            unconditional_guidance_scale=cfg_scale,
            unconditional_conditioning=un_cond
        )

        x_samples = model.decode_first_stage(samples)
        x_samples = (rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(
            np.uint8)

        for sample in x_samples:
            sample_pil = Image.fromarray(sample)

        output["name"] = img_name
        output["image"] = sample_pil

        return output
