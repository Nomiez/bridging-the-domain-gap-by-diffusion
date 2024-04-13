import os
from PIL import Image
import PIL

import json
from einops import rearrange
from pytorch_lightning import seed_everything

from ..._wrapper.aldm.cldm.model import create_model, load_state_dict
from ..._wrapper.aldm.cldm.ddim_hacked import DDIMSampler
from ..._wrapper.aldm.dataloader.cityscapes import CityscapesBaseInfo
from ..._wrapper.aldm.dataloader.custom_transform import *

from ...components import Module
from ...components.pipeline_config import PipelineConfig
from .config import ALDMConfig

H, W = 512, 1024
label_color_map = CityscapesBaseInfo().create_label_colormap(version='ade20k')


def convert_labels(label, labels_info):
    lb_map = {el['id']: el['trainId'] for el in labels_info}
    for k, v in lb_map.items():
        label[label == k] = v
    return label


def read_label(label_path, labels_info):
    label = Image.open(label_path)
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


class ALDM(Module):
    def __init__(self, *, config: ALDMConfig):
        self.config = config

    def run(self, input_data: dict[str, str], pipeline_config: PipelineConfig):
        model_config = 'models/cldm_seg_ade20k_multi_step_D.yaml'
        checkpoint_dir = os.path.join(pipeline_config.checkpoint_dir, "ade20k_step9.ckpt")

        with open(os.path.join(pipeline_config.checkpoint_dir, "info/cityscapes_info.json"), 'r') as fr:
            labels_info = json.load(fr)

        segmenter_config = None
        model = create_model(model_config, extra_segmenter_config=segmenter_config).cpu()

        model.load_state_dict(load_state_dict(checkpoint_dir, location='cpu'), strict=False)
        model = model.cuda()
        ddim_sampler = DDIMSampler(model)

        if hasattr(model, 'model_attend'):
            model.model_attend.image_ratio = W / H
            model.model_attend.attention_store.image_ratio = W / H

        test_label_path = input_data['label_path']

        label = read_label(test_label_path, labels_info)
        label_color = label_encode_color(label)  # (512, 1024, 3)
        label_id = label_encode_id(label)  # (1, 512, 1024)
        control_cond = convert_id_to_control(label_id, model)  # (768, 128, 256)

        # Define the prompt here
        prompt = self.config.prompt if not self.config.neg else ''

        # # Define the negative prompt here
        n_prompt = self.config.prompt if self.config.neg else ''

        num_samples = self.config.num_samples
        ddim_steps = self.config.ddim_steps
        cfg_scale = self.config.cfg_scale
        seed = self.config.seed

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

        sample_pil.save(f'{self.config.output_dir}/test_output_{seed}.png')
