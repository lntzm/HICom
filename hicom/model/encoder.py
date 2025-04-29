import os
import einops
import torch
import torch.nn as nn

from typing import Dict
from functools import partial, reduce
from PIL import Image
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)

from safetensors import safe_open
from transformers import (
    CLIPVisionModel, CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPVisionConfig, CLIPTextModelWithProjection, CLIPTokenizerFast,
    SiglipModel, SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig, SiglipTextModel, SiglipTokenizer
)
from ..utils import rank0_print


class ModifiedSiglipImageProcessor:
    def __init__(
        self, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5), size=(384, 384),
        crop_size: Dict[str, int] = None, resample=PILImageResampling.BICUBIC,
        rescale_factor=1 / 255, data_format=ChannelDimension.FIRST
    ):
        crop_size = crop_size if crop_size is not None else {"height": 384, "width": 384}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            # to adapt video data
            images = [to_numpy_array(image) for image in images]
            assert isinstance(images, list)

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(resize, size=self.size, resample=self.resample, data_format=self.data_format),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
            partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)


def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor


class CLIPVisionTower(nn.Module):

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.guide_encoder_is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.use_guide = getattr(args, 'use_guide', None)

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        if not self.is_loaded:
            rank0_print("Loading vision_tower")
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            self.vision_tower = CLIPVisionModelWithProjection.from_pretrained(self.vision_tower_name)
            self.vision_tower._no_split_modules = []
            for name, param in self.vision_tower.named_parameters():
                if not param.data.is_contiguous():
                    param.data = param.data.contiguous()
            # self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
            self.vision_tower.requires_grad_(False)
            
            self.is_loaded = True
        else:
            rank0_print("vision_tower is already loaded")

        if self.use_guide not in [None, "off"]:
            if not self.guide_encoder_is_loaded:
                rank0_print("Loading guide_encoder")
                self.guide_encoder = CLIPTextModelWithProjection.from_pretrained(self.vision_tower_name)
                state_dict = torch.load(os.path.join(self.vision_tower_name, "pytorch_model.bin"))
                self.logit_scale = nn.Parameter(state_dict['logit_scale'])
                for name, param in self.guide_encoder.named_parameters():
                    if not param.data.is_contiguous():
                        param.data = param.data.contiguous()
                self.guide_tokenizer = CLIPTokenizerFast.from_pretrained(self.vision_tower_name)

                self.guide_encoder.requires_grad_(False)
                self.logit_scale.requires_grad_(False)
            else:
                rank0_print("guide_encoder is already loaded")

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    # @torch.no_grad()
    def forward(self, images, guided_input=None):
        if type(images) is list:
            raise NotImplementedError
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_feature = einops.rearrange(image_feature, 'b (h w) d -> b h w d', h=self.num_patches_per_side, w=self.num_patches_per_side)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            image_features = einops.rearrange(image_features, 'b (h w) d -> b h w d', h=self.num_patches_per_side, w=self.num_patches_per_side)

            if self.use_guide not in [None, "off"]:
                text_forward_out = self.guide_encoder(**guided_input.to(device=self.device))
                text_embeds = text_forward_out.text_embeds
                if self.select_feature == 'patch':
                    image_embeds = self.vision_tower.visual_projection(image_forward_outs.last_hidden_state[:, 1:, :])
                elif self.select_feature == 'cls_patch':
                    image_embeds = self.vision_tower.visual_projection(image_forward_outs.last_hidden_state)
                image_embeds = einops.rearrange(image_embeds, 'b (h w) d -> b h w d', h=self.num_patches_per_side, w=self.num_patches_per_side)
                
            else:
                text_embeds = None
                image_embeds = None

        return image_features, image_embeds, text_embeds

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size


class SiglipVisionTower(nn.Module):

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.guide_encoder_is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.use_guide = getattr(args, 'use_guide', None)

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        if not self.is_loaded:
            rank0_print("Loading vision_tower")
            image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
            self.image_processor = ModifiedSiglipImageProcessor(
                image_mean=image_processor.image_mean, image_std=image_processor.image_std, 
                size=(image_processor.size["height"], image_processor.size["width"]),
                rescale_factor=image_processor.rescale_factor
            )
            self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
            self.vision_tower.requires_grad_(False)
            
            self.is_loaded = True
        else:
            rank0_print("vision_tower is already loaded, skipping...")

        if self.use_guide not in [None, "off"]:
            if not self.guide_encoder_is_loaded:
                rank0_print("Loading guide_encoder")
                self.guide_encoder = SiglipTextModel.from_pretrained(self.vision_tower_name)
                self.guide_tokenizer = SiglipTokenizer.from_pretrained(self.vision_tower_name)
                
                self.guide_encoder.requires_grad_(False)
                self.guide_encoder_is_loaded = True
            else:
                rank0_print("guide_encoder is already loaded, skipping...")

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    # @torch.no_grad()
    def forward(self, images, guided_input=None):
        if type(images) is list:
            raise NotImplementedError
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_feature = einops.rearrange(image_feature, 'b (h w) d -> b h w d', h=self.num_patches_per_side, w=self.num_patches_per_side)
                image_features.append(image_feature.squeeze(0))
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            image_features = einops.rearrange(image_features, 'b (h w) d -> b h w d', h=self.num_patches_per_side, w=self.num_patches_per_side)
            
            if self.use_guide not in [None, "off"]:
                text_forward_out = self.guide_encoder(**guided_input.to(device=self.device))
                if self.use_guide == "fine":
                    text_embeds = text_forward_out.last_hidden_state
                    text_embeds = self.guide_encoder.text_model.head(text_embeds)
                else:
                    text_embeds = text_forward_out.pooler_output

                image_embeds = self.vision_tower.vision_model.head.layernorm(image_forward_outs.last_hidden_state)
                image_embeds = image_forward_outs.last_hidden_state + self.vision_tower.vision_model.head.mlp(image_embeds)
                image_embeds = einops.rearrange(image_embeds, 'b (h w) d -> b h w d', h=self.num_patches_per_side, w=self.num_patches_per_side)

            else:
                text_embeds = None
                image_embeds = None

        return image_features, image_embeds, text_embeds

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if 'clip' in vision_tower:
        vision_tower = CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'siglip' in vision_tower:
        vision_tower = SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')

    return vision_tower
