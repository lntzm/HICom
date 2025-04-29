# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import re
import os
from abc import ABC, abstractmethod

import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from .projector import load_mm_projector, build_vision_projector
from .encoder import build_vision_tower
from ..constants import IGNORE_INDEX, NUM_FRAMES, MODAL_INDEX_MAP
from ..mm_utils import get_anyres_image_grid_shape, unpad_image, post_process_visual_feature
from ..utils import rank0_print


class HIComMetaModel:

    def __init__(self, config):
        super(HIComMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            # If delay_load is True, from_pretrained from ckpt will not load the vision tower,
            # vision_tower will be loaded from original CLIP. So delay_load=True is only for freezing vision_tower
            # If the vision_tower is finetuned, delay_load shuold be False
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.mm_projector = build_vision_projector(config)
            
            # if "unpad" in getattr(config, "mm_patch_merge_type", ""):
            if "anyres" in getattr(config, "image_aspect_ratio", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = self.config.mm_vision_tower if model_args.vision_tower is None else model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_weights = model_args.pretrain_weights
        mm_patch_merge_type = model_args.mm_patch_merge_type
        image_aspect_ratio = model_args.image_aspect_ratio

        self.config.mm_vision_tower = vision_tower

        # FIXME: Only can overwrite the vision tower config when self.get_vision_tower() is None
        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            
            if model_args.vision_tower is not None:
                vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        self.config.image_aspect_ratio = image_aspect_ratio

        self.config.delay_load = model_args.delay_load
        self.config.use_guide = model_args.use_guide
        self.config.max_num_frames = model_args.max_num_frames
        self.config.use_clip_scale = model_args.use_clip_scale

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
            if "anyres" in image_aspect_ratio:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)

        # else:
        #     # In case it is frozen by LoRA
        #     for p in self.mm_projector.parameters():
        #         p.requires_grad = True

        if pretrain_weights is not None:
            if os.path.exists(pretrain_weights):
                is_local = True
                if os.path.isdir(pretrain_weights):
                    mm_projector_weights = load_mm_projector(pretrain_weights)
                else:
                    mm_projector_weights = torch.load(pretrain_weights, map_location='cpu')
            else:
                # Support loading projector weights from remote HuggingFace model hub
                raise ValueError(f"{pretrain_weights} not found, loading from HuggingFace model hub is not supported yet.")
                is_local = False
                pretrain_weights = pretrain_weights.replace('mm_projector.bin', '')
                pretrain_weights = pretrain_weights.strip('/').strip('\\').strip()
                mm_projector_weights = load_mm_projector(pretrain_weights)

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            # set strict=False to avoid missing key error regarding bert.embeddings.position_ids
            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)
            rank0_print(f"Loaded mm_projector weights from {pretrain_weights}. Incompatible keys: {incompatible_keys}")


class HIComMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def num_frames(self):
        if hasattr(self.config, 'num_frames'):
            return self.config.num_frames
        else:
            return NUM_FRAMES

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images_or_videos(self, images, guided_input=None):
        # num_frames = self.config.num_frames if hasattr(self.config, 'num_frames') else NUM_FRAMES

        data_batch = []
        image_video_split_size = []
        image_sizes = []
        modalities = []
        for i, (data, image_size, modal) in enumerate(images):
            image_video_split_size.append(data.shape[0])
            data_batch.append(data)
            image_sizes.append(image_size)
            modalities.append(modal)

        frames = torch.cat(data_batch, dim=0)     # (b t) c h w

        frames_features, frames_embeds, guide_embeds = self.get_model().get_vision_tower()(frames, guided_input)
        frames_features = frames_features.split(image_video_split_size, dim=0)  # b t h w d
        if frames_embeds is not None:
            frames_embeds = frames_embeds.split(image_video_split_size, dim=0)  # b t h w d

        video_features = []
        for i, modal in enumerate(modalities):
            frames_feature = frames_features[i]    # t h w d
            frames_embed = frames_embeds[i] if frames_embeds is not None else None
            guide_embed = guide_embeds[i] if guide_embeds is not None else None
            image_size = image_sizes[i]
            if modal == "image" and frames_feature.shape[0] > 1:
                frames_feature = self.process_anyres_image_feature(frames_feature, image_size)
                if frames_embed is not None:
                    frames_embed = self.process_anyres_image_feature(frames_embed, image_size)

            video_feature = self.visual_compressor(frames_feature, frames_embed, guide_embed, modal)
            video_features.append(video_feature)
        
        return video_features

    def visual_compressor(self, frames_feature, frames_embed, guide_embed, modal):
        """Temporal aggregation of frames feature.
        Args:
            frames_features (torch.Tensor): Frames feature with shape (t, h, w, d).
        Returns:
            torch.Tensor: Video features with shape (t2, h2, w2, d).
        """
        mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        image_newline = getattr(self.model, "image_newline", None)
        # TODO: improve the merging method.
        # *********** mean pooling *************
        if self.config.mm_projector_type == "mlp2x_gelu" or self.config.mm_projector_type == "linear":
            if isinstance(frames_feature, dict):
                if frames_feature["base"] is not None:
                    base_video_feature = self.get_model().mm_projector(frames_feature["base"].unsqueeze(0))
                    base_video_feature = post_process_visual_feature(self.config, base_video_feature, modal, image_newline, is_anyres=False)
                patch_video_feature = self.get_model().mm_projector(frames_feature["patch"].unsqueeze(0))
                patch_video_feature = post_process_visual_feature(self.config, patch_video_feature, modal, image_newline, is_anyres=True)
                video_feature = torch.cat([base_video_feature, patch_video_feature], dim=-2) if frames_feature["base"] is not None else patch_video_feature
            else:
                video_feature = self.get_model().mm_projector(frames_feature)
                if modal == "video":
                    video_feature = einops.rearrange(video_feature, 't h w d -> d t h w')
                    scaled_shape = [video_feature.shape[-3], math.ceil(video_feature.shape[-2] / 2), math.ceil(video_feature.shape[-1] / 2)]
                    video_feature = F.interpolate(video_feature, size=scaled_shape, mode='trilinear')
                    video_feature = einops.rearrange(video_feature, 'd t h w -> t h w d')
                video_feature = post_process_visual_feature(self.config, video_feature, modal, image_newline, is_anyres=False)
                
            # video_feature = einops.rearrange(video_feature, 't h w d -> (t h w) d')
        else:
            video_feature = self.get_model().mm_projector(frames_feature, frames_embed, guide_embed, modal, image_newline)

        return video_feature

    def process_anyres_image_feature(self, image_feature, image_size):
        mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")

        if mm_patch_merge_type.startswith("spatial"):
            base_image_feature = image_feature[0]
            patch_image_feature = image_feature[1:]

            if "anyres_max" in image_aspect_ratio:
                matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                if matched_anyres_max_num_patches:
                    max_num_patches = int(matched_anyres_max_num_patches.group(1))
            
            assert image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio
            if hasattr(self.get_vision_tower(), "image_size"):
                vision_tower_image_size = self.get_vision_tower().image_size
            else:
                raise ValueError("vision_tower_image_size is not found in the vision tower.")
            try:
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_size, self.config.image_grid_pinpoints, vision_tower_image_size)
            except Exception as e:
                rank0_print(f"Error: {e}")
                num_patch_width, num_patch_height = 2, 2

            patch_image_feature = einops.rearrange(patch_image_feature, '(nh nw) h w d -> nh nw h w d', nh=num_patch_height, nw=num_patch_width)

            if "maxpool2x2" in mm_patch_merge_type:
                patch_image_feature = einops.rearrange(patch_image_feature, 'nh nw h w d -> d (nh h) (nw w)')
                patch_image_feature = F.max_pool2d(patch_image_feature, 2)
                patch_image_feature = einops.rearrange(patch_image_feature, 'd h w -> h w d')
            elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                unit = patch_image_feature.shape[2]
                patch_image_feature = einops.rearrange(patch_image_feature, 'nh nw h w d -> d (nh h) (nw w)')
                patch_image_feature = unpad_image(patch_image_feature, image_size)
                c, h, w = patch_image_feature.shape
                times = math.sqrt(h * w / (max_num_patches * unit**2))
                if times > 1.1:
                    patch_image_feature = patch_image_feature[None]
                    patch_image_feature = F.interpolate(patch_image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                patch_image_feature = einops.rearrange(patch_image_feature, 'd h w -> h w d')
            elif "unpad" in mm_patch_merge_type:
                patch_image_feature = einops.rearrange(patch_image_feature, 'nh nw h w d -> d (nh h) (nw w)')
                patch_image_feature = unpad_image(patch_image_feature, image_size)
                patch_image_feature = einops.rearrange(patch_image_feature, 'd h w -> h w d')
            else:
                patch_image_feature = einops.rearrange(patch_image_feature, 'nh nw h w d -> (nh h) (nw w) d')
            
            if "nobase" in mm_patch_merge_type:
                image_feature = {"base": None, "patch": patch_image_feature}
            else:
                # image_feature = torch.cat((base_image_feature.flatten(0,1), patch_image_feature), dim=0)
                image_feature = {"base": base_image_feature, "patch": patch_image_feature}
        
        return image_feature

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, guided_input=None
    ):
        vision_tower = self.get_vision_tower()
        # NOTE: text-only situation
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            # if past_key_values is not None and vision_tower is not None and Xs is not None and input_ids.shape[1] == 1:
            #    attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        mm_features = self.encode_images_or_videos(images, guided_input)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_mm_idx = 0
        # replace image/video/audio tokens with pre-computed embeddings
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_multimodals = sum((cur_input_ids == mm_token_idx).sum() for mm_token_idx in MODAL_INDEX_MAP.values())
            # pure text input
            if num_multimodals == 0:
                half_len = cur_input_ids.shape[0] // 2
                cur_mm_features = mm_features[cur_mm_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_mm_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_mm_idx += 1 
                continue

            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            mm_token_indices = torch.where(sum([cur_input_ids == mm_token_idx for mm_token_idx in MODAL_INDEX_MAP.values()]))[0]
            while mm_token_indices.numel() > 0:
                cur_mm_features = mm_features[cur_mm_idx]
                mm_token_start = mm_token_indices[0]

                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:mm_token_start])) 
                cur_new_input_embeds.append(cur_mm_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:mm_token_start])
                    cur_new_labels.append(torch.full((cur_mm_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[mm_token_start+1:]

                cur_mm_idx += 1
                cur_input_ids = cur_input_ids[mm_token_start+1:] 
                mm_token_indices = torch.where(sum([cur_input_ids == mm_token_idx for mm_token_idx in MODAL_INDEX_MAP.values()]))[0]

            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            # NOTE: one cur_new_input_embeds per each  
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        # padding
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels
