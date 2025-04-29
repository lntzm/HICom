import os
import re
import math
import copy
import torch
import einops
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Tuple
from torch.nn.init import trunc_normal_
from transformers import AutoModel, TRANSFORMERS_CACHE
from transformers.integrations import is_deepspeed_zero3_enabled
from hicom.mm_utils import post_process_visual_feature


def parse_snapshot_folder(repo_id, cache_dir=None, repo_type="model"):
    revision = "main"
    # 1. parse the downloaded cache folder
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    else:
        cache_dir = cache_dir
    object_id = repo_id.replace("/", "--")
    repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
    # 2. resolve refs (for instance to convert main to the associated commit sha)
    refs_dir = os.path.join(repo_cache, "refs")
    if os.path.isdir(refs_dir):
        revision_file = os.path.join(refs_dir, revision)
        if os.path.isfile(revision_file):
            with open(revision_file) as f:
                revision = f.read()
    # 3. acquire the snapshot folder
    folder = os.path.join(repo_cache, "snapshots", revision)

    return folder


def load_mm_projector(model_path, cache_dir=None, token=None):
    if os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
        is_local = True
        folder = model_path
    else:
        is_local = False
        folder = parse_snapshot_folder(model_path, cache_dir=cache_dir, repo_type="model")
        if not os.path.exists(os.path.join(folder, 'mm_projector.bin')):
            # downloading from remote repo
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_path, cache_dir=cache_dir, token=token)

    mm_projector_weights = torch.load(os.path.join(folder, 'mm_projector.bin'), map_location='cpu')
    mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    return mm_projector_weights


def get_3d_position_embedding(t, h, w, d_model):
    """
    生成3D余弦位置编码。
    
    参数:
    t (int): 时间维度长度
    h (int): 高度维度长度
    w (int): 宽度维度长度
    d_model (int): 嵌入维度
    
    返回:
    pos_encoding (np.ndarray): 维度为 (t, h, w, d_model) 的位置编码
    """
    def get_angles(pos, i, d_model):
        return pos / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    
    angle_rads_t = get_angles(np.arange(t)[:, np.newaxis],
                              np.arange(d_model)[np.newaxis, :],
                              d_model)
    angle_rads_h = get_angles(np.arange(h)[:, np.newaxis],
                              np.arange(d_model)[np.newaxis, :],
                              d_model)
    angle_rads_w = get_angles(np.arange(w)[:, np.newaxis],
                              np.arange(d_model)[np.newaxis, :],
                              d_model)
    
    pos_encoding_t = np.zeros_like(angle_rads_t)
    pos_encoding_t[:, 0::2] = np.sin(angle_rads_t[:, 0::2])
    pos_encoding_t[:, 1::2] = np.cos(angle_rads_t[:, 1::2])
    
    pos_encoding_h = np.zeros_like(angle_rads_h)
    pos_encoding_h[:, 0::2] = np.sin(angle_rads_h[:, 0::2])
    pos_encoding_h[:, 1::2] = np.cos(angle_rads_h[:, 1::2])
    
    pos_encoding_w = np.zeros_like(angle_rads_w)
    pos_encoding_w[:, 0::2] = np.sin(angle_rads_w[:, 0::2])
    pos_encoding_w[:, 1::2] = np.cos(angle_rads_w[:, 1::2])
    
    pos_encoding_t = pos_encoding_t[:, np.newaxis, np.newaxis, :]
    pos_encoding_h = pos_encoding_h[np.newaxis, :, np.newaxis, :]
    pos_encoding_w = pos_encoding_w[np.newaxis, np.newaxis, :, :]
    
    pos_encoding = pos_encoding_t + pos_encoding_h + pos_encoding_w
    
    return pos_encoding


class IdentityMap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    # @property
    # def config(self):
    #     return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.,):
        super().__init__()
        # self.config = config
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        logit_scale: Optional[torch.Tensor] = None,
        logit_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = query.size()
        kv_len = key.shape[1]

        query_states = self.q_proj(query)
        key_states = self.k_proj(key)
        value_states = self.v_proj(value)

        if logit_scale is not None:
            query_states = query_states / query_states.norm(p=2, dim=-1, keepdim=True)
            key_states = key_states / key_states.norm(p=2, dim=-1, keepdim=True)
            scale = logit_scale.exp()
            bias = logit_bias
        else:
            scale = self.scale
            bias = 0.

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scale + bias

        if attn_weights.size() != (batch_size, self.num_heads, q_len, kv_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, kv_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, kv_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, q_len, kv_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "linear":
        # NOTE: for both linear and mlp2x_gelu projector type, mean pooling is adopted to aggreate video features
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    
    local_compressor = global_compressor = None
    if "local" in projector_type:
        local_phase = projector_type.split("local")[-1].split("global")[0]
        local_num = ''
        for s in local_phase:
            if s.isdigit():
                local_num += s
            else:
                break
        temporal_kernel_size = int(local_num[0])
        if len(local_num) == 2:
            spatial_kernel_size = int(local_num[1])
        elif len(local_num) == 3:
            spatial_kernel_size = int(local_num[1:3])
        
        adapt_q = adapt_k = adapt_v = adapt_guide = False
        if 'adapt' in local_phase:
            for s in local_phase.split("adapt")[-1]:
                if s == 'q':
                    adapt_q = True
                elif s == 'k':
                    adapt_k = True
                elif s == 'v':
                    adapt_v = True
                elif s == 'g':
                    adapt_guide = True
                else:
                    break
        
        force_use_guide = False
        if 'guide' in local_phase:
            force_use_guide = local_phase.split("guide")[-1].split("_")[0]
        
        local_compressor = LocalCompressor(
            config, temporal_kernel_size, spatial_kernel_size, 
            adapt_q, adapt_k, adapt_v, adapt_guide, force_use_guide=force_use_guide
        )

    if "global" in projector_type:
        global_phase = projector_type.split("global")[-1].split("local")[0]
        global_num = ''
        for s in global_phase:
            if s.isdigit():
                global_num += s
            else:
                break
        num_queries = int(global_num)
        use_pos_emb = True
        adapt_guide = 'adaptg' in global_phase
        force_use_guide = False
        if 'guide' in global_phase:
            force_use_guide = global_phase.split("guide")[-1].split("_")[0]

        global_compressor = GlobalCompressor(
            config, num_queries, use_pos_emb,
            adapt_guide, force_use_guide=force_use_guide
        )
    
    return HIComProjector(config, local_compressor, global_compressor)


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class GuideInjector(nn.Module):
    def __init__(
        self, use_guide, text_dim, qk_dim, adapt_guide=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), mlp_depth=2
    ):
        super().__init__()
        self.use_guide = use_guide

        if text_dim != qk_dim:
            self.text2qk_proj = build_mlp(mlp_depth, text_dim, qk_dim)
        else:
            self.text2qk_proj = nn.Identity()
        
        if adapt_guide:
            self.guide_proj = build_mlp(mlp_depth, qk_dim, qk_dim)
            self.guide_norm = norm_layer(qk_dim)
            self.guide_alpha = nn.Parameter(torch.zeros(1))
        else:
            self.guide_proj = nn.Identity()
            self.guide_norm = nn.Identity()
            self.guide_alpha = 0
        
        if self.use_guide == "coarse":
            self.coarse_proj = build_mlp(mlp_depth, qk_dim, qk_dim * 2)
            self.coarse_norm = norm_layer(qk_dim)
        elif self.use_guide == "fine":
            self.fine_proj = MultiheadAttention(qk_dim, num_heads=qk_dim // 128)
            self.fine_norm = norm_layer(qk_dim)
    
    def forward(self, visual_embed, guide_embed):
        if self.use_guide in ["direct", "coarse"]:
            return self.forward_direct_and_coarse(visual_embed, guide_embed)
        elif self.use_guide in ["fine"]:
            return self.forward_fine(visual_embed, guide_embed)
        else:
            raise NotImplementedError

    def forward_direct_and_coarse(self, visual_embed, guide_embed):
        if visual_embed.ndim == 4:
            t, h, w = visual_embed.shape[:3]
            guide_embed = einops.rearrange(guide_embed, 'd -> 1 1 1 d')
            guide_embed = guide_embed.repeat(t, h, w, 1)
        elif visual_embed.ndim == 2:
            n = visual_embed.shape[0]
            guide_embed = einops.rearrange(guide_embed, 'd -> 1 d')
            guide_embed = guide_embed.repeat(n, 1)
        else:
            raise ValueError("Invalid input shape for guide embedding.")
        
        guide_embed = self.text2qk_proj(guide_embed)
        guide_embed = (1 - self.guide_alpha) * guide_embed + self.guide_alpha * self.guide_norm(self.guide_proj(guide_embed))

        if self.use_guide == "direct":
            return guide_embed
        elif self.use_guide == "coarse":
            guide = self.coarse_proj(guide_embed)
            scale, shift = torch.chunk(guide, 2, dim=-1)
            return self.coarse_norm(visual_embed * (1 + scale) + shift)

    def forward_fine(self, visual_embed, guide_embed):
        if visual_embed.ndim == 4:
            t, h, w = visual_embed.shape[:3]
            query = einops.rearrange(visual_embed, 't h w d -> (t h w) 1 d')
            guide_embed = einops.rearrange(guide_embed, 'l d -> 1 l d')
            guide_embed = guide_embed.repeat(t*h*w, 1, 1)
        elif visual_embed.ndim == 2:
            n = visual_embed.shape[0]
            query = einops.rearrange(visual_embed, 'n d -> 1 n d')
            guide_embed = einops.rearrange(guide_embed, 'l d -> 1 l d')
            # guide_embed = guide_embed.repeat(n, 1, 1)
        else:
            raise ValueError("Invalid input shape for guide embedding.")
        
        guide_embed = self.text2qk_proj(guide_embed)
        guide_embed = (1 - self.guide_alpha) * guide_embed + self.guide_alpha * self.guide_norm(self.guide_proj(guide_embed))

        attn_out, _ = self.fine_proj(query, guide_embed, guide_embed)
        injected_out = self.fine_norm(query + attn_out)

        if visual_embed.ndim == 4:
            return einops.rearrange(injected_out, '(t h w) 1 d -> t h w d', t=t, h=h, w=w)
        else:
            return einops.rearrange(injected_out, '1 n d -> n d')

class LocalCompressor(nn.Module):
    def __init__(
        self, config, temporal_kernel_size=4, spatial_kernel_size=2,
        adapt_q=False, adapt_k=False, adapt_v=False, adapt_guide=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mlp_depth=2, force_use_guide=False
    ):
        super().__init__()
        if 'siglip-so400m-patch14-384' in config.mm_vision_tower:
            qk_dim = 1152
            hw = 27
        elif 'clip-vit-large-patch14-336' in config.mm_vision_tower:
            qk_dim = 768
            hw = 24
        else:
            raise NotImplementedError
        encoder_hidden_size = config.mm_hidden_size
        output_hidden_size = config.hidden_size
        self.qk_dim = qk_dim

        self.spatial_kernel_size = spatial_kernel_size
        self.temporal_kernel_size = temporal_kernel_size

        self.use_guide = getattr(config, "use_guide", None) if force_use_guide is False else force_use_guide
        if self.use_guide in [None, "off"]:
            self.guide_injector = IdentityMap()
        else:
            self.guide_injector = GuideInjector(self.use_guide, qk_dim, qk_dim, adapt_guide, norm_layer, mlp_depth)
        
        if self.use_guide == "direct":
            adapt_q = False

        if adapt_q:
            # self.q_proj = build_mlp(mlp_depth, qk_dim, qk_dim)
            self.q_proj = nn.Linear(qk_dim, qk_dim, bias=False)
            self.q_norm = norm_layer(qk_dim)
            self.q_alpha = nn.Parameter(torch.zeros(1))
        else:
            self.q_proj = nn.Identity()
            self.q_norm = nn.Identity()
            self.q_alpha = 0

        if adapt_k:
            self.k_proj = build_mlp(mlp_depth, qk_dim, qk_dim)
            self.k_norm = norm_layer(qk_dim)
            self.k_alpha = nn.Parameter(torch.zeros(1))
        else:
            self.k_proj = nn.Identity()
            self.k_norm = nn.Identity()
            self.k_alpha = 0

        if adapt_v:
            self.v_proj = build_mlp(mlp_depth, encoder_hidden_size, encoder_hidden_size)
            self.v_norm = norm_layer(encoder_hidden_size)
            self.v_alpha = nn.Parameter(torch.zeros(1))
        else:
            self.v_proj = nn.Identity()
            self.v_norm = nn.Identity()
            self.v_alpha = 0
        
        self.readout = build_mlp(mlp_depth, encoder_hidden_size, output_hidden_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def divide_feature(self, x, kernel_size):
        t, h, w = x.shape[:3]

        if t % kernel_size[0] == 0:
            t_reshape_x = einops.rearrange(x, '(t1 t2) h w d -> t2 t1 h w d', t2=kernel_size[0])
        else:
            t_reshape_x = self.balance_divide_feature(x, kernel_size[0])

        t_reshape_x = einops.rearrange(t_reshape_x, 't2 t1 h w d -> h t2 t1 w d')
        if h % kernel_size[1] == 0:
            th_reshape_x = einops.rearrange(t_reshape_x, '(h1 h2) t2 t1 w d -> h2 h1 t2 t1 w d', h2=kernel_size[1])
        else:
            th_reshape_x = self.balance_divide_feature(t_reshape_x, kernel_size[1])

        th_reshape_x = einops.rearrange(th_reshape_x, 'h2 h1 t2 t1 w d -> w h2 h1 t2 t1 d')
        if w % kernel_size[2] == 0:
            thw_reshape_x = einops.rearrange(th_reshape_x, '(w1 w2) h2 h1 t2 t1 d -> w2 w1 h2 h1 t2 t1 d', w2=kernel_size[2])
        else:
            thw_reshape_x = self.balance_divide_feature(th_reshape_x, kernel_size[2])

        reshape_x = einops.rearrange(thw_reshape_x, 'w2 w1 h2 h1 t2 t1 d -> (t1 h1 w1) (t2 h2 w2) d')
        
        # reshape_x = einops.rearrange(
        #     x, 'b (t1 t2) (h1 h2) (w1 w2) d -> (b t1 h1 w1) (t2 h2 w2) d', 
        #     t2=kernel_size[0], h2=kernel_size[1], w2=kernel_size[2]
        # )
        return reshape_x

    def balance_divide_feature(self, x, kernel_size):
        # split_size是分成多少组
        # kernel_size是感受野的大小
        t = x.shape[0]
        split_size = math.ceil(t / kernel_size)

        # repeat_num = group_size * split_size - t
        no_repeat_num = t % split_size
        if no_repeat_num == 0:
            no_repeat_num = split_size
        split_idx_list = [kernel_size - (0 if i < no_repeat_num else 1)for i in range(split_size)]
        start_idx = 0
        splited_x = []
        for i in range(split_size):
            end_idx = start_idx + split_idx_list[i]
            if split_idx_list[i] < kernel_size:
                start_idx -= 1
            splited_x.append(x[start_idx:end_idx, ...])
            start_idx = end_idx
        splited_x = torch.stack(splited_x, dim=1)
        # splited_x = einops.rearrange(splited_x, 'b t_kernel t_split h w d -> t_kernel (b t_split) h w d')
        return splited_x
    
    def forward(self, frames_feature, frames_embed, guide_embed, modal, logit_scale, logit_bias):
        t, h, w = frames_feature.shape[:3]

        if frames_embed is not None and logit_scale is not None:
            frames_embed = frames_embed / frames_embed.norm(p=2, dim=-1, keepdim=True)
            guide_embed = guide_embed / guide_embed.norm(p=2, dim=-1, keepdim=True)

        # query = self.q_norm(self.q_proj(guide_embed))
        frames_embed = frames_feature if frames_embed is None else frames_embed
        key = (1 - self.k_alpha) * frames_embed + self.k_alpha * self.k_norm(self.k_proj(frames_embed))
        value = (1 - self.v_alpha) * frames_feature + self.v_alpha * self.v_norm(self.v_proj(frames_feature))

        temporal_kernel_size = 1 if (modal == "image" or t == 1) else self.temporal_kernel_size
        downsampled_size = (math.ceil(t/temporal_kernel_size), math.ceil(h/self.spatial_kernel_size), math.ceil(w/self.spatial_kernel_size))

        q = F.interpolate(einops.rearrange(frames_feature, 't h w d -> 1 d t h w'), size=downsampled_size, mode='trilinear')
        q = einops.rearrange(q, '1 d t h w -> t h w d')
        q = (1 - self.q_alpha) * q + self.q_alpha * self.q_norm(self.q_proj(q))
        query = self.guide_injector(q, guide_embed)

        reshape_key = self.divide_feature(key, (temporal_kernel_size, self.spatial_kernel_size, self.spatial_kernel_size))
        reshape_value = self.divide_feature(value, (temporal_kernel_size, self.spatial_kernel_size, self.spatial_kernel_size))
        reshape_query = self.divide_feature(query, (1, 1, 1))

        if logit_scale is not None:
            attn = torch.softmax(torch.bmm(reshape_query, reshape_key.permute(0,2,1)) * logit_scale.exp() + logit_bias, dim=-1)
        else:
            attn = torch.softmax(torch.bmm(reshape_query, reshape_key.permute(0,2,1)) / math.sqrt(self.qk_dim), dim=-1)

        out = torch.bmm(attn, reshape_value)
        x = einops.rearrange(
            out, '(t1 h1 w1) (t2 h2 w2) d -> (t1 t2) (h1 h2) (w1 w2) d',
            t2=1, h2=1, w2=1,
            t1=downsampled_size[0], h1=downsampled_size[1], w1=downsampled_size[2]
        )
        return self.readout(x)


class GlobalCompressor(nn.Module):
    def __init__(
        self, config, num_queries, use_pos_emb=True,
        adapt_guide=False, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mlp_depth=2, force_use_guide=False
    ):
        super().__init__()
        if 'siglip-so400m-patch14-384' in config.mm_vision_tower:
            text_dim = 1152
            hw = 27
        elif 'clip-vit-large-patch14-336' in config.mm_vision_tower:
            text_dim = 768
            hw = 24
        else:
            raise NotImplementedError
        self.embed_dim = embed_dim = config.mm_hidden_size
        output_hidden_size = config.hidden_size
        num_heads = embed_dim // 128
        self.use_pos_emb = use_pos_emb
        max_num_frames = getattr(config, "max_num_frames", 256)

        self.query = nn.Parameter(torch.zeros(num_queries, embed_dim))

        self.use_guide = getattr(config, "use_guide", None) if force_use_guide is False else force_use_guide
        if self.use_guide in [None, "off"]:
            self.guide_injector = IdentityMap()
        else:
            self.guide_injector = GuideInjector(self.use_guide, text_dim, embed_dim, adapt_guide, norm_layer, mlp_depth)
        
        # self.attn_layer = nn.TransformerDecoderLayer(
        #     d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim,
        #     activation="gelu", batch_first=True
        # )
        self.attn_layer = MultiheadAttention(embed_dim, num_heads)
        self.readout = build_mlp(mlp_depth, embed_dim, output_hidden_size)

        self.apply(self._init_weights)
        if self.use_pos_emb:
            self.max_size = [max_num_frames, hw, hw]
            self._set_3d_pos_cache(max_num_frames, hw, hw)

    def _set_3d_pos_cache(self, max_t, max_h, max_w, device='cpu'):
        if is_deepspeed_zero3_enabled():
            device='cuda'
        pos_embed = torch.from_numpy(get_3d_position_embedding(max_t, max_h, max_w, self.embed_dim)).float().to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)
    
    def _adjust_pos_cache(self, tgt_sizes, device):
        any_change = False
        if tgt_sizes[0] > self.max_size[0]:
            any_change = True
            self.max_size[0] = tgt_sizes[0]
        if tgt_sizes[1] > self.max_size[1]:
            any_change = True
            self.max_size[1] = tgt_sizes[1]
        if tgt_sizes[2] > self.max_size[2]:
            any_change = True
            self.max_size[2] = tgt_sizes[2]
        if any_change:
            self._set_3d_pos_cache(self.max_size[0], self.max_size[1], self.max_size[2], device)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(self, frames_feature, frames_embed, guide_embed, modal, logit_scale, logit_bias):
        t, h, w = frames_feature.shape[:3]
        if self.use_pos_emb:
            self._adjust_pos_cache((t, h, w), frames_feature.device)
            pos_embed = self.pos_embed[:t, :h, :w, :].to(frames_feature.dtype)
            # pos_embed = einops.rearrange(pos_embed, 't h w d -> 1 (t h w) d')
            frames_feature = frames_feature + pos_embed
        
        query = self.guide_injector(self.query, guide_embed)
        batch_q = einops.rearrange(query, 'n d -> 1 n d')
        batch_kv = einops.rearrange(frames_feature, 't h w d -> 1 (t h w) d')
        x, attn = self.attn_layer(batch_q, batch_kv, batch_kv, logit_scale=logit_scale, logit_bias=logit_bias)
        return self.readout(query + x.squeeze(0))


class HIComProjector(nn.Module):
    def __init__(
        self, config, local_compressor=None, global_compressor=None,
    ):
        super().__init__()
        self.config = config
        use_clip_scale = getattr(config, 'use_clip_scale', '').split(',')
        self.local_use_clip_scale = 'local' in use_clip_scale
        self.global_use_clip_scale = 'global' in use_clip_scale
        self.local_logit_scale, self.local_logit_bias = None, None
        self.global_logit_scale, self.global_logit_bias = None, None
        if self.local_use_clip_scale or self.global_use_clip_scale:
            clip_model = AutoModel.from_pretrained(config.mm_vision_tower)
            logit_scale = clip_model.logit_scale
            logit_bias = clip_model.logit_bias
            # self.logit_scale.requires_grad = False
            # self.logit_bias.requires_grad = False
            del clip_model
        if self.local_use_clip_scale:
            self.local_logit_scale, self.local_logit_bias = logit_scale, logit_bias
        if self.global_use_clip_scale:
            self.global_logit_scale, self.global_logit_bias = copy.deepcopy(logit_scale), copy.deepcopy(logit_bias)
        
        self.local_compressor = local_compressor
        self.global_compressor = global_compressor
        assert local_compressor is not None or global_compressor is not None, "At least one compressor should be provided."

    def forward(self, frames_feature, frames_embed, guide_embed, modal, image_newline=None):
        local_x = global_x = None
        if self.local_compressor is not None:
            if isinstance(frames_feature, dict):
                if frames_feature["base"] is not None:
                    base_frames_feature = frames_feature["base"].unsqueeze(0)
                    base_frames_embed = frames_embed["base"].unsqueeze(0) if frames_embed is not None else None
                    base_local_x = self.local_compressor(base_frames_feature, base_frames_embed, guide_embed, modal, self.local_logit_scale, self.local_logit_bias)
                    base_local_x = post_process_visual_feature(self.config, base_local_x, modal, image_newline, is_anyres=False)
                patch_frames_feature = frames_feature["patch"].unsqueeze(0)
                patch_frames_embed = frames_embed["patch"].unsqueeze(0) if frames_embed is not None else None
                patch_local_x = self.local_compressor(patch_frames_feature, patch_frames_embed, guide_embed, modal, self.local_logit_scale, self.local_logit_bias)
                patch_local_x = post_process_visual_feature(self.config, patch_local_x, modal, image_newline, is_anyres=True)
                local_x = torch.cat([base_local_x, patch_local_x], dim=-2) if frames_feature["base"] is not None else patch_local_x
            else:
                local_x = self.local_compressor(frames_feature, frames_embed, guide_embed, modal, self.local_logit_scale, self.local_logit_bias)
                local_x = post_process_visual_feature(self.config, local_x, modal, image_newline, is_anyres=False)

        if self.global_compressor is not None:
            if isinstance(frames_feature, dict):
                patch_frames_feature = frames_feature["patch"].unsqueeze(0)
                patch_frames_embed = frames_embed["patch"].unsqueeze(0) if frames_embed is not None else None
                global_x = self.global_compressor(patch_frames_feature, patch_frames_embed, guide_embed, modal, self.global_logit_scale, self.global_logit_bias)
            else:
                global_x = self.global_compressor(frames_feature, frames_embed, guide_embed, modal, self.global_logit_scale, self.global_logit_bias)
        
        if local_x is None:
            return global_x
        elif global_x is None:
            return local_x
        
        x = torch.cat([local_x, global_x], dim=-2)
        return x

