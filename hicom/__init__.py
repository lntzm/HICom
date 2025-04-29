import os
import copy
import warnings
import shutil
from functools import partial

import torch

from .model import load_pretrained_model
from .mm_utils import process_image, process_video, tokenizer_multimodal_token, get_model_name_from_path, KeywordsStoppingCriteria
from .constants import NUM_FRAMES, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, MODAL_INDEX_MAP


def model_init(model_path, **kwargs):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, **kwargs)

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    
    aspect_ratio = getattr(model.config, "image_aspect_ratio", None)
    image_grid_pinpoints = getattr(model.config, "image_grid_pinpoints", None)
    image_crop_resolution = getattr(model.config, "image_crop_resolution", None)
    image_split_resolution = getattr(model.config, "image_split_resolution", None)

    processor = {
        'image': partial(
            process_image, processor=processor, aspect_ratio=aspect_ratio,
            image_grid_pinpoints=image_grid_pinpoints, image_crop_resolution=image_crop_resolution, image_split_resolution=image_split_resolution
        ),
        'video': partial(
            process_video, processor=processor, aspect_ratio=aspect_ratio,
            num_frames=model.config.num_frames
        ),
    }

    return model, processor, tokenizer


def mm_infer(image_or_video, instruct, model, tokenizer, modal='video', image_size=None, dtype=None, **kwargs):
    """inference api of HICom for video understanding.

    Args:
        model: HICom model.
        image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        modal (str): inference modality.
    Returns:
        str: response of the model.
    """

    # 1. text preprocess (tag process & generate prompt).
    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modal == 'text':
        modal_token = ''
    else:
        raise ValueError(f"Unsupported modal: {modal}")

    # 1. vision preprocess (load & transform image or video).
    if modal == 'text':
        tensor = None
    else:
        tensor = image_or_video.to(torch.float16 if dtype is None else dtype).cuda()
        tensor = [(tensor, image_size, modal)]

    # 2. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        message = [{'role': 'user', 'content': modal_token + '\n' + instruct}]
    elif isinstance(instruct, list):
        message = copy.deepcopy(instruct)
        message[0]['content'] = modal_token + '\n' + message[0]['content']
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")

    system_message = []

    message = system_message + message
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    use_guide = getattr(model.config, "use_guide", None)
    if use_guide not in [None, "off"]:
        guide_instruct = kwargs["guide_instruct"]
        guide_tokenizer = model.get_vision_tower().guide_tokenizer
        guided_input = guide_tokenizer(guide_instruct, return_tensors="pt", padding="max_length", truncation=True)
        for k in guided_input.keys():
            guided_input[k] = guided_input[k].cuda()
    else:
        guided_input = None
    
    # 3. generate response according to visual signals and prompts. 
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            guided_input=guided_input,
            images=tensor,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs
