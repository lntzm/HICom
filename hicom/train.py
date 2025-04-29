# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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
import ast
import copy
import json
import yaml
import math
import random
import pathlib
import traceback
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

# torch-related packages
# NOTE: torch must be imported before transformers. Otherwise, `Segmentation fault (core dumped)` will occur.
import torch
from torch.utils.data import Dataset

import transformers
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

import sys
sys.path.append('./')
from hicom.model import *
from hicom.constants import NUM_FRAMES, IGNORE_INDEX, MODAL_INDEX_MAP
from hicom.mm_utils import (tokenizer_multimodal_token, 
    process_video, process_image, extract_guided_prompt
)
from hicom.utils import rank0_print, check_ckpt_exists
from hicom.hicom_trainer import (HIComTrainer,
    get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, 
    find_all_linear_names, safe_save_model_for_hf_trainer
)

# NOTE: fast tokenizer warning issue: https://github.com/huggingface/transformers/issues/5486   
os.environ["TOKENIZERS_PARALLELISM"] = "true"

local_rank = None



def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ModelArguments:
    # LLM Arguments
    model_type: Optional[str] = field(default="hicom", metadata={"help": "Model type selected in the list: " + ", ".join(VLLMs.keys())})
    model_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")
    version: Optional[str] = field(default="v1", metadata={"help": "Version of the conversation template."})
    freeze_backbone: bool = field(default=False, metadata={"help": "Whether to freeze the LLM backbone."})
    # Connector Arguments
    mm_projector_type: Optional[str] = field(default='linear')
    # tune_mm_mlp_adapter: bool = field(default=False)
    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_projector", "mm_projector,language_model", "vision_tower,mm_projector,language_model"'}
    )
    pretrain_weights: Optional[str] = field(default=None)
    # Vision tower Arguments
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_newline_position: Optional[str] = field(default="no_token")
    delay_load: Optional[bool] = field(default=False)

    use_guide: str = field(default=None)
    max_num_frames: Optional[int] = field(default=256)
    use_clip_scale: str = field(default='')


@dataclass
class DataArguments:
    # Path Arguments
    data_path: List[str] = field(default=None, metadata={"help": "Path to the training data."})
    # image_folder: Optional[str] = field(default=None)
    # video_folder: Optional[str] = field(default=None)
    data_folder: Optional[str] = field(default=None)
    # Loading Arguments
    is_multimodal: bool = False
    lazy_preprocess: bool = False
    num_frames: Optional[int] = field(default=None)
    # Preprocess Arguments
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    add_time_instruction: Optional[bool] = field(default=False)
    is_pretraining: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    guide_injector_lr: Optional[float] = None
    freeze_mm_mlp_adapter: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    # Training Data Arguments 
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    modal_token: str = None,
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    conversations = []
    input_ids = []
    targets = []
    for source in sources:
        # 1. apply chat template for input conversation
        assert len(source) == 2
        assert modal_token in source[0]['value']
        message = [
            {'role': 'user', 'content': modal_token},
            {'role': 'assistant', 'content': source[1]['value']}
        ]
        conversation = " ".join([sentence['value'] for sentence in source])

        input_id = tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt')
        target = copy.deepcopy(input_id)
        target[input_id == MODAL_INDEX_MAP[modal_token]] = IGNORE_INDEX

        input_ids.append(input_id)
        targets.append(target)

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    modal_token: str = None,
    process_guided: bool = False,
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    # Apply prompt templates
    conversations = []
    input_ids = []
    targets = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != "user":
            # Skip the first one if it is not from human
            source = source[1:]

        message = [{'role': roles[sentence['from']], 'content': sentence['value']} for sentence in source]
        conversation = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        input_ids.append(tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt'))
        targets.append(copy.deepcopy(input_ids[-1]))

        assert len(source) % 2 == 0, f"Invalid conversation length {len(source)}."

        cur = 0
        message = []
        for idx, sentence in enumerate(source):
            if idx % 2 == 1:
                tmp_message = [
                    {'role': roles[source[idx-1]['from']], 'content': source[idx-1]['value']}, 
                    {'role': roles[sentence['from']], 'content': sentence['value']}
                ]

                instruction = tokenizer.apply_chat_template(message + tmp_message[:1], tokenize=False, add_generation_prompt=True)
                conversation = tokenizer.apply_chat_template(message + tmp_message, tokenize=False, add_generation_prompt=False)

                instruction_len = len(tokenizer_multimodal_token(instruction, tokenizer, modal_token, return_tensors='pt'))
                conversation_len = len(tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt'))

                if process_guided:
                    if idx == len(source) - 1:
                        targets[-1][cur:instruction_len] = IGNORE_INDEX
                    else:
                        targets[-1][cur:conversation_len] = IGNORE_INDEX
                else:
                    targets[-1][cur:instruction_len] = IGNORE_INDEX

                cur = conversation_len
                message += tmp_message

    return dict(input_ids=input_ids, labels=targets)


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments,
    modal_token: str = None,
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    assert modal_token in MODAL_INDEX_MAP, f"Unsupported modal token {modal_token}."

    for source in sources:
        for sentence in source:
            if modal_token in sentence['value']:
                sentence['value'] = sentence['value'].replace(modal_token, '').strip()
                sentence['value'] = modal_token + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
            replace_token = modal_token
            # TODO: fix this for multimedia, e.g., <video>, <audio>, etc.
            sentence["value"] = sentence["value"].replace(modal_token, replace_token)

    return sources


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: List[str],
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = []
        if len(data_path) == 1 and data_path[0].endswith(".yaml"):
            with open(data_path[0], "r") as f:
                yaml_data = yaml.safe_load(f)
            datasets = yaml_data.get("datasets")
            # file should be in the format of:
            # datasets:
            #   - json_path: xxxx1.json
            #     data_root: /path/data/root
            #     sampling_strategy: all
            #   - json_path: xxxx2.json
            #     data_root: /path/data/root
            #     sampling_strategy: end:3000
            #   - json_path: xxxx3.json
            #     data_root: /path/data/root
            #     sampling_strategy: random:50%
            for dataset in datasets:
                json_path = dataset.get("json_path")
                data_root = dataset.get("data_root", None)
                sampling_strategy = dataset.get("sampling_strategy", "all")
                sampling_number = None

                with open(json_path, "r") as json_file:
                    if json_path.endswith(".json"):
                        cur_data_dict = json.load(json_file)
                    elif json_path.endswith(".jsonl"):
                        cur_data_dict = [json.loads(l.strip('\n')) for l in json_file.readlines()]
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")
                
                # replace data path with data_root
                if data_root is not None:
                    for d in cur_data_dict:
                        if 'image' in d:
                            d['image'] = os.path.join(data_root, d['image'])
                        elif 'video' in d:
                            d['video'] = os.path.join(data_root, d['video'])

                # Apply the sampling strategy
                if ":" in sampling_strategy:
                    sampling_strategy, sampling_number = sampling_strategy.split(":")
                    if "%" in sampling_number:
                        sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                    else:
                        sampling_number = int(sampling_number)
                if sampling_strategy == "first" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[:sampling_number]
                elif sampling_strategy == "end" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[-sampling_number:]
                elif sampling_strategy == "random" and sampling_number is not None:
                    random.shuffle(cur_data_dict)
                    cur_data_dict = cur_data_dict[:sampling_number]
                elif sampling_strategy == "all":
                    pass
                else: 
                    raise ValueError(f"Unsupported sampling strategy: {sampling_strategy}")

                rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path} with sampling strategy [{sampling_strategy}]")
                list_data_dict.extend(cur_data_dict)
        
        else:
            for dp in data_path:
                with open(dp, "r") as f:
                    if dp.endswith(".json"):
                        _datas = json.load(f)
                    elif dp.endswith(".jsonl"):
                        _datas = [json.loads(l.strip('\n')) for l in f.readlines()]
                    else:
                        raise ValueError(f"Unsupported file type: {dp}")
                rank0_print(f"Loaded {len(_datas)} samples from {dp}")
                list_data_dict.extend(_datas)

        rank0_print(f"All loaded samples: {len(list_data_dict)}")
        if data_args.use_guide not in [None, "off"]:
            new_list_data_dict = []


            for sample in tqdm(list_data_dict, desc="Converting guide format"):
                if "image" not in sample and "video" not in sample:
                    new_list_data_dict.append(sample)
                    continue

                conversations = sample["conversations"]
                if len(conversations) % 2 != 0:
                    rank0_print("wrong conversations length")
                    continue

                if "image" in sample and "<image>" not in conversations[0]['value']:
                    conversations[0]['value'] = "<image>\n" + conversations[0]['value']
                if "video" in sample and "<video>" not in conversations[0]['value']:
                    if "<image>" in conversations[0]['value']:
                        conversations[0]['value'] = conversations[0]['value'].replace("<image>", "<video>")
                    else:
                        conversations[0]['value'] = "<video>\n" + conversations[0]['value']
                
                for i, conversation in enumerate(conversations):
                    if i % 2 == 0:
                        continue
                    assert conversations[i-1]['from'] == 'human'
                    assert conversations[i]['from'] == 'gpt'
                    new_conversations = conversations[:i+1]
                    new_sample = sample.copy()
                    new_sample['conversations'] = new_conversations
                    new_list_data_dict.append(new_sample)
            
            del list_data_dict
            list_data_dict = new_list_data_dict
            rank0_print(f"Guided format samples: {len(list_data_dict)}")

        # rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 576 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        image_processor = self.data_args.image_processor
        video_processor = self.data_args.video_processor

        num_frames = NUM_FRAMES if self.data_args.num_frames is None else self.data_args.num_frames

        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.data_folder
            if isinstance(image_file, list):
                image_file = [os.path.join(image_folder, f) for f in image_file]
            else:
                image_file = [os.path.join(image_folder, image_file)]
            try:
                image, image_size = process_image(
                    image_file, image_processor,
                    aspect_ratio=self.data_args.image_aspect_ratio,
                    image_grid_pinpoints=self.data_args.image_grid_pinpoints,
                    image_crop_resolution=self.data_args.image_crop_resolution,
                    image_split_resolution=self.data_args.image_split_resolution
                )
            except:
                traceback.print_exc()
                backup_idx = random.randint(0, len(self.list_data_dict) - 1)
                print(f"Encounted error when reading image {image_file}, use {backup_idx}-th example instead!!!")
                return self.__getitem__(backup_idx)

            single_guided_prompt = extract_guided_prompt(sources[0]['conversations'][-2]['value']) if self.data_args.use_guide not in [None, "off"] else None
            guided_prompt = [single_guided_prompt] * len(image_file)
            # place <image> tag to question head.
            modal_token = "<image>"
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args, modal_token)
        elif 'video' in sources[0]:
            video_file = self.list_data_dict[i]['video']
            video_folder = self.data_args.data_folder
            video_file = os.path.join(video_folder, video_file)

            try:
                video = process_video(video_file, video_processor, aspect_ratio=self.data_args.image_aspect_ratio, num_frames=num_frames)
            except Exception as e:
                traceback.print_exc()
                backup_idx = random.randint(0, len(self.list_data_dict) - 1)
                print(f"Encounted error when reading video {video_file}, use {backup_idx}-th example instead!!!")
                return self.__getitem__(backup_idx)

            guided_prompt = [extract_guided_prompt(sources[0]['conversations'][-2]['value']) if self.data_args.use_guide not in [None, "off"] else None]
            # place <video> tag to question head.
            modal_token = "<video>"
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args, modal_token)
        else:
            guided_prompt = ['']
            modal_token = None
            sources = copy.deepcopy([e["conversations"] for e in sources])

        if self.data_args.is_pretraining:
            data_dict = preprocess_plain(sources, self.tokenizer, modal_token=modal_token)
        else:
            process_guided = (self.data_args.use_guide not in [None, "off"]) and (modal_token in ["<image>", "<video>"])
            data_dict = preprocess(sources, self.tokenizer, modal_token=modal_token, process_guided=process_guided)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['image_size'] = image_size
        elif 'video' in self.list_data_dict[i]:
            data_dict['video'] = video
            data_dict['image_size'] = [None]
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            data_dict['image'] = torch.zeros(1, 3, self.data_args.image_size, self.data_args.image_size)
            data_dict['image_size'] = [None]
        if self.data_args.use_guide not in [None, "off"]:
            data_dict['guided_prompt'] = guided_prompt
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    data_args: DataArguments

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # work for 'images' argument in `prepare_inputs_labels_for_multimodal` of LlavaMetaForCausalLM in llava_arch.py
        batch['images'] = []
        for instance in instances:
            for modal_token in MODAL_INDEX_MAP.keys():
                modal_token = modal_token.lower()
                # MODAL_TOKEN shape like: <image>, <video>, ...
                modal_name = re.findall(f'[<](.*)[>]', modal_token)
                assert len(modal_name) == 1
                modal_name = modal_name[0]
                if modal_name in instance:
                    if len(instance['image_size']) > 1:
                        assert len(instance['image_size']) == len(instance[modal_name])
                        for i, img in enumerate(instance[modal_name]):
                            batch['images'].append((img.unsqueeze(0), instance['image_size'][i], modal_name))
                    else:
                        batch['images'].append((instance[modal_name], instance['image_size'][0], modal_name))
        
        if self.data_args.use_guide not in [None, "off"]:
            guided_prompt = []
            for instance in instances:
                guided_prompt += instance['guided_prompt']
            batch["guided_input"] = self.data_args.guide_tokenizer(
                guided_prompt, return_tensors='pt', padding="max_length", truncation=True
            )

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank
    # set_seed(42)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            # device_map={"": training_args.device},
            # BUG: High version transformers report error: 
            # ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time
            # load_in_4bit=training_args.bits == 4,
            # load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type, # {'fp4', 'nf4'}
                bnb_4bit_quant_storage=compute_dtype,
            )
        ))

    config = VLLMConfigs[model_args.model_type].from_pretrained(model_args.model_path, trust_remote_code=True)
    config._attn_implementation = attn_implementation

    # if model_args.vision_tower is not None:
    model = VLLMs[model_args.model_type].from_pretrained(
        model_args.model_path,
        config=config,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        do_sample=True,
        **bnb_model_from_pretrained_args
    )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    if model_args.vision_tower is not None or getattr(model.config, "mm_vision_tower", None) is not None:
        rank0_print("Post Initializing vision modules...")
        # initialize vision encoder + multi-modal projector
        model_args.image_aspect_ratio = data_args.image_aspect_ratio
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_size = vision_tower.image_size

        data_args.image_processor = vision_tower.image_processor
        data_args.video_processor = vision_tower.video_processor if hasattr(vision_tower, "video_processor") else vision_tower.image_processor

        data_args.use_guide = model_args.use_guide
        if data_args.use_guide not in [None, "off"]:
            data_args.guide_tokenizer = vision_tower.guide_tokenizer

        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_grid_pinpoints is not None:
            if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
                try:
                    patch_size = data_args.image_processor.size[0]
                except Exception as e:
                    patch_size = data_args.image_processor.size["shortest_edge"]
                
                assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
                # Use regex to extract the range from the input string
                matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
                range_start = tuple(map(int, matches[0]))
                range_end = tuple(map(int, matches[-1]))
                # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
                grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
                # Multiply all elements by patch_size
                data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
            elif isinstance(data_args.image_grid_pinpoints, str):
                data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)

        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.image_crop_resolution = data_args.image_crop_resolution
        model.config.image_split_resolution = data_args.image_split_resolution
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.mm_newline_position = model_args.mm_newline_position
        model.config.add_time_instruction = data_args.add_time_instruction

        # model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        # if training_args.freeze_mm_mlp_adapter:
        #     for p in model.get_model().mm_projector.parameters():
        #         p.requires_grad = False


        model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
        # Set the entire model to not require gradients by default
        model.get_model().requires_grad_(False)
        model.get_vision_tower().requires_grad_(False)
        model.get_model().mm_projector.requires_grad_(False)
        # Parse the mm_tunable_parts to decide which parts to unfreeze
        tunable_parts = model_args.mm_tunable_parts.split(",")
        if "mm_projector" in tunable_parts:
            for name, param in model.get_model().mm_projector.named_parameters():
                if "logit_scale" not in name and "logit_bias" not in name:
                    param.requires_grad_(True)
        
        if "pure_vision_model" in tunable_parts:
            for name, param in model.named_parameters():
                if "vision_tower.vision_model" in name and "head" not in name:
                    param.requires_grad_(True)
        
        if data_args.use_guide not in [None, "off"]:
            if "vision_model_head" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower.vision_model" in name and "head" in name:
                        param.requires_grad_(True)

            if "guide_encoder" in tunable_parts:
                for name, param in model.named_parameters():
                    if "guide_encoder" in name:
                        param.requires_grad_(True)

            if "attn_scale" in tunable_parts:
                for name, param in model.get_model().mm_projector.named_parameters():
                    if "logit_scale" in name or "logit_bias" in name:
                        param.requires_grad_(True)
        
        if "language_model" in tunable_parts:
            for name, param in model.named_parameters():
                if "vision_tower" not in name and "mm_projector" not in name:
                    param.requires_grad_(True)

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.vision_tower_lr = training_args.vision_tower_lr
        model.config.guide_injector_lr = training_args.guide_injector_lr
        model.config.num_frames = NUM_FRAMES if data_args.num_frames is None else data_args.num_frames

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    rank0_print("Current model:", model)
    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Total parameters: ~{total_params/1e6:.2f} M)")
    rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} M)")
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # select a Trainer
    trainer = HIComTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if check_ckpt_exists(training_args.output_dir):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    # train("flash_attention_2")
    train("sdpa")
