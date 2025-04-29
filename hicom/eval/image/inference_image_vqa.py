import os
import re
import math
import json
import argparse
import warnings
import shortuuid
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('./')
from hicom import model_init, mm_infer
from hicom.utils import disable_torch_init
from hicom.mm_utils import get_model_name_from_path

# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class ImageDataset(Dataset):

    def __init__(self, data_list, processor):
        self.data_list = data_list
        self.processor = processor

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        image_name = line["image"]
        question = line["text"]
        question_id = line["question_id"]
        image_path = os.path.join(args.image_folder, image_name)

        image_tensor, image_size = self.processor(image_path)

        return {
            'image': image_tensor,
            'image_name': image_name,
            'image_size': image_size[0],
            'question': question,
            'question_id': question_id,
        }


def collate_fn(batch):
    image = [x['image'] for x in batch]
    img_id = [x['image_name'] for x in batch]
    img_size = [x['image_size'] for x in batch]
    qus = [x['question'] for x in batch]
    qid = [x['question_id'] for x in batch]
    image = torch.stack(image, dim=0)
    return image, img_id, img_size, qus, qid


def run_inference(args):
    disable_torch_init()

    # Initialize the model
    if args.dtype == 'float16':
        dtype = torch.float16
    elif args.dtype == 'bfloat16':
        dtype = torch.bfloat16

    model, processor, tokenizer = model_init(args.model_path, torch_dtype=dtype, attn_implementation=args.attn_implementation)
    model_name = get_model_name_from_path(args.model_path)

    with open(args.question_file, "r") as f:
        if args.question_file.endswith(".json"):
            questions = json.load(f)
        elif args.question_file.endswith(".jsonl"):
            questions = [json.loads(l.strip('\n')) for l in f.readlines()]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    assert args.batch_size == 1, "Batch size must be 1 for inference"
    dataset = ImageDataset(questions, processor['image'])
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    # Iterate over each sample in the ground truth file
    for i, (image_tensors, image_names, image_sizes, questions, question_ids) in enumerate(tqdm(dataloader)):

        # reduce batch dimension
        image_tensor = image_tensors[0]
        image_name = image_names[0]
        image_size = image_sizes[0]
        question = questions[0]
        question_id = question_ids[0]

        guide_instruct = get_guide_instruct(question, args.benchmark)

        output = mm_infer(
            image_tensor,
            question,
            model=model,
            tokenizer=tokenizer,
            modal='image',
            image_size=image_size,
            do_sample=False,
            dtype=dtype,
            guide_instruct=guide_instruct,
        )
        ans_id = shortuuid.uuid()
        qa = {"question_id": question_id, "prompt": question, "text": output, "answer_id": ans_id, "model_id": model_name, "metadata": {}}

        ans_file.write(json.dumps(qa) + "\n")

    ans_file.close()


def get_guide_instruct(question, benchmark):
    if benchmark in ["gqa", "MME", "pope", "vqav2"]:
        return question.replace("\nAnswer the question using a single word or phrase.", "")
    elif benchmark in ["scienceqa", "seed_bench"]:
        return question.split("\nA. ")[0]
    elif benchmark in ["textvqa"]:
        return question.split("\nReference OCR token:")[0]
    elif benchmark in ["vizwiz"]:
        return question.replace("\nWhen the provided information is insufficient, respond with 'Unanswerable'.\nAnswer the question using a single word or phrase.", "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', help='', required=True)
    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--image-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--dtype", type=str, required=False, default='float16')
    parser.add_argument("--attn_implementation", type=str, required=False, default=None)
    parser.add_argument("--batch-size", type=int, required=False, default=1)
    parser.add_argument("--num-workers", type=int, required=False, default=8)
    args = parser.parse_args()

    run_inference(args)
