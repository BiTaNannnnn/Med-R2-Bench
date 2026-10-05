import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os.path as osp
import random
import traceback
from typing import Any, Dict, List, Union
import glob  # 导入 glob

import datetime
import numpy as np
import torch.distributed as dist
from prettytable import PrettyTable
from torch.utils.data.distributed import DistributedSampler
from peft import PeftModel, PeftConfig, get_peft_model

sys.path.append("")
from videollama3 import disable_torch_init, model_init, mm_infer
from videollama3.model import load_pretrained_model
from videollama3.mm_utils import load_images, process_images, process_video, tokenizer_multimodal_token, \
    get_model_name_from_path, KeywordsStoppingCriteria
from videollama3.model.processor import Videollama3Processor
from evaluation.benchmarks import build_dataset
from evaluation.register import INFERENCES
from evaluation.utils import CUDADataLoader

import re
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from difflib import SequenceMatcher

from PIL import Image
import math
import random

CAPTION_TEMPLATES = [
    "Please generate a complete radiology report including both findings and impression for this 3D medical image.",
    "Could you produce a structured clinical report summarizing the findings and diagnostic impression of this 3D scan?",
    "Provide a detailed report that describes the imaging findings and states the diagnostic impression.",
    "Please compile a medical report that includes the key findings and your overall impression.",
    "Generate a full diagnostic report covering both the observed findings and clinical interpretation.",
    "Summarize this 3D image in a report containing a findings section and an impression section.",
    "Please prepare a radiology-style report describing the findings followed by an impression.",
    "Create a comprehensive report summarizing the results and diagnostic conclusion of this 3D image.",
    "Could you draft a report including both the observed findings and the diagnostic assessment?",
    "Provide a complete clinical report integrating findings and impression based on this 3D medical image."
]


# =========================================================================================
# 新增的函数：从文件夹加载3D切片
# =========================================================================================
def _natural_key(path: str):
    """用于文件名的自然排序：把数字块当作数字比较，避免 1,10,2 的错序。"""
    name = os.path.basename(path)
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', name)]


def _dicom_sort_key(ds, path):
    """生成排序键：优先 InstanceNumber，其次 ImagePositionPatient[Z]，最后文件名自然序。"""
    inst = getattr(ds, "InstanceNumber", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if inst is not None:
        try:
            return (0, int(inst))
        except Exception:
            pass
    if ipp and len(ipp) >= 3:
        try:
            return (1, float(ipp[2]))
        except Exception:
            pass
    return (2, _natural_key(path))


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
        height: int, width: int, factor: int = 28, min_pixels: int = 4 * 28 * 28, max_pixels: int = 16384 * 28 * 28
) -> tuple[int, int]:
    MAX_RATIO = 200
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


IMAGE_FACTOR = 28
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28


def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    # NOTE: 确保这里的 device_map 设置正确，例如 'cuda:0' 或 'auto'
    # 1. 加载基础模型

    tokenizer, base_model, image_processor, context_len = load_pretrained_model(
        args.model_path, None, model_name, device_map='cuda:0'
    )

    # 2. 加载 LoRA 微调配置
    if args.lora_path:
        peft_config = PeftConfig.from_pretrained(args.lora_path)
        model = PeftModel.from_pretrained(base_model, args.lora_path, device_map='cuda:0')
        print("Load LoRA Model")
    else:
        model = base_model

    model.config.use_token_compression = True
    processor = Videollama3Processor(image_processor, tokenizer)

    if args.return_gating_logit is not None:
        from moellava.utils import get_gating_logit_by_hook
        print(model)
        fea_hooks = get_gating_logit_by_hook(model)
        all_gating_logits = {}

    # 读取 jsonl
    questions = []
    with open(os.path.expanduser(args.question_file), "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))

    # 原有逻辑保持不变
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    cnt = -1
    for i, line in enumerate(tqdm(questions)):
        # line = questions[890]
        cnt += 1
        if 1:
            question = line["question"]
            try:
                options = line["options"]
            except:
                options = ""
            if isinstance(options, list):
                options_str = "\n".join(options)
            else:
                options_str = str(options)

            question = line["question"] + "\n" + options_str
            gt_ans = line["answer"]
            meta_data = line

        qs = question
        # qs = random.choice(CAPTION_TEMPLATES)
        cur_prompt = qs

        # 如果文件夹为空或无法加载图像，则跳过此数据点

        qs = question

        qs = qs.replace('<video>\n', '').strip()
        qs = qs.replace('<image>\n', '').strip()
        qs = qs.replace('<video>', '').strip()
        qs = qs.replace('<image>', '').strip()
        cur_prompt = qs
        if "clinical report" in qs:
            qs = qs
        else:
            qs = qs + "\nAnswer the question using a single word or phrase."
        # print(qs)
        # qs= qs + "\nPlease answer the question concisely."
        # image_file = line["image_paths"].replace("xxx/path", "xxx/path")
        if "valid" in line["image_paths"]:
            if "noise_recognition" in line["image_paths"]:
                image_file = "xxx/path" + '/' + line["image_paths"].split('/')[-2] + '/' + \
                             line["image_paths"].split('/')[-1]
            elif "artifact_recognition" in line["image_paths"]:
                image_file = "xxx/path" + '/' + line["image_paths"].split('/')[-2] + '/' + \
                             line["image_paths"].split('/')[-1]
            else:
                image_file = "xxx/path" + '/' + line["image_paths"].split('/')[-3] + '/' + \
                             line["image_paths"].split('/')[-2] + '/' + line["image_paths"].split('/')[-1]
        else:
            image_file = line["image_paths"].replace("xxx/path",
                                                     "xxx/path")
        if isinstance(image_file, str):  # 检查是否为字符串
            # image = load_images([os.path.join(args.image_folder, image_file)])
            # print(image_file)
            image = load_images(image_file)
        else:
            image = load_images(image_file[0])
        len_img = len(image)
        # print(len_img)
        # print(qs)
        image_entries = [{"type": "image"} for _ in range(len_img)]
        conversation = [{
            "role": "user",
            "content": image_entries + [
                {
                    "type": "text",
                    "text": qs,
                }
            ]
        }]
        modal = 'image'
        inputs = processor(
            images=[image] if modal != "text" else None,
            text=conversation,
            merge_size=2 if modal == "video" else 1,
            return_tensors="pt"
        )
        inputs = {k: v.cuda().to('cuda') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        vision_calls = []

        def _make_hook(name):
            def hook(module, inp, out):
                vision_calls.append(name)

            return hook

        hooks = []
        for n, m in model.named_modules():
            low = n.lower()
            if any(x in low for x in ["vision", "visual", "image"]):
                try:
                    h = m.register_forward_hook(_make_hook(n))
                    hooks.append(h)
                except Exception:
                    pass

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                modals=[modal],
                temperature=0,
                max_new_tokens=4096,
                top_p=0.95,
                repetition_penalty=1.0,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        print(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "prompt": cur_prompt,
            "text": outputs,
            "GT_ANSSWER": gt_ans,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": meta_data}) + "\n")
        ans_file.flush()

    ans_file.close()
    if args.return_gating_logit is not None:
        torch.save(all_gating_logits, f'{args.return_gating_logit}.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")  # 这个参数在脚本中似乎未使用
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="")
    parser.add_argument('--type', default='single', required=False, choices=['single', 'joint'])

    parser.add_argument("--conv-mode", type=str, default="phi")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--return_gating_logit", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA 微调模型路径")
    args = parser.parse_args()

    eval_model(args)