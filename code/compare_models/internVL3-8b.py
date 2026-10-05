#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import torch
import math
import numpy as np
import torch
import torchvision.transforms as T

from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoTokenizer, AutoModel

import os

# 使用transformers pipeline
from transformers import pipeline

# ==== Configuration ====
MODEL_DIR = Path("xxx/path")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# class Qwen25PipelineWrapper:
#     """封装Qwen2.5-VL模型"""
#
#     def __init__(self):
#         print("[INFO] Loading Qwen2.5-VL-7B-Instruct pipeline ...")
#         self.pipe = pipeline(
#             task="image-text-to-text",
#             model=str(MODEL_DIR),
#             torch_dtype="auto",
#             device_map={"": 7},
#         )
#
#     def infer(self, image_path: Path, prompt: str, max_new_tokens: int = 512) -> str:
#         """推理单张图片"""
#         try:
#             # 读取图像
#             image = Image.open(image_path).convert("RGB")
#
#             # 构建消息
#             messages = [
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "image"},
#                         {"type": "text", "text": prompt},
#                     ],
#                 }
#             ]
#
#             # 调用模型
#             outputs = self.pipe(
#                 text=messages,
#                 images=[image],
#                 max_new_tokens=max_new_tokens,
#                 return_full_text=False,  # 只返回生成的文本
#                 do_sample=False,  # 对于答案提取，最好不用采样
#             )
#
#             return outputs[0]["generated_text"].strip()
#
#         except Exception as e:
#             print(f"[ERROR] Inference failed for {image_path}: {e}")
#             return f"Error: {str(e)}"


def extract_answer_from_response(response: str) -> str:
    """从响应中提取答案"""
    response = response.strip()

    # 如果是单个字母
    if len(response) == 1 and response.upper() in "ABCDEFGH":
        return response.upper()

    # 从中文回答中提取字母
    chinese_pattern = r'[选答是]\s*[：:]?\s*([A-H])'
    match = re.search(chinese_pattern, response)
    if match:
        return match.group(1).upper()

    # 从英文回答中提取字母
    english_pattern = r'[Aa]nswer[:\s]+([A-H])'
    match = re.search(english_pattern, response)
    if match:
        return match.group(1).upper()

    # 直接匹配字母
    match = re.search(r'\b([A-H])\b', response)
    if match:
        return match.group(1).upper()

    # 如果没有找到字母，返回原响应
    return response


def robust_parse_response(response):
    # 处理字符串输入
    if isinstance(response, str):
        # 移除方括号
        response = re.sub(r'[\[\]]', '', response)
        # 按逗号分割并清理
        parts = [p.strip().strip("'\"") for p in response.split(',') if p.strip()]
        response = parts

    result = []
    for part in response:
        # 提取所有大写字母
        letters = re.findall(r'[A-Z]', part)
        result.extend(letters)

    # 确保有4个答案
    while len(result) < 4:
        result.append('unknown')

    return result[:4]

# ==== Entry point ====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--INPUT_PATH', default='xxx/path',required=False)
    parser.add_argument('--OUTPUT_PATH', default='xxx/path',required=False)
    parser.add_argument('--type', default='single', required=False, choices=['single', 'joint'])  # joint_qa Level4 visual_evidence_joint_qa
    args = parser.parse_args()
    # 确保输出目录存在
    output_path = Path(args.OUTPUT_PATH)
    output_dir = output_path.parent

    # 创建目录（如果不存在）
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Created output directory: {output_dir}")
    except Exception as e:
        print(f"[WARNING] Failed to create directory {output_dir}: {e}")

    # 确保输出文件可以被创建
    if not output_path.exists():
        print(f"[INFO] Creating new output file: {output_path}")
    else:
        print


    # 初始化模型
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=False)
    # set the max number of tiles in `max_num`




    if args.type == 'single':
        # 读取数据
        with open(args.INPUT_PATH, 'r', encoding='utf-8') as f:
            lines = [l for l in f if l.strip()]

        # 处理数据
        with open(args.OUTPUT_PATH, 'w', encoding='utf-8') as fout:
            for line in tqdm(lines, desc=f"Processing data"):
                rec = json.loads(line)
                img_path_old = rec.get('image_paths')
                # 如果路径包含指定的前缀并且该路径存在，则进行替换
                if 'validation_data' in img_path_old:
                    img_path = img_path_old.replace(
                        'xxx/path',
                        'xxx/path')

                if 'DrVD-Bench' in img_path_old:
                    img_path = img_path_old.replace('xxx/path',
                                                                'xxx/path')


                if 'question_type' in rec and rec.get('question_type') == 'open_question':
                    '''open question'''
                    # report or choice
                    if "report" in str(rec.get("task", "")).lower():
                        max_tokens = 500
                        prompt_text = (rec['question'] + '\n\nWrite the image report as a single coherent paragraph, no more than 500 words.')

                    else:
                        max_tokens = 50
                        prompt_text = (rec['question'] + '\n\nWrite the correct answer, no more than 10 words.')

                    pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
                    generation_config = dict(max_new_tokens=max_tokens, do_sample=True)
                    # 调用模型
                    response = model.chat(
                        tokenizer,
                        pixel_values,
                        prompt_text,
                        generation_config
                    )

                    # 提取答案
                    answer_text = response

                else:
                    '''单项选择题'''
                    prompt_text = ("Question: " + rec['question'] + "\nOptions:\n" + "\n".join(
                        rec['options']) + "\n\nProvide only the letter as your answer.")
                    max_tokens = 50

                    pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
                    generation_config = dict(max_new_tokens=max_tokens, do_sample=True)
                    # 调用模型
                    response = model.chat(
                        tokenizer,
                        pixel_values,
                        prompt_text,
                        generation_config
                    )

                    answer_text = extract_answer_from_response(response)

                # 构建输出
                output_rec = {
                    "prompt": prompt_text,
                    "text": answer_text,
                    "GT_ANSWER": rec.get('answer', ''),
                    "model_id": "InternVL3_5-8B",
                    "metadata": rec
                }

                fout.write(json.dumps(output_rec, ensure_ascii=False) + '\n')

        print(f"✅ Output written to {args.OUTPUT_PATH}")

    else:
        '''joint mode'''
        items = []
        with open(args.INPUT_PATH, 'r', encoding='utf-8') as f:
            for l in f:
                if l.strip(): items.append(json.loads(l))

        results = []
        for rec in tqdm(items, desc="Processing joint"):
            img_path_old = rec.get('image_paths')
            # 如果路径包含指定的前缀并且该路径存在，则进行替换
            if 'validation_data' in img_path_old:
                img_path = img_path_old.replace(
                    'xxx/path',
                    'xxx/path')

            if 'DrVD-Bench' in img_path_old:
                img_path = img_path_old.replace('xxx/path',
                                                'xxx/path')
            prompt = rec.get('joint_prompt', '')

            # 调用模型
            pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
            generation_config = dict(max_new_tokens=100, do_sample=True)
            # 调用模型
            response = model.chat(
                tokenizer,
                pixel_values,
                prompt,
                generation_config
            )

            parts = response.split(',')
            if len(parts) == 4:
                response1, response2, response3, response4 = parts[:4]
            else:
                # 处理缺少部分的情况
                response1, response2, response3, response4 = robust_parse_response(response)



            # 按照您提供的顺序分配给对应的answer变量
            model_modality_answer = response1  # 第一个选项对应modality_answer
            model_organ_answer = response2  # 第二个选项对应organ_answer
            model_lesion_answer = response3  # 第三个选项对应lesion_answer
            model_diagnosis_answer = response4  # 第四个选项对应diagnosis_answer

            # 构建输出
            output_rec = {
                "prompt": prompt,
                "text": response,
                "model_modality_answer": model_modality_answer,
                "model_organ_answer": model_organ_answer,
                "model_lesion_answer": model_lesion_answer,
                "model_diagnosis_answer": model_diagnosis_answer,
                "GT_modality_answer": rec.get('modality_answer', ''),
                "GT_organ_answer": rec.get('organ_answer', ''),
                "GT_lesion_answer": rec.get('lesion_answer', ''),
                "GT_diagnosis_answer": rec.get('diagnosis_answer', ''),
                "model_id": "InternVL3_5-8B",
                "metadata": rec
            }
            results.append(output_rec)

        with open(args.OUTPUT_PATH, 'w', encoding='utf-8') as fout:
            for rec in results:
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')

        print(f"✅ Output written to {args.OUTPUT_PATH}")


if __name__ == '__main__':
    main()
