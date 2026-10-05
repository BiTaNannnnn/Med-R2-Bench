#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm
import os
from HuatuoGPT_Vision.cli import HuatuoChatbot

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==== Configuration ====
huatuogpt_vision_model_path = "xxx/path"


def extract_answer_from_response(response: list) -> str:
    """从响应中提取答案"""
    # 确保是字符串
    response = response[0].strip() if isinstance(response, list) else response.strip()

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
    """解析 response，以确保它有 4 个答案"""
    if isinstance(response, list):
        response = response[0]  # 只取列表的第一个元素

    response = re.sub(r'[\[\]]', '', response)  # 移除方括号
    parts = [p.strip().strip("'\"") for p in response.split(',') if p.strip()]

    result = []
    for part in parts:
        # 提取所有大写字母
        letters = re.findall(r'[A-Z]', part)
        result.extend(letters)

    # 确保有 4 个答案
    while len(result) < 4:
        result.append('unknown')

    return result[:4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--INPUT_PATH', default='xxx/path',
                        required=False)
    parser.add_argument('--OUTPUT_PATH', default='xxx/path',
                        required=False)
    parser.add_argument('--type', default='joint', required=False, choices=['single', 'joint']) # joint_qa  Level3
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

    # 初始化模型
    model = HuatuoChatbot(huatuogpt_vision_model_path)

    if args.type == 'single':
        # 读取数据
        with open(args.INPUT_PATH, 'r', encoding='utf-8') as f:
            lines = [l for l in f if l.strip()]

        # 处理数据
        with open(args.OUTPUT_PATH, 'w', encoding='utf-8') as fout:
            for line in tqdm(lines, desc=f"Processing data"):
                rec = json.loads(line)
                img_path_old = rec.get('image_paths', None)  # 获取图像路径

                # 如果路径包含指定的前缀并且该路径存在，则进行替换
                if 'validation_data' in img_path_old:
                    img_path = img_path_old.replace(
                        'xxx/path',
                        'xxx/path')

                if 'DrVD-Bench' in img_path_old:
                    img_path = img_path_old.replace('xxx/path',
                                                    'xxx/path')

                if img_path:
                    img_path = [img_path] if isinstance(img_path, str) else img_path  # 确保是列表

                if 'question_type' in rec and rec.get('question_type') == 'open_question':
                    # open question
                    if "report" in str(rec.get("task", "")).lower():
                        max_tokens = 500
                        prompt_text = (rec[
                                           'question'] + '\n\nWrite the image report as a single coherent paragraph, no more than 500 words.')
                    else:
                        max_tokens = 50
                        prompt_text = (rec['question'] + '\n\nWrite the correct answer, no more than 10 words.')

                    # 调用模型
                    response = model.inference(prompt_text, img_path)  # 使用列表作为输入
                    answer_text = response[0] if isinstance(response, list) else response

                else:
                    # 单项选择题
                    prompt_text = ("Question: " + rec['question'] + "\nOptions:\n" + "\n".join(
                        rec['options']) + "\n\nProvide only the letter as your answer.")
                    max_tokens = 50


                    response = model.inference(prompt_text, img_path)

                    answer_text = extract_answer_from_response(response)

                # 构建输出
                output_rec = {
                    "prompt": prompt_text,
                    "text": answer_text,
                    "GT_ANSWER": rec.get('answer', ''),
                    "model_id": "huatuogpt-vision-7b",
                    "metadata": rec
                }

                fout.write(json.dumps(output_rec, ensure_ascii=False) + '\n')

        print(f"✅ Output written to {args.OUTPUT_PATH}")

    else:
        # joint mode
        items = []
        with open(args.INPUT_PATH, 'r', encoding='utf-8') as f:
            for l in f:
                if l.strip():
                    items.append(json.loads(l))

        results = []
        for rec in tqdm(items, desc="Processing joint"):
            img_path_old = rec.get('image_paths', None)
            prompt = rec.get('joint_prompt', '')

            # 如果路径包含指定的前缀并且该路径存在，则进行替换
            if 'validation_data' in img_path_old:
                img_path = img_path_old.replace(
                    'xxx/path',
                    'xxx/path')

            if 'DrVD-Bench' in img_path_old:
                img_path = img_path_old.replace('xxx/path',
                                                'xxx/path')



            if img_path:
                img_path = [img_path] if isinstance(img_path, str) else img_path  # 确保是列表

            # 调用模型
            response = model.inference(prompt, img_path)
            response_text = response[0] if isinstance(response, list) else response

            parts = response_text.split(',')
            if len(parts) == 4:
                response1, response2, response3, response4 = parts[:4]
            else:
                # 处理缺少部分的情况
                response1, response2, response3, response4 = robust_parse_response(response_text)

            output_rec = {
                "prompt": prompt,
                "text": response_text,
                "model_modality_answer": response1,
                "model_organ_answer": response2,
                "model_lesion_answer": response3,
                "model_diagnosis_answer": response4,
                "GT_modality_answer": rec.get('modality_answer', ''),
                "GT_organ_answer": rec.get('organ_answer', ''),
                "GT_lesion_answer": rec.get('lesion_answer', ''),
                "GT_diagnosis_answer": rec.get('diagnosis_answer', ''),
                "model_id": "huatuogpt-vision-7b",
                "metadata": rec
            }
            results.append(output_rec)

        with open(args.OUTPUT_PATH, 'w', encoding='utf-8') as fout:
            for rec in results:
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')

        print(f"✅ Output written to {args.OUTPUT_PATH}")


if __name__ == '__main__':
    main()



# query = 'What does the picture show?'
# image_paths = ['xxx/path']
# huatuogpt_vision_model_path = "xxx/path"
# from HuatuoGPT_Vision.cli import HuatuoChatbot
# bot = HuatuoChatbot(huatuogpt_vision_model_path) # loads the model
# # output = bot.inference(query, image_paths) # generates
# output = bot.inference(query, image_paths)
# print(output)  # Prints the model output