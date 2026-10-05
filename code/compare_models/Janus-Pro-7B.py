#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

import torch


from transformers import AutoModelForCausalLM
from Janus.janus.models.modeling_vlm import MultiModalityCausalLM
from Janus.janus.models.processing_vlm import VLChatProcessor
from Janus.janus.utils.io import load_pil_images
import os



# ==== Configuration ====
MODEL_DIR = Path("xxx/path")

class JanusPipelineWrapper:
    def __init__(self):
        print(f"[INFO] Loading Janus model from {MODEL_DIR} ...")
        # self.pipe = pipeline(
        #     task="image-text-to-text",
        #     model=str(MODEL_DIR),
        #     torch_dtype="auto",
        #     device_map="cuda:0",
        #     trust_remote_code=True,
        # )
        # 额外加载 processor 用于模板与预处理
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(MODEL_DIR)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True)
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

        print("[INFO] Lingshu pipeline & processor loaded.")

    def __call__(self, img_path: Path, question: str) -> str:
        # 1. 构造对话消息（带图像占位）
        messages = [
            {
                "role": "<|User|>",
                "content":f"<image_placeholder>\n{question}",
                "images": [img_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        # load images and prepare for inputs
        pil_images = load_pil_images(messages)
        
        # 3. 预处理：把文本+图像转成模型输入
        prepare_inputs = self.vl_chat_processor(conversations=messages, images=pil_images, force_batchify=True).to(self.vl_gpt.device)

        # 4. 生成
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        # # run the model to get the response
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        # 5. 提取助手回复部分
        return answer










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
    parser.add_argument('--type', default='joint', required=False, choices=['single', 'joint'])  # joint_qa Level4 visual_evidence_joint_qa
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
    model = JanusPipelineWrapper()

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

                    # 调用模型
                    response = model(
                        img_path,
                        prompt_text,
                    )

                    # 提取答案
                    answer_text = response

                else:
                    '''单项选择题'''
                    prompt_text = ("Question: " + rec['question'] + "\nOptions:\n" + "\n".join(
                        rec['options']) + "\n\nProvide only the letter as your answer.")

                    max_tokens = 50
                    # 调用模型
                    response = model(
                        img_path,
                        prompt_text,
                    )
                    answer_text = extract_answer_from_response(response)

                # 构建输出
                output_rec = {
                    "prompt": prompt_text,
                    "text": answer_text,
                    "GT_ANSWER": rec.get('answer', ''),
                    "model_id": "Janus-Pro-7B",
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
            response = model(
                img_path,
                prompt,
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
                "model_id": "Janus-Pro-7B",
                "metadata": rec
            }
            results.append(output_rec)

        with open(args.OUTPUT_PATH, 'w', encoding='utf-8') as fout:
            for rec in results:
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')

        print(f"✅ Output written to {args.OUTPUT_PATH}")


if __name__ == '__main__':
    main()
