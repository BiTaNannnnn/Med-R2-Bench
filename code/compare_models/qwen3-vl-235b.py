# -*- coding: utf-8 -*-
"""
非 Batch 版（逐条调用 chat.completions）
"""
import json, time, argparse, re, base64, mimetypes, io
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import httpx
import os
from google import genai
from google.genai import types
import base64




# ---------------- 你原来的函数 ----------------
def extract_answer_from_response(response: str) -> str:
    response = response.strip()
    if len(response) == 1 and response.upper() in "ABCDEFGH":
        return response.upper()
    for pat in [r'[选答是]\s*[：:]?\s*([A-H])', r'[Aa]nswer[:\s]+([A-H])', r'\b([A-H])\b']:
        m = re.search(pat, response)
        if m:
            return m.group(1).upper()
    return response


def robust_parse_response(response):
    if isinstance(response, str):
        response = re.sub(r'[\[\]]', '', response)
        parts = [p.strip().strip("'\"") for p in response.split(',') if p.strip()]
    else:
        parts = response
    letters = []
    for part in parts:
        letters += re.findall(r'[A-Z]', part)
    while len(letters) < 4:
        letters.append('unknown')
    return letters[:4]
# ---------------------------------------------


def image_to_b64(img_path: str) -> str:
    """本地图片 -> base64 字符串"""
    mime, _ = mimetypes.guess_type(img_path)
    if mime is None:
        mime = "image/jpeg"
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{encoded}"


# ------------ 构造单条消息 ------------
def build_messages(sample: dict, mode: str):
    img_path_old = sample['image_paths']
    if 'validation_data' in img_path_old:
        img_path = img_path_old.replace(
            'xxx/path',
            'xxx/path')
    elif 'DrVD-Bench' in img_path_old:
        img_path = img_path_old.replace('xxx/path',
                                        'xxx/path')
    else:
        img_path = img_path_old

    # img_bytes = image_to_b64(img_path)
    # img_bytes = Path(img_path).read_bytes()
    img_b64 = image_to_b64(img_path)

    if mode == "single":
        if sample.get('question_type') == 'open_question':
            if "report" in str(sample.get("task", "")).lower():
                prompt = sample['question'] + '\n\nWrite the image report as a single coherent paragraph, no more than 500 words.'
                max_tokens = 500
            else:
                prompt = sample['question'] + '\n\nWrite the correct answer, no more than 10 words.'
                max_tokens = 50
        else:
            prompt = ("Question: " + sample['question'] +
                      "\nOptions:\n" + "\n".join(sample['options']) +
                      "\n\nProvide only the letter as your answer.")
            max_tokens = 10
    else:  # joint
        prompt = sample.get('joint_prompt', '')
        max_tokens = 20

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": img_b64}}
            ]
        }
    ]


    return messages, max_tokens, prompt

# ------------ 主函数 ------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--API_KEY', default='xxx')
    parser.add_argument('--INPUT_PATH',  default='xxx/path')
    parser.add_argument('--OUTPUT_PATH', default='xxx/path')
    # parser.add_argument('--INPUT_PATH',  default='xxx/path')
    # parser.add_argument('--OUTPUT_PATH', default='xxx/path')
    parser.add_argument('--type', choices=['single', 'joint'], default='single') # Level3 joint_qa
    args = parser.parse_args()

    output_path = Path(args.OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(
        api_key=args.API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 1. 读样本
    samples = []
    with open(args.INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    # 2. 逐条调用（带进度）
    results = []
    pbar = tqdm(samples, desc="Inferring")
    for idx, sample in enumerate(pbar):
        # 实时显示当前图片路径
        pbar.set_postfix({"img": sample.get('image_paths', 'unknown')[:60]})
        messages, max_tokens, prompt = build_messages(sample, args.type)
        try:
            resp = client.chat.completions.create(
                model="qwen3-vl-235b-a22b-instruct",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0
            )
            text = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[WARN] 样本 {idx} 失败: {e}")
            text = "模型API调用错误ing"

        # 3. 解析答案
        if args.type == "single":
            if sample.get('question_type') == 'open_question':
                answer = text
            else:
                answer = extract_answer_from_response(text)
            out = {
                "prompt": prompt,
                "text": answer,
                "GT_ANSWER": sample.get('answer', ''),
                "model_id": "qwen3-vl-235b",
                "metadata": sample
            }
        else:  # joint
            r1, r2, r3, r4 = (text.split(',') if len(text.split(',')) == 4 else robust_parse_response(text))
            out = {
                "prompt": prompt,
                "text": text,
                "model_modality_answer": r1,
                "model_organ_answer": r2,
                "model_lesion_answer": r3,
                "model_diagnosis_answer": r4,
                "GT_modality_answer": sample.get('modality_answer', ''),
                "GT_organ_answer": sample.get('organ_answer', ''),
                "GT_lesion_answer": sample.get('lesion_answer', ''),
                "GT_diagnosis_answer": sample.get('diagnosis_answer', ''),
                "model_id": "qwen3-vl-235b",
                "metadata": sample
            }
        results.append(out)

    # 4. 写结果
    with open(args.OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(">>> 结果已写入:", args.OUTPUT_PATH)


if __name__ == '__main__':
    main()
