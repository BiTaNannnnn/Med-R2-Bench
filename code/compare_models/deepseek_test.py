
from pathlib import Path
import argparse
import json
import time
import base64
import mimetypes
import re
import io
import os
from io import BytesIO
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
# ==== Configuration ====
MODEL_NAME = "qwq-32b"
BASE_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_RETRIES = 5
RETRY_DELAY = 1  # seconds

# System prompts
SYSTEM_PROMPT_ANALYSIS = "You are a helpful medical image analysis assistant."
SYSTEM_PROMPT_DISCRETE = "Only respond with one capital letter."
SYSTEM_PROMPT_JOINT    = "Respond with four or five capital letters (A-H), separated by commas."


# ==== Image process ====
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



# ==== Image handling ====
def compress_image(path: str, max_dim: int = 1024, quality: int = 85, size_limit_mb: int = 10) -> bytes:
    """Compress image to JPEG under size limit (MB)."""
    img = Image.open(path)
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    data = buf.getvalue()
    # account for base64 inflation (~1.37x)
    if len(data) * 1.37 > size_limit_mb * 1024 * 1024:
        return compress_image(path, int(max_dim * 0.9), int(quality * 0.9), size_limit_mb)
    return data


def encode_image(path: str, max_dim: int = 1024, quality: int = 85) -> str:
    """Return data URL of compressed image."""
    try:
        img_bytes = compress_image(path, max_dim, quality)
    except Exception:
        with open(path, 'rb') as f:
            img_bytes = f.read()
    b64 = base64.b64encode(img_bytes).decode('utf-8')
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = 'image/jpeg'
    return f"data:{mime};base64,{b64}"

# ==== Generic API call ====
def api_infer(prompt: str,
              system_prompt: str,
              image_path: str,
              client: OpenAI,
              max_dim: int = 1024,
              quality: int = 85,
              size_limit_mb: int = 10,
              max_tokens: int = None,
              temperature: float = None) -> str:
    data_url = encode_image(image_path, max_dim, quality)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}}
        ]}
    ]
    last_exc = None
    for i in range(1, MAX_RETRIES + 1):
        try:
            params = {"model": MODEL_NAME, "messages": messages}
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            if temperature is not None:
                params["temperature"] = temperature
            resp = client.chat.completions.create(**params)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_exc = e
            print(f"[Warning] attempt {i}/{MAX_RETRIES} failed: {e}")
            if i < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    raise last_exc




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
    parser.add_argument('--API_KEY', default='xxx')
    parser.add_argument('--INPUT_PATH', default='xxx/path',required=False)
    parser.add_argument('--OUTPUT_PATH', default='xxx/path',required=False)
    parser.add_argument('--type', default='single', required=False, choices=['single', 'joint'])  # joint_qa  visual_evidence_joint_qa
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

    client = OpenAI(api_key=args.API_KEY, base_url=BASE_URL)

    if args.type == 'single':
        # 读取数据
        with open(args.INPUT_PATH, 'r', encoding='utf-8') as f:
            lines = [l for l in f if l.strip()]

        # 处理数据
        with open(args.OUTPUT_PATH, 'w', encoding='utf-8') as fout:
            for line in tqdm(lines, desc=f"Processing data"):
                rec = json.loads(line)
                img_path = rec.get('image_paths')

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
                    answer_text = api_infer(prompt_text, SYSTEM_PROMPT_DISCRETE, img_path, client,
                                      max_dim=512, quality=75,
                                      max_tokens=max_tokens, temperature=0)


                else:
                    '''单项选择题'''
                    prompt_text = ("Question: " + rec['question'] + "\nOptions:\n" + "\n".join(
                        rec['options']) + "\n\nProvide only the letter as your answer.")
                    max_tokens = 50
                    response = api_infer(prompt_text, SYSTEM_PROMPT_DISCRETE, img_path, client,
                                            max_dim=512, quality=75,
                                            max_tokens=max_tokens, temperature=0)

                    answer_text = extract_answer_from_response(response)

                # 构建输出
                output_rec = {
                    "prompt": prompt_text,
                    "text": answer_text,
                    "GT_ANSWER": rec.get('answer', ''),
                    "model_id": "qwen2.5-vl-7b",
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
            img_path = rec.get('image_paths')
            prompt_text = rec.get('joint_prompt', '')

            # 调用模型
            response = api_infer(prompt_text, SYSTEM_PROMPT_DISCRETE, img_path, client,
                                 max_dim=512, quality=75,
                                 max_tokens=100, temperature=0)

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
                "prompt": prompt_text,
                "text": response,
                "model_modality_answer": model_modality_answer,
                "model_organ_answer": model_organ_answer,
                "model_lesion_answer": model_lesion_answer,
                "model_diagnosis_answer": model_diagnosis_answer,
                "GT_modality_answer": rec.get('modality_answer', ''),
                "GT_organ_answer": rec.get('organ_answer', ''),
                "GT_lesion_answer": rec.get('lesion_answer', ''),
                "GT_diagnosis_answer": rec.get('diagnosis_answer', ''),
                "model_id": "qwen2.5-vl-7b",
                "metadata": rec
            }
            results.append(output_rec)

        with open(args.OUTPUT_PATH, 'w', encoding='utf-8') as fout:
            for rec in results:
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')

        print(f"✅ Output written to {args.OUTPUT_PATH}")


if __name__ == '__main__':
    main()
