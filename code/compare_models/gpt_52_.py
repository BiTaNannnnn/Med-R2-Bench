#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch API 最小改动版（含图片 base64）
"""
import json, time, argparse, re, base64, mimetypes
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import time, sys
import os, httpx
import io
os.environ["https_proxy"] = "http://xxx:xxxx"
os.environ["http_proxy"]  = "http://xxx:xxxx"


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


def build_request(sample: dict, idx: int, mode: str) -> dict:
    """构造 Batch 要求的单行请求（含图片）"""
    img_path_old= sample['image_paths']
    # 如果路径包含指定的前缀并且该路径存在，则进行替换
    if 'validation_data' in img_path_old:
        img_path = img_path_old.replace(
            'xxx/path',
            'xxx/path')

    if 'DrVD-Bench' in img_path_old:
        img_path = img_path_old.replace('xxx/path',
                                        'xxx/path')

    img_b64 = image_to_b64(img_path)   # 你原字段叫 image_paths
    if mode == "single":
        if sample.get('question_type') == 'open_question':
            ## report
            if "report" in str(sample.get("task", "")).lower():
                prompt = sample['question'] + '\n\nWrite the image report as a single coherent paragraph, no more than 500 words.'
                max_tokens = 500

            else:
                ## open quesion size
                prompt = sample['question'] + '\n\nWrite the correct answer, no more than 10 words.'
                max_tokens = 50
        else:
            ## single quesion
            prompt = ("Question: " + sample['question'] +
                      "\nOptions:\n" + "\n".join(sample['options']) +
                      "\n\nProvide only the letter as your answer.")
            max_tokens = 10
    else:  # joint qa
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

    return {
        "custom_id": f"{mode}_{idx}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-5.2",      # 你要测的模型名，不动
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": 0
        },
        "prompt": prompt,
        "sample": sample
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--API_KEY', default='xxx')
    parser.add_argument('--INPUT_PATH',  default='xxx/path')
    parser.add_argument('--OUTPUT_PATH', default='xxx/path')
    parser.add_argument('--type', choices=['single', 'joint'], default='joint') # joint_qa Level3
    args = parser.parse_args() # xxx/path
    # 确保输出目录存在
    output_path = Path(args.OUTPUT_PATH)
    output_dir = output_path.parent

    # 创建目录（如果不存在）
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Created output directory: {output_dir}")
    except Exception as e:
        print(f"[WARNING] Failed to create directory {output_dir}: {e}")


    client = OpenAI(api_key=args.API_KEY,
                    http_client=httpx.Client(
                        proxies="http://xxx:xxxx",
                        timeout=120  # 给慢节点 2 分钟握手时间
                    ),
                    max_retries=5
                    )

    # 1. 逐行读 jsonl
    samples = []
    with open(args.INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    # 2. 构造请求池
    requests = [build_request(s, i, args.type)
                for i, s in enumerate(tqdm(samples, desc="Building requests"))]

    # 3. 写 jsonl（只保留 batch 需要的字段）
    input_jsonl = "batch_input.jsonl"
    with open(input_jsonl, "w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps({k: r[k] for k in ["custom_id", "method", "url", "body"]}, ensure_ascii=False) + "\n")

    # 4. 上传 & 建 batch
    uploaded = client.files.create(file=open(input_jsonl, "rb"), purpose="batch")
    batch_obj = client.batches.create(input_file_id=uploaded.id,
                                        endpoint="/v1/chat/completions",
                                        completion_window="24h")
    print(">>> batch 已提交，id:", batch_obj.id)

    # 5. 轮询（静默 + 状态变化才打印）
    last_status = None
    while True:
        job = client.batches.retrieve(batch_obj.id)
        if job.status != last_status:  # 只在状态变化时输出
            print(f">>> batch 状态: {job.status}  "
                  f"(completed={job.request_counts.completed}, failed={job.request_counts.failed})")
            last_status = job.status

        if job.status in {"completed", "failed", "expired", "cancelled"}:
            break
        time.sleep(30)

    if job.status != "completed":
        raise RuntimeError(f"Batch 失败！status={job.status}")

    # 6. 下载结果 + 失败详情
    if job.output_file_id:  # 正确结果
        result_text = client.files.content(job.output_file_id).text
        Path("xxx/path").write_text(result_text)
        print(">>> 正确结果已保存 → batch_success.jsonl")

    if job.error_file_id:  # 失败详情
        error_text = client.files.content(job.error_file_id).text
        Path("xxx/path").write_text(error_text)
        print(">>> 失败详情已保存 → batch_failed.jsonl")

    # 6. 下载结果
    # result_text = client.files.content(job.output_file_id).text

    # 7. 解析并写回你原来的输出格式
    results = {}
    for line in result_text.splitlines():
        if not line.strip():
            continue
        resp = json.loads(line)
        idx = int(resp["custom_id"].split("_")[1])
        text = resp["response"]["body"]["choices"][0]["message"]["content"]
        results[idx] = text

    fout = open(args.OUTPUT_PATH, "w", encoding="utf-8")
    for idx, rec in enumerate(samples):
        text = results.get(idx, "")
        if args.type == "single":
            if rec.get('question_type') == 'open_question':
                answer = text
            else:
                answer = extract_answer_from_response(text)
            out = {
                "prompt": requests[idx]["prompt"],
                "text": answer,
                "GT_ANSWER": rec.get('answer', ''),
                "model_id": "gpt-5.2",
                "metadata": rec
            }
        else:  # joint
            r1, r2, r3, r4 = (text.split(',') if len(text.split(',')) == 4 else robust_parse_response(text))
            out = {
                "prompt": requests[idx]["prompt"],
                "text": text,
                "model_modality_answer": r1,
                "model_organ_answer": r2,
                "model_lesion_answer": r3,
                "model_diagnosis_answer": r4,
                "GT_modality_answer": rec.get('modality_answer', ''),
                "GT_organ_answer": rec.get('organ_answer', ''),
                "GT_lesion_answer": rec.get('lesion_answer', ''),
                "GT_diagnosis_answer": rec.get('diagnosis_answer', ''),
                "model_id": "gpt-5.2",
                "metadata": rec
            }
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    fout.close()
    print(">>> 结果已写入:", args.OUTPUT_PATH)


if __name__ == '__main__':
    main()