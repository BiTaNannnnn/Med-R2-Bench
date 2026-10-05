#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import base64
import mimetypes
import time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from io import BytesIO
from tqdm import tqdm
from PIL import Image
from openai import OpenAI
import argparse
from collections import defaultdict
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from concurrent.futures import ThreadPoolExecutor, as_completed
import evaluate
# Parse command-line arguments






os.environ['CUDA_VISIBLE_DEVICES'] = '6'


parser = argparse.ArgumentParser(description="Compute report generation metrics using DeepSeek and NLP metrics")
parser.add_argument("--API_KEY", default='xxx', help="Your DeepSeek API key"
)
parser.add_argument("--JSON_PATH", default="xxx/path", help="input jsonl")
parser.add_argument("--GPU", default="6", help="GPU id, int or 'cpu'")
args = parser.parse_args()

device = f"cuda:{args.GPU}" if args.GPU.isdigit() else "cpu"


# Initialize DeepSeek client
client = OpenAI(
    api_key=args.API_KEY.strip(),
    base_url="https://api.deepseek.com"
)

# Caching for repeated queries
cache = {}

def extract_key_info(text):
    prompt = (
        "Given the following description of a medical image, extract only clinically relevant information that can be visually determined from the image. "
        "This includes both **normal findings** (e.g., 'no lung opacity', 'normal heart size') and **abnormal findings** (e.g., 'fracture', 'tumor mass'). "
        "Exclude any details that cannot be inferred from the image itself (e.g., patient history, lab values).\n\n"
        f"{text}\n\n"
        "Return a concise, comma-separated list of visually identifiable clinical features. Do not include any irrelevant words or phrases, do not include explanations."
    )
    if text in cache:
        return cache[text]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = response.choices[0].message.content.strip()
    cache[text] = content
    return content

# Load and prepare data
input_path = args.JSON_PATH
modality_refs = defaultdict(list)
modality_preds = defaultdict(list)

caption_items = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        ref = item.get("GT_ANSWER", "")
        pred = item.get("text", "")
        metadata = item.get('metadata')
        modality = metadata.get("modality", "Unknown")
        model_id = item.get('model_id')
        if ref and pred:
            caption_items.append((modality, ref, pred))

# Extract keywords in parallel
print("Extracting keywords in parallel...")
results = {}
with ThreadPoolExecutor(max_workers=50) as executor:
    future_to_info = {}
    for i, (_, ref, _) in enumerate(caption_items):
        future = executor.submit(extract_key_info, ref)
        future_to_info[future] = ("ref", i)
    for i, (_, _, pred) in enumerate(caption_items):
        future = executor.submit(extract_key_info, pred)
        future_to_info[future] = ("pred", i)

    for future in tqdm(as_completed(future_to_info), total=len(future_to_info)):
        kind, idx = future_to_info[future]
        try:
            results[(kind, idx)] = future.result()
        except Exception:
            results[(kind, idx)] = "ERROR"

# Organize by modality
for idx, (modality, ref, pred) in enumerate(caption_items):
    ref_kw = results.get(("ref", idx), "")
    pred_kw = results.get(("pred", idx), "")
    if ref_kw != "ERROR" and pred_kw != "ERROR":
        modality_refs[modality].append(ref_kw)
        modality_preds[modality].append(pred_kw)

# Compute metrics
modality_scores = {}
smoothie = SmoothingFunction().method4


# rouge  = evaluate.load('rouge')
for modality in tqdm(modality_refs.keys(), desc="Computing scores per modality"):
    refs = modality_refs[modality]
    preds = modality_preds[modality]

    _, _, F1 = bert_score(
        preds,
        refs,
        model_type="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        num_layers=12,
        lang="en",
        rescale_with_baseline=False,
        device='cpu'
    )
    avg_f1 = F1.mean().item()


    bleu_scores = [
        sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
        for ref, pred in zip(refs, preds)
    ]
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0


    # # Rouge
    # rouge_result = rouge.compute(predictions=preds, references=[[r] for r in refs], rouge_types=['rouge1'])
    # avg_rouge1 = rouge_result['rouge1']

    modality_scores[modality] = {"bert_f1": avg_f1, "bleu": avg_bleu,
                                 # "avg_rouge1":avg_rouge1
                                 }


# 创建输出字典
output_data = []

for modality, scores in modality_scores.items():
    output_data.append({
        "modality": modality,
        "bert_f1": scores["bert_f1"],
        "bleu": scores["bleu"]
    })
    print(
        f"{modality}: BERTScore F1 = {scores['bert_f1']:.4f}, BLEU = {scores['bleu']:.4f}, "
        # f"Rouge1 = {scores['avg_rouge1']:.4f}"
    )

    # 3. 输出到同目录 Level5_accuracy.jsonl
    out_file = os.path.join(
        os.path.dirname(args.JSON_PATH),
        "Level5_llm_accuracy.jsonl"
    )
    with open(out_file, "w", encoding="utf-8") as json_file:
        json.dump(output_data, json_file, indent=4)

    print(f"Results have been written to {out_file}")
