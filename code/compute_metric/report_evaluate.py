#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
# 优先使用国内镜像，避免首次下载卡死
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
from tqdm import tqdm
import numpy as np
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import defaultdict
import argparse
import evaluate

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

parser = argparse.ArgumentParser(description="Report-metrics without API")
parser.add_argument("--JSON_PATH", default="xxx/path", help="input jsonl")
parser.add_argument("--GPU", default="6", help="GPU id, int or 'cpu'")
args = parser.parse_args()

device = f"cuda:{args.GPU}" if args.GPU.isdigit() else "cpu"

# 1. 读数据
modality_refs = defaultdict(list)
modality_preds = defaultdict(list)

with open(args.JSON_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        ref = item.get("GT_ANSWER", "")
        pred = item.get("text", "")
        metadata = item.get('metadata')
        modality = metadata.get("modality", "Unknown")
        model_id = item.get('model_id')
        if ref and pred:
            modality_refs[modality].append(ref)
            modality_preds[modality].append(pred)

# 2. 逐模态计算
smoothie = SmoothingFunction().method4
modality_scores = {}

# === 1. 提到循环外 ===
bleu   = evaluate.load('bleu')
bertscore = evaluate.load('bertscore')
rouge  = evaluate.load('rouge')

# === 2. 计算部分 ===
for modality in tqdm(modality_refs.keys(), desc="Scoring"):
    refs = modality_refs[modality]
    preds = modality_preds[modality]


    # BLEU-1
    bleu_result = bleu.compute(predictions=preds, references=[[r] for r in refs], max_order=1)
    avg_bleu = bleu_result['bleu']

    # BERTScore
    bert_result = bertscore.compute(predictions=preds, references=[[r] for r in refs], lang="en")
    avg_f1 = np.mean(bert_result['f1'])

    # ROUGE-1
    rouge_result = rouge.compute(predictions=preds, references=[[r] for r in refs], rouge_types=['rouge1'])
    avg_rouge1 = rouge_result['rouge1']

    modality_scores[modality] = {"bert_f1": avg_f1, "bleu": avg_bleu, "rouge1": avg_rouge1}

# 3. 输出到同目录 Level5_accuracy.jsonl
out_file = os.path.join(
    os.path.dirname(args.JSON_PATH),
    "Level5_accuracy.jsonl"
)

with open(out_file, "w", encoding="utf-8") as f_out:
    for modality, scores in modality_scores.items():
        print(f"model_id = {model_id}, "
              f"{modality}: BERTScore F1 = {scores['bert_f1']:.4f}, "
              f"BLEU = {scores['bleu']:.4f}, ROUGE-1 = {scores['rouge1']:.4f}")

        f_out.write(
            json.dumps(
                {
                    "model_id": model_id,
                    "modality": modality,
                    "BERTScore_F1": round(scores["bert_f1"], 4),
                    "BLEU": round(scores["bleu"], 4),
                    "ROUGE_1": round(scores["rouge1"], 4),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

        print(f"✅ 结果已写入：{out_file}")