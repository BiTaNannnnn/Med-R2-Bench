'''

首先去DrVD jointQA 里找 MRI+CT  的数据
然后在归一化
最后做QA

'''
import glob
import numpy as np
from PIL import Image, ImageDraw
import nibabel as nib
import random
import copy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import argparse
import json
import os
import re
import sys
import tempfile
from typing import Iterable, Dict, Any, List, Optional



def normalize_modality_option(opt: str) -> str:
    """
    去掉前缀 A./B./C./D. 以及多余引号
    """
    opt = opt.strip()
    opt = re.sub(r"^[A-Z]\.\s*", "", opt)
    opt = opt.strip("'\" ")
    return opt





def rebuild_modality_options(old_options, old_answer):
    """
    修改 record 中的 modality_options 和 modality_answer
    只保留 CT / MRI，随机 A/B，但保证答案语义正确
    """

    # --- 判断原答案语义 ---
    answer_low = old_answer.lower()
    if "ct" in answer_low:
        correct_sem = "CT"
    elif "magnetic resonance imaging" in answer_low:
        correct_sem = "Magnetic Resonance Imaging"
    else:
        return

    # --- 随机分配 A / B ---
    items = ["CT", "Magnetic Resonance Imaging"]
    random.shuffle(items)

    new_options = []
    new_answer = None
    labels = ["A", "B"]

    for label, text in zip(labels, items):
        opt = f"{label}. {text}"
        new_options.append(opt)
        if text == correct_sem:
            new_answer = opt

    return new_options, new_answer



def _find_correct_text(options: List[str], answer: str) -> Optional[str]:
    """
    根据 organ_answer 找到正确选项的“文本部分”（不带 A./B. 前缀）
    兼容 answer 可能是 'A. pancreas' 或仅 'pancreas'
    """
    if not isinstance(answer, str):
        return None
    ans_text = normalize_modality_option(answer).lower()

    # 优先：在 options 里精确匹配文本
    for opt in options:
        if normalize_modality_option(opt).lower() == ans_text:
            return normalize_modality_option(opt)

    # 其次：如果 answer 带字母且字母能定位
    m = re.match(r"^([A-Z])\.\s*", answer.strip())
    if m:
        ans_letter = m.group(1)
        for opt in options:
            if opt.strip().startswith(f"{ans_letter}."):
                return normalize_modality_option(opt)

    return None

def normalize_organ_mcq(organ_options, organ_answer):
    """
    如果 organ_options > 4，则重采样为 4 选项并更新 organ_answer。
    返回 True 表示发生了修改；False 表示未修改（<=4 或数据不合法）。
    """


    options = organ_options
    answer = organ_answer

    # 如果已经 <=4，不动（你也可以改成 len!=4 就强制重建）
    if len(options) > 4:
        # 找到正确答案的文本
        correct_text = _find_correct_text(options, answer)
    
        # 去重（保序）后的候选池（文本）
        seen = set()
        texts = []
        for opt in options:
            t = normalize_modality_option(opt)
            tl = t.lower()
            if t and tl not in seen:
                texts.append(t)
                seen.add(tl)
    
        # 错误候选（排除正确）
        wrong_candidates = [t for t in texts if t.lower() != correct_text.lower()]
        if len(wrong_candidates) < 3:
            return False
    
        # 采样 3 个错误选项
        wrong_pick = random.sample(wrong_candidates, 3)
    
        # 组成 4 个，并随机打乱
        all_texts = [correct_text] + wrong_pick
        random.shuffle(all_texts)
    
        # 重新标号 A/B/C/D，并找正确答案对应的字母
        labels = ["A", "B", "C", "D"]
        new_options = []
        new_answer = None
        for lab, txt in zip(labels, all_texts):
            opt_str = f"{lab}. {txt}"
            new_options.append(opt_str)
            if txt.lower() == correct_text.lower():
                new_answer = opt_str
        return new_options, new_answer

    else:
       return options, answer
        

def join_options(options):
    """
    options 可以是：
    - list[str]  -> 直接 join
    - dict       -> 格式化成 'A. xxx'
    """
    if isinstance(options, dict):
        return "\n".join([f"{k}. {v}" for k, v in options.items()])
    return "\n".join(options)

















if __name__ == "__main__":
    box_points_path = "xxx/path"  # 你的 region 标注 JSON
    vqa_abnormality_path = 'xxx/path'
    out_qas = "xxx/path"

    ############# add DrVD-Bench MRI+CT 数据
    DrVD_Bench_joint = 'xxx/path'
    built = 0
    with open(DrVD_Bench_joint, "r", encoding="utf-8") as fin, \
        open(out_qas, "w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)  # ✅ 关键：把 str 变成 dict
            modality = record.get("modality")
            if modality == "Magnetic Resonance Imaging" or modality == "CT":
                oldpath = record.get("image_paths")
                record["image_paths"] = "xxx/path" + oldpath
                if "options_num" in record:
                    record["question_num"] = record.pop("options_num")

                # change q1 options + answer
                modality_options = record.get("modality_options")
                modality_answer = record.get("modality_answer")
                options1 , answer1 = rebuild_modality_options(modality_options, modality_answer)
                record["modality_options"] = options1
                record["modality_answer"] = answer1

                # change q2 options + answer
                organ_options = record.get("organ_options")
                organ_answer = record.get("organ_answer")
                options2, answer2 =  normalize_organ_mcq(organ_options, organ_answer)
                record["organ_options"] = options2
                record["organ_answer"] = answer2

                # change q3 options + answer
                lesion_options = record.get("lesion_options")
                lesion_answer = record.get("lesion_answer")
                options3, answer3 = normalize_organ_mcq(lesion_options, lesion_answer)
                record["lesion_options"] = options3
                record["lesion_answer"] = answer3

                # change q4 options + answer
                diagnosis_options = record.get("diagnosis_options")
                diagnosis_answer = record.get("diagnosis_answer")
                options4, answer4 = normalize_organ_mcq(diagnosis_options, diagnosis_answer)
                record["diagnosis_options"] = options4
                record["diagnosis_answer"] = answer4

                # ================= joint_prompt =================
                q1 = "What imaging modality is used in this image?"  # 该图像采取哪种成像方式？ CT/MRI 二选一
                q2 = "Which organ appears to be abnormal in this image?"  # 红框里是什么器官有病变？
                q3 = "Based on the abnormal organ, what lesion or finding is most clearly visible?"
                q4 = "Considering all the above findings, what is the most likely diagnosis?"
                joint_prompt = (
                        "You are participating in a step-by-step medical reasoning diagnosis task "
                        "based on interpretation of a medical image.\n\n"
                        "Please answer the following four questions sequentially. Each question builds "
                        "upon the reasoning of the previous one. Carefully analyze the image and select "
                        "the most appropriate answer at each step. For each question, choose one capital "
                        "letter (A, B, C, or D). Do not skip any step.\n\n"
                        f"1. {q1}\nOptions:\n" + join_options(options1) + "\n\n"
                        f"2. {q2}\nOptions:\n" + join_options(options2) + "\n\n"
                        f"3. {q3}\nOptions:\n" + join_options(options3) + "\n\n"
                        f"4. {q4}\nOptions:\n" + join_options(options4) + "\n\n"
                        "Please reply with your four selected letters in order, separated by commas "
                        "(e.g., A,C,B,A). Do not provide explanations."
                )

                # ================= 用原结构做模板 =================

                qa = {
                    "unique_id": record.get("unique_id"),
                    "image_paths": record.get("image_paths"),
                    "joint_prompt": joint_prompt,

                    "question_num": 4,

                    "modality_options": options1,
                    "modality_answer": answer1,

                    "organ_options": options2,
                    "organ_answer": answer2,

                    "lesion_options": options3,
                    "lesion_answer": answer3,

                    "diagnosis_options": options4,
                    "diagnosis_answer": answer4,

                    "modality": modality,
                    "modal_response": {"modality": None, "organ": None, "lesion": None, "diagnosis": None}
                }

                fout.write(json.dumps(qa, ensure_ascii=False) + "\n")
                built += 1
            print(f"生成对抗QA(每个id一次): {built}, 输出: {out_qas}")



