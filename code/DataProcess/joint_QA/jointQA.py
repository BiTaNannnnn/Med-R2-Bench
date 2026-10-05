'''

首先去validation_vqa_abnormality.csv 这里找有问题的id + 区域
然后在visual_evidence_box_points.json 这里找对应的数据
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


def process_data(record: dict) -> dict:
    # 获取路径和文件名
    file_name = record.get('Volumename', "")
    Sentence = record.get('Sentence', "")
    # 获取区域信息和异常描述
    region = record.get('Anatomy', "")
    abnormality = record.get("Abnormality", "")

    # 处理region，将其按'/'分割，取第一个"lung"
    region_parts = region.split("/")[0]

    # 处理异常描述，按逗号分隔
    # abnormalities = [ab.strip() for ab in abnormality.split(",")]
    if abnormality == 'no findings':
         return None

    # 返回处理后的数据
    return {
        "unique_id": file_name,  # 文件名作为ID
        "asked_region": region_parts,
        "Sentence": Sentence
    }


def read_jsonl_folder(folder_path: str):
    records = []  # 用来保存所有的处理后的记录
    df = pd.read_csv(folder_path)

    seen = set()  # 记录已经出现过的 (Volumename, Anatomy) 组合

    for index, row in df.iterrows():
        # 提取 Anatomy 列中的第一个词（以 / 分割）

        if pd.isna(row["Anatomy"]) or row["Anatomy"] == "":
            continue  # 跳过空值的行
        anatomy_part = row["Anatomy"].split("/")[0]

        # 筛选
        if anatomy_part =='others' or anatomy_part =="abdomen" or anatomy_part =="pleura":
            continue

        # 构造唯一 key，包含 Volumename 和提取的 Anatomy 部分
        key = (row["Volumename"], anatomy_part)

        # 如果这个组合已经处理过了，就跳过
        if key in seen:
            continue
        seen.add(key)

        record = row.to_dict()
        processed_record = process_data(record)  # 处理每一行数据

        if processed_record is not None:
            records.append(processed_record)

    return records


# 从 CSV 文件读取数据
def load_from_csv(file_path):
    return pd.read_csv(file_path)



def yn_semantic(s: str) -> Optional[str]:
    """提取 yes/no 语义（鲁棒处理 A./B. 前缀、大小写、后续句子等）"""
    if not s:
        return None
    s = s.strip().lower()
    s = re.sub(r'^[a-z]\.\s*', '', s)  # 去掉 A. / B.
    if s.startswith("MRI"):
        return "Magnetic Resonance Imaging"
    if s.startswith("CT"):
        return "CT"
    return None

def make_random_yesno_options(correct_sem: str):
    """
    correct_sem: 'yes' or 'no'
    返回 (options_list, answer_str)
    """
    sems = ["Magnetic Resonance Imaging", "CT"]
    random.shuffle(sems)  # 随机决定 yes/no 在 A/B 的位置
    labels = ["A", "B"]

    options = []
    answer = None
    for label, sem in zip(labels, sems):
        opt = f"{label}. {sem}"
        options.append(opt)
        if sem == correct_sem:
            answer = opt

    # 理论上不会 None，这里做保险
    if answer is None:
        # fallback：强行设定 A/B
        options = ["A. Magnetic Resonance Imaging", "B. CT"]
        answer = "A. Magnetic Resonance Imaging" if correct_sem == "Magnetic Resonance Imaging" else "B. no"

    return options, answer



def make_random_4choice_options(correct_text, candidate_pool):
    """
    correct_text: 正确答案内容（如 asked_region）
    candidate_pool: 候选池（如 askregion_list）

    返回:
        options_dict: {"A": "...", "B": "...", "C": "...", "D": "..."}
        answer_letter: "A"/"B"/"C"/"D"
    """
    wrong_candidates = [x for x in candidate_pool if x != correct_text]
    if len(wrong_candidates) < 3:
        return None, None

    wrong_options = random.sample(wrong_candidates, 3)
    all_options = [correct_text] + wrong_options
    random.shuffle(all_options)

    labels = ["A", "B", "C", "D"]
    options_dict = {}
    answer_str = None

    for label, text in zip(labels, all_options):
        options_dict[label] = text
        if text == correct_text:
            answer_str = f"{label}. {text}"

    return options_dict, answer_str

def join_options(options):
    """
    options 可以是：
    - list[str]  -> 直接 join
    - dict       -> 格式化成 'A. xxx'
    """
    if isinstance(options, dict):
        return "\n".join([f"{k}. {v}" for k, v in options.items()])
    return "\n".join(options)


def strip_label(ans: str) -> str:
    """把 'C. heart' -> 'heart'"""
    if not ans:
        return ""
    m = re.match(r'^[A-D]\.\s*(.*)$', ans.strip())
    return m.group(1).strip() if m else ans.strip()

def make_joint_4choice_from_q2_q3(options2, answer2, options3, answer3):
    correct_organ = strip_label(answer2)
    correct_abn = strip_label(answer3)

    organ_others = [v for v in options2.values() if v != correct_organ]
    abn_others = [v for v in options3.values() if v != correct_abn]

    if len(organ_others) < 3 or len(abn_others) < 3:
        return None, None

    random.shuffle(abn_others)

    correct_pair = f"{correct_organ} with {correct_abn}"
    distractors = [f"{o} with {a}" for o, a in zip(organ_others[:3], abn_others[:3])]

    all_pairs = [correct_pair] + distractors
    random.shuffle(all_pairs)

    labels = ["A", "B", "C", "D"]
    options4_list, answer4 = [], None

    for lab, text in zip(labels, all_pairs):
        opt = f"{lab}. {text}"
        options4_list.append(opt)
        if text == correct_pair:
            answer4 = opt

    return options4_list, answer4


def joint_data(record, boximg_path, abnormalities_list, askregion_list):
    asked_region = record.get('asked_region')
    abnormal = record.get("abnormalities")
    unique_id = record.get('unique_id')

    if abnormal not in abnormalities_list:
        return None

    # ================= Q1 =================
    q1 = "What imaging modality is used in this image?" # 该图像采取哪种成像方式？ CT/MRI 二选一
    options1, answer1 = make_random_yesno_options(correct_sem="CT")

    # ================= Q2 =================
    q2 = "Which organ appears to be abnormal in this image?" # 红框里是什么器官有病变？
    options2, answer2 = make_random_4choice_options(
        correct_text=asked_region,
        candidate_pool=askregion_list
    )

    # ================= Q3 =================
    q3 = "Based on the abnormal organ, what lesion or finding is most clearly visible?"
    # 基于这个不正常的器官，最清晰可见的病变或体征是什么？
    options3, answer3 = make_random_4choice_options(
        correct_text=abnormal,
        candidate_pool=abnormalities_list
    )

    # ================= Q4 =================
    q4 = "Considering all the above findings, what is the most likely diagnosis?"
    # 综合以上所有发现，最可能的诊断是什么？
    options4, answer4 = make_joint_4choice_from_q2_q3(
        options2, answer2, options3, answer3
    )

    # ================= joint_prompt =================
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
        "unique_id": unique_id,
        "image_paths" : boximg_path,
        "joint_prompt" :joint_prompt,

        "question_num" : 4,

        "modality_options" : options1,
        "modality_answer" : answer1,

        "organ_options" : options2,
        "organ_answer" : answer2,

        "lesion_options" : options3,
        "lesion_answer" : answer3,

        "diagnosis_options" : options4,
        "diagnosis_answer" : answer4,

        "modality" : "CT",
        "modal_response": {"modality": None,  "organ": None, "lesion": None, "diagnosis": None}
        }

    return qa

if __name__ == "__main__":
    box_points_path = "xxx/path"  # 你的 region 标注 JSON
    vqa_abnormality_path = 'xxx/path'
    out_qas = "xxx/path"

    loaded_records_csv = load_from_csv('xxx/path')
    # 将 DataFrame 转换为字典列表
    records_list = loaded_records_csv.to_dict(orient='records')

    # 找出所有疾病 + 小于四个词的
    abnormalities_list = []
    for record in records_list:
        abn = record.get("abnormalities", "")
        # 按空格分词，词数大于 4 就跳过
        if len(abn.split()) > 4:
            continue
        abnormalities_list.append(abn)

    # 找出所有部位（保序去重，排除 "others"）
    askregion_list = []
    seen = set()
    for record in records_list:
        abn = record.get("asked_region", "")
        if (
                abn
                and abn.lower() != "others"  # ✅ 排除 others
                and abn not in seen
        ):
            askregion_list.append(abn)
            seen.add(abn)


    out_qas = "xxx/path"
    emitted_ids = set()
    mode = "a" if os.path.exists(out_qas) else "w"  # 追加到文件末尾 # 已经写出的 unique_id
    total = 0
    built = 0
    Path = "xxx/path"
    Path_no_box = "xxx/path"
    # 找到对应的box img
    with open(out_qas, "a", encoding="utf-8") as fout:
        for record in records_list:
            unique_id = record['unique_id']
            unique_id = unique_id.split(".")[0]
            asked_region = record['asked_region']

            ''''# region_box 找到画对的框 '''
            box_path_global = os.path.join(Path, f"box_{unique_id}", f"{asked_region}_*")  # 使用 * 来匹配 xxx 部分
            # 使用 glob 查找所有符合模式的文件
            boximg_path = glob.glob(box_path_global)
            if not boximg_path:
                # 可选：打印一下方便排查
                # print(f"[warn] no match: {box_path_global}")
                continue

            boximg_path = boximg_path[0]  # 或者 sorted(matches)[0] 确保确定性

            '''joint qa'''
            qa = joint_data(record, boximg_path, abnormalities_list, askregion_list)


            if qa is None:
                continue

            fout.write(json.dumps(qa, ensure_ascii=False) + "\n")
            built += 1
    print(f"输入记录: {total}, 生成对抗QA(每个id一次): {built}, 输出: {out_qas}")







