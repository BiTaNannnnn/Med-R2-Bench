
'''
json 文件中
模态 = ['Ultrasound', 'Magnetic_Resonance_Imaging', 'CT', 'Radiograph', 'Pathology']
      ['超声', '磁共振成像', 'CT', 'X光片', '病理学']


'''


import json
import re
import random
import os
from typing import Dict, Any, Optional, List, Tuple

import json
import os



# 提取目标实体（如 stomach）
QUESTION_PATTERNS = [
    re.compile(r"^\s*Which\s+color\s+represents\s+(?:the\s+)?(.+?)\s+in\s+this\s+image\??\s*$", re.IGNORECASE),
]

def extract_target_entity(question: str) -> Optional[str]:
    if not isinstance(question, str):
        return None
    for pat in QUESTION_PATTERNS:
        m = pat.match(question.strip())
        if m:
            return m.group(1).strip(" .?;,")
    return None

def strip_label_prefix(option: str) -> str:
    # 去掉 "A. " / "B." 等前缀
    if not isinstance(option, str):
        return ""
    return re.sub(r"^\s*[A-Z]\.\s*", "", option.strip())

def find_correct_index(options: List[str], answer: str) -> Optional[int]:
    # 1) 直接匹配整项
    try:
        return options.index(answer)
    except ValueError:
        pass
    # 2) "C. Blue" 或 "C"
    m = re.match(r"^([A-Z])(?:\.\s*.*)?$", str(answer).strip(), re.IGNORECASE)
    if m:
        idx = ord(m.group(1).upper()) - ord('A')
        if 0 <= idx < len(options):
            return idx
    # 3) 文本部分相等
    ans_text = strip_label_prefix(answer).lower()
    for i, opt in enumerate(options):
        if strip_label_prefix(opt).lower() == ans_text and ans_text != "":
            return i
    return None

def extract_correct_color(options: List[str], answer: str) -> Optional[str]:
    if not isinstance(options, list) or not options:
        return strip_label_prefix(answer) or None
    idx = find_correct_index(options, answer)
    if idx is not None and 0 <= idx < len(options):
        return strip_label_prefix(options[idx])
    # fallback
    txt = strip_label_prefix(answer)
    return txt or None

def ensure_trailing_newline(path: str) -> None:
    # 确保旧的输出文件末尾有换行，避免黏连
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) != b"\n":
            f.write(b"\n")

def convert_jsonl_to_open(in_path: str, out_path: str) -> None:
    processed = 0
    written = 0

    ensure_trailing_newline(out_path)  # 只在追加前做一次
    with open(out_path, "a", encoding="utf-8") as fout:  # 追加，不覆盖
        with open(in_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # 在原数据后面添加 "question_type": "normal_question"
                obj["question_type"] = "normal_question"
                processed += 1


                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1

    print(f"Scanned: {processed}, Appended(open-Q): {written}, Output: {out_path}")






def find_json(json_Path):


    # 用一个列表来存储所有的task
    tasks = set()

    # 逐行读取文件并解析每行的JSON数据
    with open(json_Path, 'r') as f:
        for line in f:
            # 解析每一行作为一个独立的JSON对象
            try:
                data = json.loads(line.strip())  # 使用strip()去掉换行符
                tasks.add(data['modality'])  # 将task添加到集合中，自动去重
            except json.JSONDecodeError:
                print("Error parsing line:", line)

    # 将集合转换为列表并输出去重后的task列表
    tasks = list(tasks)
    print(tasks)

    # 'artifact_recognition'# 伪影
    # 'organ_recognition'



def find_region_location_jsonl(in_path, out_path):
    modalities = {"Magnetic_Resonance_Imaging", "CT"}
    tasks = {"organ_location"}

    kept = 0
    total = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("modality") not in modalities:
                continue
            if obj.get("task") not in tasks:
                continue

            # add prefix to image_paths if needed
            img = obj.get("image_paths")
            if isinstance(img, str):
                obj["image_paths"] = make_full_image_path(img)

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Total: {total}, Kept: {kept}, Output: {out_path}")

def filter_jsonl(in_path, out_path):
    # 允许的取值
    modalities = {"Magnetic_Resonance_Imaging", "CT"}
    tasks = {
        "artifact_recognition",
        "organ_recognition",
        "organ_location",
        "lesion_recognition",
        "noise_recognition",
        "view_recognition",
        "diagnosis",
        "organtissue_recognition",
        "bodypart_recognition",
    }

    kept = 0
    total = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # 跳过坏行；也可以选择 raise
                continue

            modality = obj.get("modality")
            task = obj.get("task")

            # 第一步筛选
            if modality not in modalities:
                continue

            # 第二步筛选
            if task not in tasks:
                continue

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Total: {total}, Kept: {kept}, Output: {out_path}")



def make_full_image_path(p: str) -> str:
    BASE_PREFIX = "xxx/path"
    if not p:
        return p
    # 已是绝对路径则不修改
    if os.path.isabs(p):
        return p
    # 若已有指定前缀，也不重复添加
    if p.startswith((""+BASE_PREFIX, BASE_PREFIX + "/")):
        return p
    # 去掉可能的前导 ./ 再拼接
    p = p.lstrip("./")
    return os.path.normpath(os.path.join(BASE_PREFIX, p))

def append_noise_to_json_array(in_path: str, out_path: str):
    # 读已有 A.json（如果存在且是 JSON 数组）
    TEMPLATE_KEYS = [
        "unique_id",
        "image_paths",
        "question",
        "options",
        "answer",
        "modality",
        "task",
        "options_num",
    ]

    out_records = []
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                out_records = existing
            else:
                print(f"Warning: {out_path} 不是 JSON 数组，将覆盖重写。")
        except json.JSONDecodeError:
            print(f"Warning: {out_path} 不是有效 JSON，将覆盖重写。")

    added = 0
    total = 0
    with open(in_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # 跳过坏行
                continue

            if obj.get("task") != "noise_recognition":
                continue

            # 按模板键顺序重建；值来自输入行
            new_obj = {k: obj.get(k) for k in TEMPLATE_KEYS}
            # 这里加前缀
            new_obj["image_paths"] = make_full_image_path(new_obj.get("image_paths"))

            out_records.append(new_obj)
            added += 1

    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(out_records, fout, ensure_ascii=False, indent=4)

    print(f"扫描行数: {total}, 追加 noise_recognition: {added}, 输出: {out_path}, 总条数: {len(out_records)}")

def strip_label_prefix(option: str) -> str:
    # 去掉 "A. " / "B." 前缀，保留文本部分
    if not isinstance(option, str):
        return ""
    return re.sub(r"^[A-Z]\.\s*", "", option.strip())

def find_correct_index(options: List[str], answer: str) -> Optional[int]:
    # 1) 精确匹配到 options 中的某一项
    try:
        idx = options.index(answer)
        return idx
    except ValueError:
        pass
    # 2) answer 形如 "C. Blue" 或 "C"
    m = re.match(r"^([A-Z])(?:\.\s*.*)?$", str(answer).strip(), re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        idx = ord(letter) - ord('A')
        if 0 <= idx < len(options):
            return idx
    # 3) answer 文本部分与某个选项文本部分一致
    ans_text = strip_label_prefix(answer).lower()
    for i, opt in enumerate(options):
        if strip_label_prefix(opt).lower() == ans_text and ans_text != "":
            return i
    return None

def pick_random_wrong_option(options: List[str], correct_idx: int) -> Tuple[str, int]:
    wrongs = [(i, opt) for i, opt in enumerate(options) if i != correct_idx]
    if not wrongs:
        return "", -1
    i, opt = random.choice(wrongs)
    return opt, i


def extract_target_entity(question: str) -> Optional[str]:
    """
    从 'Which color represents stomach in this image?' 抽取 'stomach'
    也兼容 'the stomach'
    """
    if not isinstance(question, str):
        return None
    m = re.search(r"which\s+color\s+represents\s+(?:the\s+)?(.+?)\s+in\s+this\s+image\??\s*$",
                  question.strip(), flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip(" .?;,")

    """提取 yes/no 语义（鲁棒处理 A./B. 前缀、大小写、后续句子等）"""
    if not s:
        return None
    s = s.strip().lower()
    s = re.sub(r'^[a-z]\.\s*', '', s)  # 去掉 A. / B.
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    return None


def make_random_yesno_options(correct_sem: str):
    """
    correct_sem: 'yes' or 'no'
    返回 (options_list, answer_str)
    """
    sems = ["yes", "no"]
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
        options = ["A. yes", "B. no"]
        answer = "A. yes" if correct_sem == "yes" else "B. no"

    return options, answer



def transform_record(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # 仅处理 organ_location 且问题匹配“Which color represents ...”
    if obj.get("task") != "organ_location":
        return None

    Q = obj.get("question", "")
    target = extract_target_entity(Q)  # 这里得到 'stomach'


    options = obj.get("options")
    answer = obj.get("answer", "")
    if not isinstance(options, list) or len(options) < 2:
        return None

    correct_idx = find_correct_index(options, answer)
    if correct_idx is None:
        return None

    wrong_opt_full, wrong_idx = pick_random_wrong_option(options, correct_idx)
    if wrong_idx < 0:
        return None

    wrong_color = strip_label_prefix(wrong_opt_full)
    if not wrong_color:
        return None

    # 本条是“wrong_color-correct_region”，因此答案语义固定为 no
    yn_options, yn_answer = make_random_yesno_options(correct_sem="no")

    # 生成 yes/no 迷惑性题目，答案为 no
    new_obj = {
        "unique_id": obj.get("unique_id"),
        "image_paths": obj.get("image_paths"),
        "question": f"Is the {target} represented by {wrong_color} in this image?",
        "options": yn_options,   # ✅ A/B 随机
        "answer": yn_answer,     # ✅ 自动对应正确语义（no）
        # "orig_answer": answer,
        "modality": obj.get("modality"),
        "task": obj.get("task"),  # organ_location
        "type": "wrong_color-correct_region",
        "options_num": 2,
        # "orig_question": Q,
        # "picked_wrong_option": wrong_opt_full
    }
    return new_obj



def transform_record_right(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # 仅处理 organ_location 且问题匹配“Which color represents ...”
    if obj.get("task") != "organ_location":
        return None

    Q = obj.get("question", "")
    target = extract_target_entity(Q)  # 这里得到 'stomach'


    options = obj.get("options")
    answer = obj.get("answer", "")
    answer = re.sub(r'^\s*[A-Z]\.\s*', '', answer).strip()
    if not isinstance(options, list) or len(options) < 2:
        return None

    # 本条是“wrong_color-correct_region”，因此答案语义固定为 no
    yn_options, yn_answer = make_random_yesno_options(correct_sem="yes")

    # 生成 yes/no 迷惑性题目，答案为 yes
    new_obj = {
        "unique_id": obj.get("unique_id"),
        "image_paths": obj.get("image_paths"),
        "question": f"Is the {target} represented by the {answer} in this image?",
        "options": yn_options,
        "answer": yn_answer,
        # "orig_answer": answer,
        "modality": obj.get("modality"),
        "task": obj.get("task"),  # 仍然是 organ_location
        "type": "correct_color-correct_region",
        "options_num": 2,

        # "orig_question": Q,
        # "picked_wrong_option": wrong_opt_full

    }
    return new_obj


if __name__ == "__main__":
    json_Path = 'xxx/path'

    ''' 查看DrVD  json 文件内容'''
    # tasks = find_json(json_Path)

    '''筛选 出我要的 json文件'''
    my_json_path = 'xxx/path第一遍筛选.jsonl'

    temp_path = 'xxx/path'

    # organ_location 的制作过程
    '''挑选出来CT+MRI in organ_location 的数据 save in temp_path'''
    # find_region_location_jsonl(json_Path,temp_path)

    out_path = 'xxx/path'


    '''把筛选出来的颜色问题 转变成 open  question'''
    convert_jsonl_to_open(temp_path, out_path)

    '''在挑选结果基础上 把答案变的干扰 = yes/no回答'''
    processed = 0
    written = 0

    # 以追加模式打开输出，确保不覆盖
    with open(out_path, "a", encoding="utf-8") as fout:
        with open(temp_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                processed += 1
                '''错误的颜色 + 正确的问题 '''
                # new_obj = transform_record(obj)

                '''正确的颜色 + 正确的问题 '''
                new_obj = transform_record_right(obj)

                if new_obj is None:
                    continue

                fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
                written += 1

    print(f"Scanned: {processed}, Appended: {written}, Output: {out_path}")



