'''
设计病灶级别的 QA 的对抗样本
1// 错的框 + 正确的问题
2// 正确的框 + 错误的问题

'''

from typing import Tuple, List
import json
import os
import re
from typing import Iterable, Dict, Any, List, Optional
import random
# 英文 region 到中文名的简单映射，可按需要增补
# REGION_CN = {
#     "lung": "肺",
#     "heart": "心脏",
#     "mediastinum": "纵隔",
#     "pleura": "胸膜",
#     "spine": "脊柱",
#     "aorta": "主动脉",
#     "esophagus": "食管",
#     "liver": "肝脏",
#     "kidney": "肾脏",
# }
REGION_EN_DISPLAY = {
    "lung": "the lungs",
    "heart": "the heart",
    "mediastinum": "the mediastinum",
    "pleura": "the pleura",
    "spine": "the spine",
    "aorta": "the aorta",
    "esophagus": "the esophagus",
    "liver": "the liver",
    "kidney": "the kidney",
}

def parse_region_from_filename(path: str) -> str:
    """从 xxx/region_box/box_xxx/<region>_ID<number>.png 提取 <region>"""
    name = os.path.basename(path)
    # 优先按 '_ID' 分割，若没有则去掉扩展名后按最后一次 '_' 尝试
    if "_ID" in name:
        return name.split("_ID", 1)[0]
    stem, _ = os.path.splitext(name)
    parts = stem.rsplit("_", 1)
    return parts[0] if parts else stem

def iter_input_records(in_path: str) -> Iterable[Dict[str, Any]]:
    """兼容 JSON 数组文件或 JSONL 文件"""
    with open(in_path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":  # JSON 数组
            data = json.load(f)
            if isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict):
                        yield obj
            return
        # JSONL
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj

SKIP_REGIONS = {"abdomen", "pleura"}

def parse_region_from_filename(path: str) -> str:
    name = os.path.basename(path)
    if "_ID" in name:
        return name.split("_ID", 1)[0]
    stem, _ = os.path.splitext(name)
    parts = stem.rsplit("_", 1)
    return parts[0] if parts else stem

def pick_other_region_image(folder: str, asked_region: str) -> Tuple[str, str]:
    """
    在 folder 中随机选取一个 region != asked_region 且不在 SKIP_REGIONS 的 PNG。
    返回 (candidate_path, candidate_region)。找不到返回 ("", "")。
    """
    if not os.path.isdir(folder):
        return "", ""

    asked_norm = (asked_region or "").strip().lower()
    skip_norm = {r.strip().lower() for r in SKIP_REGIONS}

    candidates: List[Tuple[str, str]] = []
    for name in os.listdir(folder):
        if not name.lower().endswith(".png"):
            continue
        region = parse_region_from_filename(name)  # 原始大小写
        region_norm = (region or "").strip().lower()
        if not region_norm:
            continue
        if region_norm == asked_norm:
            continue
        if region_norm in skip_norm:
            continue
        path = os.path.join(folder, name)
        candidates.append((path, region))

    if not candidates:
        return "", ""

    candidate_path, candidate_region = random.choice(candidates)
    return candidate_path, candidate_region

def yn_semantic(s: str) -> Optional[str]:
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

def build_adv_qa(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    box_path = record.get("BoxImg_savepath") or record.get("box_path") or record.get("image_path")
    asked_region = record.get("region")
    unique_id = record.get("unique_id")
    if not box_path or not asked_region:
        return None
    if asked_region == "abdomen" or asked_region =="pleura" :
        return None
    folder = os.path.dirname(box_path)
    '''这里来找错误的框的img'''
    other_img, box_region = pick_other_region_image(folder, asked_region)
    if not other_img:
        return None

    # 使用英文展示名；缺省回退为原始 region
    asked_en = REGION_EN_DISPLAY.get(asked_region, asked_region)
    yn_options, yn_answer = make_random_yesno_options(correct_sem="no")
    qa = {
    "unique_id": unique_id,
    "image_paths": other_img,
    "asked_region": asked_region,
    "box_region": box_region,  # 替换进去的关键词 region
    "question": f"Examine the provided image carefully, particularly the red box around the {asked_region}. Is there a lesion within this red-marked section?",
    # "question": f"Is the lesion located within the red box area of the {asked_region}?",
    "options": yn_options,  # 保持你现有格式
    "answer": yn_answer,

    # "answer": f"No, there are no {asked_region}  lesions. The red box contains the {box_region} ",
    "modality": "CT",
    "task": "bodypart_recognition",
    "type": "wrong_box_correct_Q",
    # "question_type": "open_question",
    "options_num": 2,
}
    return qa

def build_YBYQ_qa(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    box_path = record.get("BoxImg_savepath")
    asked_region = record.get("region")
    unique_id = record.get("unique_id")
    if not box_path or not asked_region:
        return None
    if asked_region == "abdomen" or asked_region =="pleura" :
        return None
    # folder = os.path.dirname(box_path)
    '''这里来找错误的框的img'''
    # other_img, box_region = pick_other_region_image(folder, asked_region)
    # if not other_img:
    #     return None

    # 使用英文展示名；缺省回退为原始 region
    asked_en = REGION_EN_DISPLAY.get(asked_region, asked_region)
    yn_options, yn_answer = make_random_yesno_options(correct_sem="yes")
    qa = {
        "unique_id": unique_id,
        "image_paths": box_path,
        "asked_region": asked_region,
        "box_region": asked_region,  # 替换进去的关键词 region
        "question": f"Examine the provided image carefully, particularly the red box around the {asked_region}. Is there a lesion within this red-marked section?",
        # "question": f"Is there a lesion in the {asked_region} within this red box?",
        "options":yn_options,
        "answer": yn_answer,
        # "answer": f"Yes, Yes, the red box contains {asked_region} lesion. ",
        "modality": "CT",
        "task": "bodypart_recognition",
        "type": "correct_box_correct_Q",
        # "question_type": "open_question",
        "options_num": 2,
    }



    return qa


def build_cbox_wq_qa(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    box_path = record.get("BoxImg_savepath") or record.get("box_path") or record.get("image_path")
    asked_region = record.get("region")
    unique_id = record.get("unique_id")
    if not box_path or not asked_region:
        return None
    if asked_region == "abdomen" or asked_region =="pleura" :
        return None
    folder = os.path.dirname(box_path)
    '''这里来找错误的框的img'''
    other_img, box_region = pick_other_region_image(folder, asked_region)
    if not other_img:
        return None

    # 使用英文展示名；缺省回退为原始 region
    asked_en = REGION_EN_DISPLAY.get(asked_region, asked_region)
    yn_options, yn_answer = make_random_yesno_options(correct_sem="no")
    qa = {
    "unique_id": unique_id,
    "image_paths": box_path,
    "asked_region": asked_region,
    "box_region": box_region,  # 替换进去的关键词 region
    "question": f"Examine the provided image carefully, particularly the red box around the {box_region}. Is there a lesion within this red-marked section?",
    # "question": f"Is there a lesion in the {box_region} within this red box?",
    "options": yn_options,  # 保持你现有格式
    "answer": yn_answer,

    # "answer": f"No, there are no {asked_region}  lesions. The red box contains the {box_region} ",
    "modality": "CT",
    "task": "bodypart_recognition",
    "type": "correct_box_wrong_Q",
    # "question_type": "open_question",
    "options_num": 2,
}
    return qa


if __name__ == "__main__":
    in_meta = "xxx/path"  # 你的 region 标注 JSON
    out_qas = "xxx/path"

    emitted_ids = set()
    mode = "a" if os.path.exists(out_qas) else "w"  # 追加到文件末尾 # 已经写出的 unique_id
    total = 0
    built = 0
    with open(out_qas, "a", encoding="utf-8") as fout:
        for rec in iter_input_records(in_meta):
            total += 1
            uid = rec.get("unique_id")
            if not uid:
                continue
            # 已经为该 id 写过一条，就跳过
            # if uid in emitted_ids:
            #     continue

            ''' "type": "wrong_box-correct_Q"'''
            # qa = build_adv_qa(rec)

            ''' "type": "correct_box-correct_Q"'''
            # qa = build_YBYQ_qa(rec)

            ''' "type": "correct_box_wrong_Q" '''
            qa = build_cbox_wq_qa(rec)
            if qa is None:
                continue



            fout.write(json.dumps(qa, ensure_ascii=False) + "\n")
            built += 1
            emitted_ids.add(uid)
    print(f"输入记录: {total}, 生成对抗QA(每个id一次): {built}, 输出: {out_qas}")
