
'''

首先去validation_vqa_abnormality.csv 这里找有问题的id + 区域 加关键词“mm” 已保存在csv 读取
然后在visual_evidence_box_points.json 这里找对应的数据
对的框 + 图 找到
标注的尺寸和闻的尺寸 不一致 + 一致
最后做QA

'''

import re
from typing import List
import glob
import cv2
import nibabel as nib
from skimage.measure import label, regionprops

from matplotlib.patches import Rectangle
import numpy as np
import random
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import argparse
import json
import os
import sys
import tempfile
import pandas as pd
from typing import List, Dict, Optional

def process_data(record: Dict) -> Optional[Dict]:
    """
    按关键词返回：Volumename/Anatomy/Abnormality/Size
    - Anatomy 取 '/' 分割后的首段（如 'lung'）
    - Abnormality 为 'no findings' 时返回 None 以便上游丢弃
    """
    vol = str(record.get("Volumename", "")).strip()
    anatomy_raw = str(record.get("Anatomy", "")).strip()
    abnormality_raw = str(record.get("Abnormality", "")).strip()
    size_val = record.get("Size", "")

    if abnormality_raw.lower() == "no findings":
        return None

    anatomy_1 = anatomy_raw.split("/")[0].strip() if anatomy_raw else ""
    anatomy_part = anatomy_raw.split("/")[-1].strip() if anatomy_raw else ""

    return {
        "unique_id": vol,
        "asked_region": anatomy_part,
        "location_name" : anatomy_1,
        "abnormalities": abnormality_raw,
        "size": size_val,
    }

def save_to_csv(records, file_path):
    df = pd.DataFrame(records)
    df.to_csv(file_path, index=False, encoding='utf-8')

def read_jsonl_folder(csv_path: str) -> List[Dict]:
    """
    读取包含列 [Volumename, Anatomy, Abnormality, Size] 的 CSV，
    按 (Volumename, Anatomy首段) 去重，并返回处理后的记录列表。
    """
    records: List[Dict] = []
    df = pd.read_csv(csv_path)

    required_cols = ["Volumename", "Anatomy", "Abnormality", "Size"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")

    seen = set()  # 记录已经出现过的 (Volumename, Anatomy首段) 组合

    for _, row in df.iterrows():
        vol = str(row.get("Volumename", "")).strip()
        anatomy_raw = str(row.get("Anatomy", "")).strip()
        anatomy_part = anatomy_raw.split("/")[0].strip() if anatomy_raw else ""
        abnormality_raw = str(row.get("Abnormality", "")).strip()
        size_val = row.get("Size", "")

        # 跳过明确标注为 no findings 的行（大小写不敏感）
        if abnormality_raw.lower() == "no findings":
            continue

        key = (vol, anatomy_part)
        if key in seen:
            continue
        seen.add(key)

        # 将原始行转为 dict，保持关键词一致传入 process
        record = {
            "Volumename": vol,
            "Anatomy": anatomy_raw,       # 原始 anatomy 传进去，由 process 再取首段
            "Abnormality": abnormality_raw,
            "Size": size_val,
        }

        processed = process_data(record)
        if processed is not None:
            records.append(processed)

    return records

def draw_random_box(box_img_savepath, points, save_multi_box_path, no_box_img_savepath):
    """
        读取无框原图，根据 points 重新画一个与之前完全重合的红色矩形框，保存到新路径。
        参数
        ----
        box_img_savepath : 之前保存的“带框图”路径，仅用来参考尺寸（可选）
        points : 之前返回的 bbox_dict['bbox_points'] 格式
        save_multi_box_path : 新框图保存路径
        no_box_img_savepath : 无框原图路径（真正读取的图）
        """
    # 1. 读无框原图，保持 uint8 原分辨率
    img = cv2.imread(no_box_img_savepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"找不到无框原图: {no_box_img_savepath}")
    # 如果是单通道直接转 RGB，方便画彩色框
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    points = points["bbox_points"]
    # 2. 从 points 里取出四个角点，计算 min/max 行列
    tl = points["top_left"]  # [row, col]
    tr = points["top_right"]
    bl = points["bottom_left"]
    br = points["bottom_right"]

    minr = int(min(tl[0], tr[0], bl[0], br[0]))
    maxr = int(max(tl[0], tr[0], bl[0], br[0]))
    minc = int(min(tl[1], tr[1], bl[1], br[1]))
    maxc = int(max(tl[1], tr[1], bl[1], br[1]))

    # 3. 画框：红色，2 像素宽
    cv2.rectangle(img,
                  (minc, minr),  # 左上角 (x,y)
                  (maxc, maxr),  # 右下角
                  color=(255, 0, 0),
                  thickness=2)

    # 4. 保存
    os.makedirs(os.path.dirname(save_multi_box_path), exist_ok=True)
    # 再转回 BGR 保存，cv2 约定
    cv2.imwrite(save_multi_box_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"重新画框完成，已保存到：{save_multi_box_path}")





# 从 CSV 文件读取数据
def load_from_csv(file_path):
    return pd.read_csv(file_path)


def make_adversarial_size(size_str: str) -> str:
    """
    生成一个“错误”的 size，遵循：
      - 最大维度 <= 20 mm → 放大
      - 最大维度 > 20 mm → 缩小
    同时保持原格式（单值 vs A x B）、原单位（mm/cm）、每个维度的小数位数。
    """
    s = size_str.strip()
    m_unit = re.search(r'(mm|cm)\s*$', s, flags=re.IGNORECASE)
    if not m_unit:
        raise ValueError(f"unknown unit in size: {size_str}")
    unit = m_unit.group(1).lower()

    # 抓取所有数值（支持小数），以及每个数值的小数位数用于格式化回写
    num_strs = re.findall(r'\d+(?:\.\d+)?', s)
    if not num_strs:
        raise ValueError(f"no numeric dimension found in size: {size_str}")
    dims = [float(ns) for ns in num_strs]
    decs = [len(ns.split('.')[1]) if '.' in ns else 0 for ns in num_strs]

    # 转成 mm 便于比较/缩放
    to_mm = (lambda v: v * 10.0) if unit == 'cm' else (lambda v: v)
    from_mm = (lambda v: v / 10.0) if unit == 'cm' else (lambda v: v)
    dims_mm = [to_mm(v) for v in dims]

    max_mm = max(dims_mm)
    need_smaller = max_mm > 20.0

    # 选择缩放比例（让数值明显变化但仍显得合理）
    if need_smaller:
        # 大于 20 mm 的就缩小
        factor = 0.4 if max_mm >= 50 else 0.5
    else:
        # 小于等于 20 mm 的就放大
        factor = 3.0 if max_mm < 10 else 2.0

    new_dims_mm = [max(0.5, v * factor) for v in dims_mm]

    # 确保最大维度至少相差 1 mm
    diff_mm = abs(max(new_dims_mm) - max_mm)
    if diff_mm < 1.0:
        bump = (1.0 - diff_mm) + 0.1
        if need_smaller:
            new_dims_mm = [max(0.5, v - bump) for v in new_dims_mm]
        else:
            new_dims_mm = [v + bump for v in new_dims_mm]

    # 转回原单位
    new_dims = [from_mm(v) for v in new_dims_mm]

    # 按原每一维的小数位格式化；若四舍五入后不变，再最小步长调整一下
    out_nums: List[str] = []
    for v_new, v_old, d in zip(new_dims, dims, decs):
        rounded = round(v_new, d)
        # 如果四舍五入后恰好回到原值，做一步最小步长的偏移，保证是“错误答案”
        step = 10 ** (-d) if d > 0 else 1
        if abs(rounded - v_old) < step * 0.5:
            rounded = (rounded - step) if need_smaller else (rounded + step)

        if d > 0:
            out = f"{rounded:.{d}f}"
        else:
            out = str(int(round(rounded)))
        out_nums.append(out)

    # 维度之间用 'x'（与常见格式一致）
    numbers = "x".join(out_nums)
    return f"{numbers} {unit}"

def box_points_data(unique_id, sick_regions, asked_region, size, boximg_path, lesion):
    wrong_size = make_adversarial_size(size)


    qa = {
        "unique_id": unique_id,
        "region": asked_region,
        # "box_region": asked_region,
        "correct_size": size,
        "wrong_size": wrong_size,
        "image_paths": boximg_path ,
        # "sick_regions": sick_regions,
        "question": f"Is the {lesion} highlighted in red box at the {sick_regions} approximate measured at {wrong_size}? If not, what is the correct measurement?",
        "answer": f"No, the {lesion} at the {sick_regions} measures approximately {size}, not {wrong_size}. " ,

        "modality": "CT",
        "task": "lesion_size",
        "type": "wrong_size_wrong_Q",
        "question_type": "open_question",
        # "lesion": lesion,

    }


    return qa

def box_points_data1(unique_id, sick_regions, asked_region, size, boximg_path, lesion):
    # wrong_size = make_adversarial_size(size)


    qa = {
        "unique_id": unique_id,
        "correct_size": size,
        # "wrong_size": wrong_size,
        "image_paths": boximg_path ,
        "region": asked_region,
        "box_region": asked_region,
        "question": f"Is the {lesion} highlighted in red box at the {sick_regions} approximate measured at approximate {size}? If not, what is the correct measurement?",
        "answer": f"Yes, The lesion in the red area is approximately {size} in size." ,

        "modality": "CT",
        "task": "lesion_size",
        "type": "correct_size_correct_Q",
        "question_type": "open_question",
        # "lesion": lesion,
        # "sick_regions": sick_regions,

    }


    return qa














if __name__ == "__main__":
    box_points_path = "xxx/path"  # 你的 region 标注 JSON
    vqa_abnormality_path = 'xxx/path'
    out_qas = "xxx/path"
    region_box = 'xxx/path'

    ct_root = "xxx/path"
    mask_root = "xxx/path"

    # # 先读取所有的不正常区域
    # records = read_jsonl_folder(vqa_abnormality_path)
    # 保存记录到指定文件
    # save_to_csv(records, 'xxx/path')

    loaded_records_csv = load_from_csv('xxx/path')
    # 将 DataFrame 转换为字典列表
    records_list = loaded_records_csv.to_dict(orient='records')
    # 找出所有疾病
    # abnormalities_list = [record['abnormalities'] for record in records_list]
    # 去重
    # abnormalities_list_unique = list(set(abnormalities_list))

    out_qas = "xxx/path"

    emitted_ids = set()
    mode = "a" if os.path.exists(out_qas) else "w"  # 追加到文件末尾 # 已经写出的 unique_id
    total = 0
    built = 0
    Path_box = "xxx/path"
    Path_no_box = "xxx/path"
    # 找到对应的box img
    with open(out_qas, "a", encoding="utf-8") as fout:
        for record in records_list:
            unique_id = record['unique_id']
            unique_id_name = unique_id.split(".")[0]
            sick_regions = record['asked_region'] # 这里是最细的位置 左上肺
            asked_region = record['location_name'] # 这里是框的部位 肺
            lesion = record['abnormalities'] # 病变名称
            size = record['size'] # 病变尺寸


            # if asked_region == "abdomen" or asked_region == "pleura":
            #     continue


            box_path_global = os.path.join(Path_box, f"box_{unique_id_name}", f"{asked_region}_*")  # 使用 * 来匹配 xxx 部分
            # 找到画框的图
            boximg_path = glob.glob(box_path_global)
            if not boximg_path:
                # 可选：打印一下方便排查
                # print(f"[warn] no match: {box_path_global}")
                continue

            boximg_path = boximg_path[0]  # 或者 sorted(matches)[0] 确保确定性




            '''正确的size + 正确的q'''
            # qa = box_points_data(unique_id, sick_regions, asked_region, size, boximg_path, lesion)
            '''错误的size + 错误的q'''
            qa = box_points_data1(unique_id, sick_regions, asked_region, size, boximg_path, lesion)


            if qa is None:
                continue

            fout.write(json.dumps(qa, ensure_ascii=False) + "\n")
            built += 1
    print(f" z输入记录: {total}, 生成对抗QA(每个id一次): {built}, 输出: {out_qas}")
