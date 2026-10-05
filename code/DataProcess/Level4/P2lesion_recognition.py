'''

首先去validation_vqa_abnormality.csv 这里找有问题的id + 区域
然后在visual_evidence_box_points.json 这里找对应的数据
最后做QA

'''
import glob
import os
from PIL import Image, ImageDraw
import nibabel as nib
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


def dedupe_jsonl_file(in_path, out_path=None, normalize=False, encoding="utf-8"):
    """
    Deduplicate a JSONL file, keeping the first occurrence.
    - in_path: path to input .jsonl
    - out_path: path to write; if None, deduplicate in-place
    - normalize: if True, dedupe by JSON content (sorted keys, no spaces)
    - encoding: file encoding
    Returns a dict with stats.
    """
    import os, json, tempfile

    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"input not found: {in_path}")

    if out_path is None:
        out_path = in_path

    seen = set()
    total = kept = removed = parse_errors = 0

    tmp_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    with open(in_path, "r", encoding=encoding) as fin, \
         tempfile.NamedTemporaryFile("w", delete=False, dir=tmp_dir, encoding=encoding) as fout:

        tmp_path = fout.name
        for line in fin:
            total += 1
            key = line
            if normalize:
                try:
                    obj = json.loads(line)
                    # Canonical form: sorted keys, no extra spaces
                    key = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                except Exception:
                    parse_errors += 1
                    key = line  # fallback to raw line key

            if key in seen:
                removed += 1
                continue

            seen.add(key)
            fout.write(line)
            kept += 1

    # Atomic replace/move so the output is safely written
    os.replace(tmp_path, out_path)

    return {
        "input": in_path,
        "output": out_path,
        "total": total,
        "kept": kept,
        "removed": removed,
        "normalize": normalize,
        "parse_errors": parse_errors,
    }


def draw_random_box(image_path, points, save_path, no_box_img_savepath):
    # 使用 matplotlib 读取图像
    img = mpimg.imread(no_box_img_savepath)

    # 提取四个顶点坐标
    bbox_points = points['bbox_points']
    top_left = tuple(bbox_points['top_left'])
    top_right = tuple(bbox_points['top_right'])
    bottom_left = tuple(bbox_points['bottom_left'])
    bottom_right = tuple(bbox_points['bottom_right'])

    # 计算原始框的宽度和高度
    box_width = top_right[1] - top_left[1]
    box_height = bottom_left[0] - top_left[0]

    # 随机生成一个不与原框重叠的框
    img_width, img_height = img.shape[1], img.shape[0]  # 获取图像宽度和高度
    while True:
        # 随机生成一个框的位置
        rand_x = random.randint(0, img_width - box_width)
        rand_y = random.randint(0, img_height - box_height)

        # 检查是否与原框重叠，避免重叠
        if (rand_x > top_right[0] or rand_x + box_width < top_left[0] or
                rand_y > bottom_left[1] or rand_y + box_height < top_left[1]):
            break

    # 计算新的框的四个点
    rand_top_left = [rand_y, rand_x]
    rand_top_right = [rand_y, rand_x + box_width]
    rand_bottom_left = [rand_y + box_height, rand_x]
    rand_bottom_right = [rand_y + box_height, rand_x + box_width]

    # 格式化返回的字典
    random_box_points = {
        "top_left": [int(rand_top_left[0]), int(rand_top_left[1])],
        "top_right": [int(rand_top_right[0]), int(rand_top_right[1])],
        "bottom_left": [int(rand_bottom_left[0]), int(rand_bottom_left[1])],
        "bottom_right": [int(rand_bottom_right[0]), int(rand_bottom_right[1])]
    }

    # 绘制图像和框
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    ax = plt.gca()

    # # 绘制原始框（红色）
    # rect1 = patches.Rectangle(top_left, box_width, box_height, linewidth=2, edgecolor='r', facecolor='none')
    # ax.add_patch(rect1)

    # 绘制随机生成的框（绿色）
    rect2 = patches.Rectangle((rand_x, rand_y), box_width, box_height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect2)

    plt.axis("off")  # 去掉坐标轴
    plt.title('')  # 不显示标题

    # 保存图像到指定路径

    # plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)  # 保存紧凑的图像，去掉多余白边

    print(f"Saved image with random box to {save_path}")

    return random_box_points, save_path # 返回生成的随机框坐标



def process_data(record: dict) -> dict:
    # 获取路径和文件名
    file_name = record.get('Volumename', "")

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
        "abnormalities": abnormality
    }

def read_jsonl_folder(folder_path: str):
    records = []  # 用来保存所有的处理后的记录
    df = pd.read_csv(folder_path)

    seen = set()  # 记录已经出现过的 (Volumename, Anatomy) 组合

    for index, row in df.iterrows():
        # 提取 Anatomy 列中的第一个词（以 / 分割）
        anatomy_part = row["Anatomy"].split("/")[0]

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


# 保存到 CSV 文件
def save_to_csv(records, file_path):
    df = pd.DataFrame(records)
    df.to_csv(file_path, index=False, encoding='utf-8')

# 从 CSV 文件读取数据
def load_from_csv(file_path):
    return pd.read_csv(file_path)


import json
from typing import Iterable, Dict, Any, List, Optional
import random
import re


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


def box_points_data(box_points_path, boximg_path, lesion):
    # 读取 JSON 文件
    with open(box_points_path, 'r') as f:
        data = json.load(f)

    # 遍历每个记录，查找 BoxImg_savepath 匹配的条目
    for record in data:
        box_img_savepath = record.get('BoxImg_savepath')

        # 如果 BoxImg_savepath 匹配
        if box_img_savepath == boximg_path:
            # 获取其他字段
            unique_id = record.get('unique_id')
            region = record.get('region')
            biggest_slice = record.get('biggest_slice')
            no_box_img_savepath = record.get('NoBoxImg_savepath')
            points = record.get('Points')

            # 画错的框
            unique_id_file = unique_id.split(".")[0]
            bbox_path  = 'xxx/path'+unique_id_file+'/'
            if not os.path.exists(bbox_path):
                os.makedirs(bbox_path)
            save_wrong_box_path = bbox_path + region+'.png'
            # save path 改变
            '''画错误的框 + save'''
            random_box_points, save_path = draw_random_box(box_img_savepath, points, save_wrong_box_path, no_box_img_savepath)
            templates = [
                "Examine the provided image carefully, particularly the red box. Is the red-boxed area indicative of {lesion}?",
                "Examine the provided image carefully, particularly the red box. Does the finding in the red box point to {lesion} on this scan?",
                "Examine the provided image carefully, particularly the red box. Is the red-boxed abnormality consistent with {lesion}?",
                "Examine the provided image carefully, particularly the red box. Does the red-highlighted region show evidence of {lesion}?",
                "Examine the provided image carefully, particularly the red box. Is there any {lesion} within the red-boxed region on this scan?",
                "Examine the provided image carefully, particularly the red box. Are the imaging features in the red box suggestive of {lesion}?",
                "Examine the provided image carefully, particularly the red box. Could the red-boxed finding represent {lesion}?",
                "Examine the provided image carefully, particularly the red box. Do the red-boxed findings raise concern for {lesion}?",
                "Examine the provided image carefully, particularly the red box. Is the red-boxed area suspicious for {lesion} in this scan?",
                "Examine the provided image carefully, particularly the red box. Does the red-boxed lesion appear compatible with {lesion}?"
            ]

            # lesion should be defined, e.g., lesion = "pulmonary embolism"
            question = random.choice(templates).format(lesion=lesion)
            yn_options, yn_answer = make_random_yesno_options(correct_sem="no")
            qa =  {
                "unique_id": unique_id,
                "image_paths": save_path,
                "region": region,
                "question": question,
                "options": yn_options,
                "answer": yn_answer,
                "task": "lesion_recognition",
                # "box_region": 'random_wrong',
                "modality": "CT",
                "type": "wrong_box_correct_Q",
                "options_num": 2,
                "lesion": lesion,
                # "biggest_slice": biggest_slice,
                # "BoxImg_savepath": box_img_savepath,
                # "NoBoxImg_savepath": no_box_img_savepath,
                # "Wrong_Points": random_box_points,
                # "Points": points,
            }

    return qa



def box_points_data_2(box_points_path, boximg_path, lesion):
    # 读取 JSON 文件
    with open(box_points_path, 'r') as f:
        data = json.load(f)

    # 遍历每个记录，查找 BoxImg_savepath 匹配的条目
    for record in data:
        box_img_savepath = record.get('BoxImg_savepath')

        # 如果 BoxImg_savepath 匹配
        if box_img_savepath == boximg_path:
            # 获取其他字段
            unique_id = record.get('unique_id')
            region = record.get('region')
            biggest_slice = record.get('biggest_slice')
            no_box_img_savepath = record.get('NoBoxImg_savepath')
            points = record.get('Points')

            # # 画错的框
            # bbox_path  = 'xxx/path'+unique_id+'/'
            # if not os.path.exists(bbox_path):
            #     os.makedirs(bbox_path)
            # save_wrong_box_path = bbox_path + region+'.png'
            # # save path 改变
            # '''画错误的框 + save'''
            # save_path = draw_random_box(box_img_savepath, points, save_wrong_box_path, no_box_img_savepath)
            templates = [
                "Examine the provided image carefully, particularly the red box. Is the red-boxed area indicative of {lesion}?",
                "Examine the provided image carefully, particularly the red box. Does the finding in the red box point to {lesion} on this scan?",
                "Examine the provided image carefully, particularly the red box. Is the red-boxed abnormality consistent with {lesion}?",
                "Examine the provided image carefully, particularly the red box. Does the red-highlighted region show evidence of {lesion}?",
                "Examine the provided image carefully, particularly the red box. Is there any {lesion} within the red-boxed region on this scan?",
                "Examine the provided image carefully, particularly the red box. Are the imaging features in the red box suggestive of {lesion}?",
                "Examine the provided image carefully, particularly the red box. Could the red-boxed finding represent {lesion}?",
                "Examine the provided image carefully, particularly the red box. Do the red-boxed findings raise concern for {lesion}?",
                "Examine the provided image carefully, particularly the red box. Is the red-boxed area suspicious for {lesion} in this scan?",
                "Examine the provided image carefully, particularly the red box. Does the red-boxed lesion appear compatible with {lesion}?"
            ]
            # lesion should be defined, e.g., lesion = "pulmonary embolism"
            question = random.choice(templates).format(lesion=lesion)
            yn_options, yn_answer = make_random_yesno_options(correct_sem="yes")
            qa =  {
                "unique_id": unique_id,
                "image_paths": box_img_savepath,
                "region": region,
                # "box_region": region ,
                # "wrong_box_save_path":'no',
                # "biggest_slice": biggest_slice,
                # "NoBoxImg_savepath": no_box_img_savepath,
                # "Points": points,
                "question": question,
                "options": yn_options,
                "answer": yn_answer,
                "task": "lesion_recognition",
                "modality": "CT",
                "type": "correct_box_correct_Q",
                "options_num": 2,
                "lesion": lesion,
            }


    return qa





def box_points_data3(box_points_path, boximg_path, lesion, lesion_list):
    wrong_lesion = []
    # 从 lesion_list 中选择不等于 lesion 的随机元素
    # 去除与当前 lesion 相同的所有项
    filtered_lesions = [item for item in lesion_list if item != lesion]
    # 确保 filtered_lesions 中有足够的元素供随机选择
    if filtered_lesions:
        wrong_lesion = random.choice(filtered_lesions)  # 随机选择一个不等于 lesion 的元素


    # 读取 JSON 文件
    with open(box_points_path, 'r') as f:
        data = json.load(f)

    # 遍历每个记录，查找 BoxImg_savepath 匹配的条目
    for record in data:
        box_img_savepath = record.get('BoxImg_savepath')

        # 如果 BoxImg_savepath 匹配
        if box_img_savepath == boximg_path:
            # 获取其他字段
            unique_id = record.get('unique_id')
            region = record.get('region')
            biggest_slice = record.get('biggest_slice')
            no_box_img_savepath = record.get('NoBoxImg_savepath')
            points = record.get('Points')

            # 画错的框
            unique_id_file = unique_id.split(".")[0]
            bbox_path  = 'xxx/path'+unique_id_file+'/'
            if not os.path.exists(bbox_path):
                os.makedirs(bbox_path)
            # save_wrong_box_path = bbox_path + region+'.png'
            # save path 改变
            '''画错误的框 + save'''
            # random_box_points, save_path = draw_random_box(box_img_savepath, points, save_wrong_box_path, no_box_img_savepath)
            templates = [
                "Examine the provided image carefully, particularly the red box. Is the red-boxed area indicative of {wrong_lesion}?",
                "Examine the provided image carefully, particularly the red box. Does the finding in the red box point to {wrong_lesion} on this scan?",
                "Examine the provided image carefully, particularly the red box. Is the red-boxed abnormality consistent with {wrong_lesion}?",
                "Examine the provided image carefully, particularly the red box. Does the red-highlighted region show evidence of {wrong_lesion}?",
                "Examine the provided image carefully, particularly the red box. Is there any {wrong_lesion} within the red-boxed region on this scan?",
                "Examine the provided image carefully, particularly the red box. Are the imaging features in the red box suggestive of {wrong_lesion}?",
                "Examine the provided image carefully, particularly the red box. Could the red-boxed finding represent {wrong_lesion}?",
                "Examine the provided image carefully, particularly the red box. Do the red-boxed findings raise concern for {wrong_lesion}?",
                "Examine the provided image carefully, particularly the red box. Is the red-boxed area suspicious for {wrong_lesion} in this scan?",
                "Examine the provided image carefully, particularly the red box. Does the red-boxed lesion appear compatible with {wrong_lesion}?",
            ]

            # lesion should be defined, e.g., lesion = "pulmonary embolism"
            question = random.choice(templates).format(wrong_lesion=wrong_lesion)
            yn_options, yn_answer = make_random_yesno_options(correct_sem="no")
            qa =  {
                "unique_id": unique_id,
                "image_paths": box_img_savepath,
                "region": region,
                "question": question,
                "options": yn_options,
                "answer": yn_answer,
                "task": "lesion_recognition",
                # "box_region": 'random_wrong',
                "modality": "CT",
                "type": "correct_box_wrong_Q",
                "options_num": 2,
                "lesion": lesion,
                "wrong_lesion": wrong_lesion,

            }

    return qa



if __name__ == "__main__":
    box_points_path = "xxx/path"  # 你的 region 标注 JSON
    vqa_abnormality_path = 'xxx/path'
    out_qas = "xxx/path"

    # 先读取所有的不正常区域
    records = read_jsonl_folder(vqa_abnormality_path)
    # 保存记录到指定文件
    save_to_csv(records, 'xxx/path')

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
    Path = "xxx/path"
    Path_no_box = "xxx/path"

    lesion_list = []  # 初始化空列表，用于存储所有的 lesions

    # 遍历每条记录并提取 abnormalities（lesion）字段
    for record in records_list:
        lesion = record.get('abnormalities', None)  # 获取 'abnormalities' 字段，若没有则为 None
        # 如果 abnormalities 不存在或为空，则跳过当前记录
        if not lesion:
            continue
        words = lesion.split()
        # 如果单词数量超过 5，则跳过当前记录
        if len(words) > 5:
            continue
            continue
            # 如果 abnormalities 存在，则加入 lesion_list
        lesion_list.append(lesion)



    # 找到对应的box img
    with open(out_qas, "a", encoding="utf-8") as fout:
        for record in records_list:
            unique_id = record['unique_id']
            unique_id = unique_id.split(".")[0]
            asked_region = record['asked_region']

            if asked_region == "abdomen" or asked_region == "pleura":
                continue
            lesion = record['abnormalities']

            box_path_global = os.path.join(Path, f"box_{unique_id}", f"{asked_region}_*")  # 使用 * 来匹配 xxx 部分
            # 使用 glob 查找所有符合模式的文件
            boximg_path = glob.glob(box_path_global)
            if not boximg_path:
                # 可选：打印一下方便排查
                # print(f"[warn] no match: {box_path_global}")
                continue

            boximg_path = boximg_path[0]  # 或者 sorted(matches)[0] 确保确定性

            # 找到这个img + 画图
            '''wrong box + correct q'''
            # qa = box_points_data(box_points_path, boximg_path, lesion)

            '''correct box + correct q'''
            # qa = box_points_data_2(box_points_path, boximg_path, lesion)

            '''correct box + wrong q'''
            qa = box_points_data3(box_points_path, boximg_path, lesion, lesion_list)


            if qa is None:
                continue

            fout.write(json.dumps(qa, ensure_ascii=False) + "\n")
            built += 1
    print(f"输入记录: {total}, 生成对抗QA(每个id一次): {built}, 输出: {out_qas}")






