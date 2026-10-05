
'''

首先去validation_vqa_abnormality.csv 这里找有问题的id + 区域 已保存在csv 读取
然后在visual_evidence_box_points.json 这里找对应的数据
同一个 ID 选三个框 分别是 红黄蓝 然后有一个是正确答案 open-question
最后做QA

'''
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

def box_points_data(bbox_dict, box_path, lesion, unique_id, asked_region, wrong_box_color):
    slice_index = bbox_dict['slice_index']
    right_box_color = bbox_dict['best_box_color']
    points = bbox_dict['bbox_points']

    templates = [
        "In the scan, there are three color-coded boxes: red, yellow, and blue. Each box represents a different area of the scan. Please carefully examine the scan and determine which of these color-coded boxes contains the {lesion}.",
        "There are three color-coded boxes in the scan: red, yellow, and blue. Each box highlights a specific area. Which of these boxes contains the {lesion}?",
        "The scan shows three color-coded boxes: red, yellow, and blue, each marking a separate region. Please analyze the scan and determine which box holds the {lesion}.",
        "In the provided scan, you see three color-coded boxes: red, yellow, and blue. Each box covers a different section. Which of these sections contains the {lesion}?",
        "In this scan, there are three color-coded boxes: red, yellow, and blue. Each marks a unique area. Could you identify the box that contains the {lesion}?",
        "The scan has three distinct color-coded boxes: red, yellow, and blue. Each one highlights a different part of the scan. Which box contains the {lesion}?",
        "In the scan, there are three color-coded boxes: red, yellow, and blue. Each box represents a different area of the scan. Please carefully examine the scan and determine which of these color-coded boxes contains the {lesion}.",
    ]

    # lesion/location are your variables
    question = random.choice(templates).format(lesion=lesion)

    color_options = ["red", "yellow", "blue"]
    # 随机打乱颜色顺序
    random.shuffle(color_options)
    # 将颜色与 A, B, C 映射
    options = {chr(65 + i): color for i, color in enumerate(color_options)}  # 将颜色映射为 A, B, C
    # 找到旧的 answer 对应的选项字母
    # 找到旧的 answer 对应的选项字母，并返回字母与颜色的组合
    correct_option = f"{[k for k, v in options.items() if v == right_box_color][0]}. {right_box_color}"

    qa = {
        "unique_id": unique_id,
        "region": asked_region,
        "box_region": asked_region,
        "right_box_color": right_box_color,
        # "wrong_box_color": wrong_box_color,
        "image_paths":box_path ,
        # "Points": points,
        "question": question,
        "options": options,
        "answer": correct_option ,
        "task": "lesion_multi_box",
        "modality": "CT",
        "question_type": "3options",
        "lesion": lesion,

    }


    return qa










def find_file(root, filename):
    for dirpath, dirnames, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None

def find_nii(ct_root, file_name):
    # 根据文件名构建对应的路径
    subfolder1 = f"valid_{file_name.split('_')[1]}"  # valid_1、valid_2 等
    subfolder2 = f"valid_{file_name.split('_')[1]}a"  # valid_1a、valid_2a 等
    niiPath = os.path.join(ct_root, subfolder1, subfolder2, file_name)
    # 检查文件是否存在并加载
    if os.path.exists(niiPath):
        return niiPath
    else:
        print("not find nii", niiPath)



def find_mask(mask_root,ct_name,anatomy):

    print(anatomy)
    base = os.path.splitext(os.path.splitext(ct_name)[0])[0]
    # 比如 valid_861_a_1.nii.gz -> valid_861_a_1
    # 拼 seg_ 前缀
    path = 'seg_' + base          # seg_valid_861_a_1
    # 再拼 anatomy 文件名
    mask_path = os.path.join(mask_root, path, anatomy + '.nii.gz')

    if os.path.exists(mask_path):
        return mask_path
    else:
        print("not find nii", mask_path)


def show_middle_slice_with_bbox_global(ct_data, mask_data, Volumename, anatomy):
    """
    1. 在整个 3D mask 里（所有 z）遍历，找出面积最大的连通区域（全局最大 region）
    2. 用这个 region 的 bbox，在“中间那一层 CT slice”上画红色加粗矩形
    3. 打印这个最大框的四个坐标和它原本所在的 z
    """
    H, W, D = mask_data.shape

    best_area = -1
    best_box = None  # (minr, minc, maxr, maxc)
    best_z = None

    # ===== ① 全局遍历 z：找面积最大的连通域 =====
    for z in range(D):
        mask_slice = mask_data[:, :, z] > 0
        if not mask_slice.any():
            continue

        labeled = label(mask_slice)
        regions = regionprops(labeled)
        if not regions:
            continue

        # 当前 z 上最大的连通域
        largest_region = max(regions, key=lambda r: r.area)
        minr, minc, maxr, maxc = largest_region.bbox
        area = (maxr - minr) * (maxc - minc)

        if area > best_area:
            best_area = area
            best_box = (minr, minc, maxr, maxc)
            best_z = z

    # 整个 3D 都没有前景
    if best_box is None:
        print("整个 3D mask 里没有任何前景。")
        return None

    minr, minc, maxr, maxc = best_box

    # ===== ② 准备路径与切片数据 =====
    ct_slice = ct_data[:, :, best_z]
    Volumename_root = Volumename.split('.')[0]

    base_dir = "xxx/path"
    folder_path = os.path.join(base_dir, Volumename_root)
    os.makedirs(folder_path, exist_ok=True)


    # ===== ③ 工具函数：框重叠检测与随机采样不重叠框 =====
    def boxes_overlap(b1, b2):
        """判断两个框是否严格有面积重叠（边贴边不算重叠）"""
        a_minr, a_minc, a_maxr, a_maxc = b1
        b_minr, b_minc, b_maxr, b_maxc = b2
        return not (a_maxr <= b_minr or b_maxr <= a_minr or
                    a_maxc <= b_minc or b_maxc <= a_minc)

    def sample_non_overlapping_box(
            H, W, avoid,
            min_h_ratio=0.08, max_h_ratio=0.35,
            min_w_ratio=0.08, max_w_ratio=0.35,
            max_tries=600
    ):
        """
        在图像内随机采样一个框：
        - 尺寸在给定比例范围内
        - 不与 avoid 列表中的任何框重叠
        - 不越界
        采样失败时返回 None
        """
        min_h = max(3, int(min_h_ratio * H))
        max_h = max(min_h + 1, int(max_h_ratio * H))
        min_w = max(3, int(min_w_ratio * W))
        max_w = max(min_w + 1, int(max_w_ratio * W))

        for _ in range(max_tries):
            h = np.random.randint(min_h, max_h + 1)
            w = np.random.randint(min_w, max_w + 1)
            if h >= H or w >= W:
                continue
            r0 = np.random.randint(0, H - h + 1)
            c0 = np.random.randint(0, W - w + 1)
            cand = (r0, c0, r0 + h, c0 + w)
            if all(not boxes_overlap(cand, other) for other in avoid):
                return cand
        return None

    # ===== ④ 生成两个不重叠、且尺寸不同的随机框（均不越界，且不与 best_box 重叠） =====
    avoid = [best_box]
    extra1 = sample_non_overlapping_box(H, W, avoid)
    if extra1 is not None:
        avoid.append(extra1)

    extra2 = None
    if extra1 is not None:
        h1 = extra1[2] - extra1[0]
        w1 = extra1[3] - extra1[1]
        # 优先尝试采样与 extra1 尺寸不同的第二个框
        for _ in range(500):
            cand = sample_non_overlapping_box(H, W, avoid)
            if cand is None:
                break
            h2 = cand[2] - cand[0]
            w2 = cand[3] - cand[1]
            if (h2, w2) != (h1, w1):
                extra2 = cand
                avoid.append(cand)
                break
        # 若未成功，则放宽为任意不重叠
        if extra2 is None:
            extra2 = sample_non_overlapping_box(H, W, [best_box, extra1])

    # 如果第一个额外框都失败，尝试缩小尺寸范围再次采样
    if extra1 is None:
        extra1 = sample_non_overlapping_box(
            H, W, [best_box],
            min_h_ratio=0.05, max_h_ratio=0.20,
            min_w_ratio=0.05, max_w_ratio=0.20
        )
        if extra1 is not None:
            avoid = [best_box, extra1]
            extra2 = sample_non_overlapping_box(
                H, W, avoid,
                min_h_ratio=0.05, max_h_ratio=0.20,
                min_w_ratio=0.05, max_w_ratio=0.20
            )

    # 兜底：再进一步放宽（更小尺寸）确保尽量生成
    if extra1 is None:
        extra1 = sample_non_overlapping_box(
            H, W, [best_box],
            min_h_ratio=0.02, max_h_ratio=0.12,
            min_w_ratio=0.02, max_w_ratio=0.12
        )
    if extra2 is None and extra1 is not None:
        extra2 = sample_non_overlapping_box(
            H, W, [best_box, extra1],
            min_h_ratio=0.02, max_h_ratio=0.12,
            min_w_ratio=0.02, max_w_ratio=0.12
        )

    # ===== ⑤ 随机分配三种颜色，并记录 best_box 的颜色 =====
    color_pool = ['red', 'yellow', 'blue']
    random.shuffle(color_pool)
    color_best = color_pool[0]
    color_extra1 = color_pool[1] if extra1 is not None else None
    color_extra2 = color_pool[2] if extra2 is not None else None

    # ===== ⑥ 绘制并保存带三框（或部分框）的图片（保持原尺寸） =====
    h, w = ct_slice.shape
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)  # 1 px = 1 pt
    ax = fig.add_axes([0, 0, 1, 1])  # 占满画布
    ax.imshow(ct_slice, cmap='gray', aspect='auto')
    ax.axis('off')

    # 绘制 best_box（适度加粗）
    ax.add_patch(Rectangle(
        (minc, minr), maxc - minc, maxr - minr,
        linewidth=2.5, edgecolor=color_best, facecolor='none'
    ))

    # 绘制额外框
    if extra1 is not None:
        r0, c0, r1, c1 = extra1
        ax.add_patch(Rectangle(
            (c0, r0), c1 - c0, r1 - r0,
            linewidth=2.0, edgecolor=color_extra1, facecolor='none'
        ))
    if extra2 is not None:
        r0, c0, r1, c1 = extra2
        ax.add_patch(Rectangle(
            (c0, r0), c1 - c0, r1 - r0,
            linewidth=2.0, edgecolor=color_extra2, facecolor='none'
        ))

    box_path = folder_path+ '/' + anatomy + '.png'
    # plt.show
    # 注意：不使用 bbox_inches/pad_inches，保证像素级尺寸
    plt.savefig(box_path, dpi=100, facecolor='black')
    plt.close(fig)

    # ===== ⑦ 打印坐标信息 =====
    def points_dict(b):
        """将 (minr, minc, maxr, maxc) 转为四角点坐标字典"""
        r0, c0, r1, c1 = b
        return {
            "top_left": [int(r0), int(c0)],
            "top_right": [int(r0), int(c1)],
            "bottom_left": [int(r1), int(c0)],
            "bottom_right": [int(r1), int(c1)],
        }

    print(f"best_box(z={best_z}, 颜色={color_best}): ({minr},{minc})-({maxr},{maxc})")
    if extra1 is not None:
        r0, c0, r1, c1 = extra1
        print(f"extra_box_1(颜色={color_extra1}): ({r0},{c0})-({r1},{c1})")
    else:
        print("extra_box_1: 生成失败")
    if extra2 is not None:
        r0, c0, r1, c1 = extra2
        print(f"extra_box_2(颜色={color_extra2}): ({r0},{c0})-({r1},{c1})")
    else:
        print("extra_box_2: 生成失败")

    # ========= ④ 打印坐标 =========
    bbox_dict = {
        "slice_index": int(best_z),
        "best_box_color": color_best,
        "bbox_points": {
            "top_left": [int(minr), int(minc)],
            "top_right": [int(minr), int(maxc)],
            "bottom_left": [int(maxr), int(minc)],
            "bottom_right": [int(maxr), int(maxc)]
        }
    }

    wrong_box_color = color_extra2
    # 返回：box 坐标 + slice ID + 带框图路径 + 未加框图路径
    return bbox_dict, box_path, wrong_box_color



if __name__ == "__main__":
    box_points_path = "xxx/path"  # 你的 region 标注 JSON
    vqa_abnormality_path = 'xxx/path'
    out_qas = "xxx/path"
    region_box = 'xxx/path'

    ct_root = "xxx/path"
    mask_root = "xxx/path"

    # 先读取所有的不正常区域
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
            asked_region = record['asked_region']

            if asked_region == "abdomen" or asked_region == "pleura":
                continue
            lesion = record['abnormalities']

            box_path_global = os.path.join(Path_box, f"box_{unique_id_name}", f"{asked_region}_*")  # 使用 * 来匹配 xxx 部分
            # 找到画框的图
            boximg_path = glob.glob(box_path_global)
            if not boximg_path:
                # 可选：打印一下方便排查
                # print(f"[warn] no match: {box_path_global}")
                continue

            boximg_path = boximg_path[0]  # 或者 sorted(matches)[0] 确保确定性

            # 找到 nii + mask
            nii_path = find_file(ct_root, unique_id)

            if nii_path is None:
                print(f"[WARN] 找不到 CT 文件: {unique_id}")
                continue


            mask_path = find_mask(mask_root, unique_id, asked_region)
            if mask_path is None:
                print(f"[WARN] 找不到 mask_path CT 文件: {unique_id}")
                continue

            ct_img = nib.load(nii_path)
            mask_img = nib.load(mask_path)
            # get data
            ct_data = ct_img.get_fdata()
            mask_data = mask_img.get_fdata()

            bbox_dict, box_path, wrong_box_color = show_middle_slice_with_bbox_global(ct_data, mask_data, unique_id, asked_region)


            # 找到这个img + 画图
            '''不同颜色选一个 是对应病症'''
            qa = box_points_data(bbox_dict, box_path, lesion, unique_id, asked_region, wrong_box_color)


            if qa is None:
                continue

            fout.write(json.dumps(qa, ensure_ascii=False) + "\n")
            built += 1
    print(f"输入记录: {total}, 生成对抗QA(每个id一次): {built}, 输出: {out_qas}")
