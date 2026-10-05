'''

首先去validation_vqa_abnormality.csv 这里找有问题的id + 区域
然后在visual_evidence_box_points.json 这里找对应的数据
最后做QA

'''
import numpy as np
import cv2
import glob
import numpy as np
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

    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)  # 保存紧凑的图像，去掉多余白边

    print(f"Saved image with random box to {save_path}")

    return random_box_points, save_path # 返回生成的随机框坐标



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


# 保存到 CSV 文件
def save_to_csv(records, file_path):
    df = pd.DataFrame(records)
    df.to_csv(file_path, index=False, encoding='utf-8')

# 从 CSV 文件读取数据
def load_from_csv(file_path):
    return pd.read_csv(file_path)



def find_red_box_coordinates(image_path):
    """
    自动提取图像中的红框坐标。
    假设图像中的红框是红色的，并且使用颜色过滤来提取。
    """
    # 加载图像
    image = cv2.imread(image_path)

    # 将图像转换为HSV色彩空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义红色的HSV范围（红色可以有两个范围，低范围和高范围）
    lower_red1 = np.array([0, 120, 70])  # 红色的下限1
    upper_red1 = np.array([10, 255, 255])  # 红色的上限1
    lower_red2 = np.array([170, 120, 70])  # 红色的下限2
    upper_red2 = np.array([180, 255, 255])  # 红色的上限2

    # 使用颜色范围过滤出红色区域
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 查找图像中的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历所有轮廓，找到红框并提取坐标
    for contour in contours:
        # 获取每个轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 过滤掉过小的区域（你可以根据需要调整这个阈值）
        if w * h > 100:  # 假设红框的面积大于 100
            return x, y, x + w, y + h  # 返回框的坐标：x1, y1, x2, y2

    # 如果没有找到框，返回 None
    return None

def add_gaussian_noise(image, x1, y1, x2, y2, sigma=10):
    """
    给图像中的指定区域（由 x1, y1, x2, y2 给出）添加高斯噪声。
    """
    # 提取框内的区域
    region = image[y1:y2, x1:x2]

    # 生成高斯噪声
    noise = np.random.normal(0, sigma, region.shape)

    # 添加噪声
    noisy_region = region + noise

    # 确保噪声后的区域值在有效范围 [0, 255]
    noisy_region = np.clip(noisy_region, 0, 255).astype(np.uint8)

    # 将噪声区域放回原图
    image[y1:y2, x1:x2] = noisy_region
    return image



def box_points_data(record, boximg_path, save_niose_report_path, Sentence):
    """
    生成包含噪声的医学图像的报告数据。
    """
    unique_id = record.get('unique_id')
    id_name = unique_id.split(".")[0]
    asked_region = record.get('asked_region')
    Sentence = Sentence

    # 加载图像
    image = cv2.imread(boximg_path)  # 使用 OpenCV 读取图像

    # 自动提取红框坐标
    box_coords = find_red_box_coordinates(boximg_path)
    if box_coords is None:
        print(f"[warn] No red box found in {boximg_path}")
        return None

    # 获取红框的坐标
    x1, y1, x2, y2 = box_coords

    # 给红框区域添加高斯噪声
    noisy_image = add_gaussian_noise(image, x1, y1, x2, y2, sigma=10)

    # 将噪声图像保存为新的文件
    noisy_path_up = save_niose_report_path + id_name
    if not os.path.exists(noisy_path_up):
        os.makedirs(noisy_path_up)
    noisy_image_path = noisy_path_up +'/' + asked_region+ '.png'
    cv2.imwrite(noisy_image_path, noisy_image)

    qa = {
        "unique_id": unique_id,
        "region": asked_region,
        "image_paths": noisy_image_path,
        "question": "Generate a clinical report for this CT scan.",
        "answer": Sentence,
        "task": "noise_report",
        "modality": "CT",
        "type": "correct_box_add_noise",
        "question_type": 'open_question',
        # "noisy_img_path": noisy_image_path  # 返回带噪声的图像路径
    }

    return qa



def box_points_data2(record, boximg_path, save_niose_report_path, Sentence):
    """
    生成包含噪声的医学图像的报告数据。
    """
    unique_id = record.get('unique_id')
    id_name = unique_id.split(".")[0]
    asked_region = record.get('asked_region')
    Sentence = Sentence

    # # 加载图像
    # image = cv2.imread(boximg_path)  # 使用 OpenCV 读取图像
    #
    # # 自动提取红框坐标
    # box_coords = find_red_box_coordinates(boximg_path)
    # if box_coords is None:
    #     print(f"[warn] No red box found in {boximg_path}")
    #     return None
    #
    # # 获取红框的坐标
    # x1, y1, x2, y2 = box_coords
    #
    # # 给红框区域添加高斯噪声
    # noisy_image = add_gaussian_noise(image, x1, y1, x2, y2, sigma=30)
    #
    # # 将噪声图像保存为新的文件
    # noisy_image_path = save_niose_report_path + id_name +'.png'
    # cv2.imwrite(noisy_image_path, noisy_image)

    qa = {
        "unique_id": unique_id,
        "region": asked_region,
        "image_paths": boximg_path,
        "question": "Generate a clinical report for this CT scan.",
        "answer": Sentence,
        "task": "noise_report",
        "modality": "CT",
        "type": "correct_box_no_noise",
        "question_type": 'open_question',
    }

    return qa




if __name__ == "__main__":
    box_points_path = "xxx/path"  # 你的 region 标注 JSON
    vqa_abnormality_path = 'xxx/path'
    out_qas = "xxx/path"
    save_niose_report_path = 'xxx/path'
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
    Path = "xxx/path"
    Path_no_box = "xxx/path"
    # 找到对应的box img
    with open(out_qas, "a", encoding="utf-8") as fout:
        for record in records_list:
            unique_id = record['unique_id']
            unique_id = unique_id.split(".")[0]
            asked_region = record['asked_region']

            Sentence = record['Sentence']

            ''''# region_box 找到画对的框 '''
            box_path_global = os.path.join(Path, f"box_{unique_id}", f"{asked_region}_*")  # 使用 * 来匹配 xxx 部分
            # 使用 glob 查找所有符合模式的文件
            boximg_path = glob.glob(box_path_global)
            if not boximg_path:
                # 可选：打印一下方便排查
                # print(f"[warn] no match: {box_path_global}")
                continue

            boximg_path = boximg_path[0]  # 或者 sorted(matches)[0] 确保确定性

            ''''# lesion_wrong_box 找到画错的框 '''
            # 获取文件所在的文件夹路径
            box_folder = os.path.dirname(boximg_path)

            # 列出文件夹中的所有文件
            all_files = os.listdir(box_folder)

            # 过滤掉正确的框文件，避免选择到它
            wrong_files = [f for f in all_files if f != os.path.basename(boximg_path)]

            # 如果有其他文件可选
            if wrong_files:
                # 随机选择一个文件作为 wrong_box_path
                wrong_box_filename = random.choice(wrong_files)
                wrong_box_path = os.path.join(box_folder, wrong_box_filename)
            else:
                # 如果文件夹里没有其他文件，可以做一些处理，例如跳过
                print(f"[warn] No other files in {box_folder} to choose from for wrong_box_path")
                continue

            # 找到这个img + 画图
            '''correct box + noise'''
            qa = box_points_data(record, boximg_path,save_niose_report_path, Sentence)

            '''correct box no noise'''
            # qa = box_points_data2(record, boximg_path, save_niose_report_path,Sentence)


            if qa is None:
                continue

            fout.write(json.dumps(qa, ensure_ascii=False) + "\n")
            built += 1
    print(f"输入记录: {total}, 生成对抗QA(每个id一次): {built}, 输出: {out_qas}")






