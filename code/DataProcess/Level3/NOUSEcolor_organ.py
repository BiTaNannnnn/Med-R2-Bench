import csv
import os
from collections import defaultdict, Counter
import nibabel as nib
import os
import csv
import nibabel as nib
from skimage.measure import label, regionprops
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from matplotlib.patches import Rectangle
import json


def process_volumename_and_anatomy(csv_path):
    """
    读取 CSV 文件并提取 Volumename 和 anatomy，统计每个 Volumename 下的唯一器官类别。
    """
    # 用字典存储每个 Volumename 对应的器官列表
    volumename_anatomy = defaultdict(set)

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头

        for row in reader:
            Volumename, anatomy, finding, presence = row

            if presence.strip() != "Yes.":
                continue  # 只处理 "Yes" 的记录

            # 提取 anatomy 字段的第一个部分（即器官名）
            anatomy = anatomy.split('/')[0].strip()

            # 使用 Volumename 作为键，anatomy 作为值
            volumename_anatomy[Volumename].add(anatomy)

    # 返回 Volumename 和它的器官类别的字典
    return volumename_anatomy

def find_file(root, filename):
    for dirpath, dirnames, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


def find_mask(mask_root,ct_name,anatomy):


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


def process_volumename_anatomy(Volumename, anatomy_list):
    """
    处理每个 Volumename 和它的器官列表，输出该 Volumename 和器官类别。
    """
    anatomy_count = len(anatomy_list)
    print(f"处理 {Volumename}，器官类别: {anatomy_list}, 器官个数: {anatomy_count}")

    # 获取 CT 数据路径
    nii_path = find_file(ct_root, Volumename)

    # 加载 CT 数据
    ct_img = nib.load(nii_path)
    ct_data = ct_img.get_fdata()

    # CT 投影（沿 z 轴取最大值）
    ct_img_proj = np.max(ct_data, axis=2)  # 前后方向压平
    ct_img_proj -= ct_img_proj.min()  # 拉到 0-255
    ct_img_proj /= ct_img_proj.max()
    ct_img_proj = (ct_img_proj * 255).astype(np.uint8)

    # 对于每一个器官进行处理
    for anatomy in anatomy_list:
        print(f"Volumename: {Volumename}, Anatomy: {anatomy}")

        # 获取 mask 文件路径
        mask_path = find_mask(mask_root, Volumename, anatomy)
        if mask_path is None:
            print(f"[WARN] 找不到 mask 文件: {Volumename}, {anatomy}")
            continue

        # 加载 mask 数据
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()

        # 对 Mask 数据做最大值投影
        mask_img_proj = np.max(mask_data, axis=2)  # 对z轴做最大值投影

        # 将 Mask 投影图像归一化处理到 [0, 255] 范围
        mask_img_proj = (mask_img_proj * 255).astype(np.uint8)

        # 设定 mask 的颜色：使用红色，绿色，蓝色，或者自定义颜色
        mask_rgb = np.zeros((mask_img_proj.shape[0], mask_img_proj.shape[1], 3), dtype=np.uint8)

        # 将 mask 图像设置为红色（或者其他你想要的颜色）
        mask_rgb[:, :, 0] = mask_img_proj  # 红色通道

        # 显示 CT 投影图像并叠加 Mask
        plt.figure(figsize=(6, 6))
        plt.imshow(ct_img_proj, cmap='gray')
        save_path = 'xxx/path'
        # 使用透明度 (alpha) 将 Mask 图像叠加到 CT 图像上
        plt.imshow(mask_rgb, alpha=0.5)  # 设置透明度为 50%
        plt.axis('off')
        plt.title(f"CT Projection with Mask: {anatomy}")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        # 单独展示 mask 图像（不叠加 CT）
        plt.figure(figsize=(6, 6))
        plt.imshow(mask_img_proj, cmap='gray')
        save_path = 'xxx/path'
        plt.title(f"Mask for {anatomy}")
        plt.axis('off')
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()




if __name__ == "__main__":
    # 设定 CSV 路径
    csv_path = 'xxx/path'  # 替换为你的 CSV 文件路径
    ct_root = "xxx/path"
    mask_root = "xxx/path"

    # 读取并处理 Volumename 和 anatomy
    volumename_anatomy = process_volumename_and_anatomy(csv_path)

    # 对每个 Volumename 和它的器官类别列表进行处理
    for Volumename, anatomy_list in volumename_anatomy.items():
        # Volumename = "valid_1082_a_1.nii.gz"
        process_volumename_anatomy(Volumename, list(anatomy_list))
