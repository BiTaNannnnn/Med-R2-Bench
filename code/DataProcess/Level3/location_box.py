import os
import csv
import nibabel as nib
from skimage.measure import label, regionprops
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from matplotlib.patches import Rectangle
import json


'''
Total nii files: 1564
Total PNG files: 9381
'''


def show_middle_slice_with_bbox(ct_data, mask_data):
    """
    在中间那一层 slice 上，根据 mask 找最大连通块，
    在对应的 CT slice 上画红色加粗的 bbox。
    """
    # 取中间层索引
    z = mask_data.shape[2] // 2

    ct_slice = ct_data[:, :, z]
    mask_slice = mask_data[:, :, z] > 0

    # 如果这一层没有 mask，就只显示CT
    if mask_slice.max() == 0:
        print("中间这层没有前景 mask，只显示 CT。")
        plt.figure(figsize=(6, 6))
        plt.imshow(ct_slice, cmap="gray")
        plt.axis("off")
        plt.title(f"Middle slice z={z} (no mask)")
        plt.show()
        return

    # 连通域标记
    labeled = label(mask_slice)
    regions = regionprops(labeled)

    if len(regions) == 0:
        print("中间这层没有连通区域，只显示 CT。")
        plt.figure(figsize=(6, 6))
        plt.imshow(ct_slice, cmap="gray")
        plt.axis("off")
        plt.title(f"Middle slice z={z} (no region)")
        plt.show()
        return

    # 选面积最大的连通块
    largest_region = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = largest_region.bbox  # (row_min, col_min, row_max, col_max)

    # 画图
    plt.figure(figsize=(6, 6))
    plt.imshow(ct_slice, cmap="gray")
    ax = plt.gca()

    # 注意：Rectangle 的参数是 (x, y, width, height)，x=列，y=行
    rect = Rectangle(
        (minc, minr),
        maxc - minc,
        maxr - minr,
        linewidth=2,        # 加粗
        edgecolor="red",    # 红色
        facecolor="none"    # 只画框，不填充
    )
    ax.add_patch(rect)

    plt.axis("off")
    plt.title(f"Middle slice z={z} with largest bbox")
    plt.show()

    print("最大框四个点坐标：")
    print("左上:", (int(minr), int(minc)))
    print("右上:", (int(minr), int(maxc)))
    print("左下:", (int(maxr), int(minc)))
    print("右下:", (int(maxr), int(maxc)))


def show_middle_slice_with_bbox_global(ct_data, mask_data, Volumename, anatomy):
    """
    1. 在整个 3D mask 里（所有 z）遍历，找出面积最大的连通区域（全局最大 region）
    2. 用这个 region 的 bbox，在“中间那一层 CT slice”上画红色加粗矩形
    3. 打印这个最大框的四个坐标和它原本所在的 z
    """
    H, W, D = mask_data.shape

    best_area = -1
    best_box = None      # (minr, minc, maxr, maxc)
    best_z = None        # 这个最大连通域所在的 z

    # ========= ① 遍历所有 z，找全局最大的连通区域 =========
    for z in range(D):
        mask_slice = mask_data[:, :, z] > 0

        if not mask_slice.any():
            continue

        labeled = label(mask_slice)
        regions = regionprops(labeled)

        if not regions:
            continue

        # 这一张 slice 里的最大连通块
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
        return None  # 返回 None，表示没有找到前景区域

    minr, minc, maxr, maxc = best_box

    # ========= ② 保存没有框的 CT slice =========
    ct_slice = ct_data[:, :, best_z]
    Volumename = Volumename.split('.')[0]
    Volumename = 'box_' + Volumename
    base_dir = "xxx/path"
    folder_path = os.path.join(base_dir, Volumename)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    # 直接保存没有框的 CT slice 图像为 PNG
    ct_no_box_path = os.path.join(folder_path, f"{anatomy}_ID{best_z}.png")
    plt.figure(figsize=(6, 6))
    plt.imshow(ct_slice, cmap="gray")
    plt.axis("off")  # 去掉坐标轴
    plt.savefig(ct_no_box_path, dpi=150, bbox_inches="tight", pad_inches=0)  # pad_inches=0 让图像紧凑
    plt.close()  # 关闭当前图形，避免内存占用
    print(f"保存未加框的 CT slice 到：{ct_no_box_path}")

    # ========= ③ 在“中间那一张 CT”上画这个全局最大框 =========
    plt.figure(figsize=(6, 6))
    plt.imshow(ct_slice, cmap="gray")
    ax = plt.gca()

    rect = Rectangle(
        (minc, minr),           # (x, y) = (列, 行)
        maxc - minc,            # 宽
        maxr - minr,            # 高
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )
    ax.add_patch(rect)

    folder_path = "xxx/path"
    folder_path = os.path.join(folder_path, Volumename)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    box_path = os.path.join(folder_path, f"{anatomy}_ID{best_z}.png")

    plt.axis("off")  # 去掉坐标轴
    plt.title('')  # 不显示标题
    plt.savefig(box_path, dpi=150, bbox_inches="tight", pad_inches=0)  # 保存紧凑的图像，去掉多余白边
    plt.close()

    # ========= ④ 打印坐标 =========
    bbox_dict = {
        "slice_index": int(best_z),
        "bbox_points": {
            "top_left": [int(minr), int(minc)],
            "top_right": [int(minr), int(maxc)],
            "bottom_left": [int(maxr), int(minc)],
            "bottom_right": [int(maxr), int(maxc)]
        }
    }

    # 返回 box坐标 + slice ID + 保存带box的第z张img + 保存没有框的 CT slice
    return bbox_dict, best_z, box_path, ct_no_box_path



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



if __name__ == "__main__":
    csv_path = "xxx/path"

    ct_root = "xxx/path"
    mask_root = "xxx/path"

    # 👉 存放所有样本的 box 信息
    results = []
    seen_pairs = set()
    # 读取nii + region
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)

        next(reader)  # ✅ 跳过第一行表头 Volumename,Anatomy,...

        for row in reader:
            Volumename, anatomy, finding, presence = row
            anatomy = anatomy.split('/')[0].strip()
            # 检查是否同意区域多次保存
            key = (Volumename, anatomy)
            if key in seen_pairs:
                # 已经处理过这个 unique_id + region，直接跳过
                continue
            seen_pairs.add(key)

            nii_path = find_file(ct_root, Volumename)

            if nii_path is None:
                print(f"[WARN] 找不到 CT 文件: {Volumename}")
                continue

            mask_path = find_mask(mask_root,Volumename,anatomy)
            if mask_path is None:
                print(f"[WARN] 找不到 mask_path CT 文件: {Volumename}")
                continue

            # --- 3. 用 nibabel 读取 nii 数据 ---

            ct_img = nib.load(nii_path)
            mask_img = nib.load(mask_path)
            # get data
            ct_data = ct_img.get_fdata()
            mask_data = mask_img.get_fdata()

            # all_bboxes, largest_bbox = get_largest_bbox_points(mask_data)
            # show_middle_slice_with_bbox(ct_data, mask_data)

            out = show_middle_slice_with_bbox_global(ct_data, mask_data, Volumename, anatomy)

            if out is None:
                continue  # 如果返回 None，就跳过当前样本

            bbox_dict, best_z, box_path, ct_no_box_path = out

            # 组装一条 JSON 记录
            record = {
                "unique_id": Volumename,
                "region": anatomy,
                "biggest_slice": best_z,
                "BoxImg_savepath": box_path,
                "NoBoxImg_savepath": ct_no_box_path,
                "Points": bbox_dict
            }
            results.append(record)

        # 统一写成一个 json 文件
    json_out_path = "xxx/path"
    with open(json_out_path, "w", encoding="utf-8") as jf:
        json.dump(results, jf, ensure_ascii=False, indent=4)

    print("Finished! JSON 已保存到：", json_out_path)



