import os
import csv
from collections import Counter


def count_unique_categories_and_counts(csv_path):
    """
    从 CSV 文件中提取 anatomy 字段的类别，统计每种类别的数量及其出现次数
    """
    categories = []  # 用列表来存储所有类别

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)

        # 跳过第一行表头
        next(reader)

        for row in reader:
            Volumename, anatomy, finding, presence = row

            # 提取 anatomy 字段中的第一个类别
            # 通过 `/` 分割，取第一个部分
            category = anatomy.split('/')[0]

            categories.append(category)  # 将类别添加到列表中

    # 使用 Counter 统计每个类别的出现次数
    category_counts = Counter(categories)

    # 输出结果
    print(f"不同的类别数量: {len(category_counts)}")
    print("每个类别出现的次数：")
    for category, count in category_counts.items():
        print(f"{category}: {count}")


def count_png_files_in_subfolders(base_dir):
    """
    遍历 base_dir 文件夹下的所有子文件夹，统计每个子文件夹中的 PNG 文件数量
    并计算所有 PNG 文件的总数
    """
    total_png_count = 0  # 用来存储所有 PNG 文件的总数

    # 遍历 base_dir 文件夹下的所有子文件夹
    for subfolder in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subfolder)

        # 确保是文件夹
        if os.path.isdir(subfolder_path):
            png_count = 0
            # 遍历子文件夹中的所有文件
            for filename in os.listdir(subfolder_path):
                # 检查文件后缀是否是 .png
                if filename.lower().endswith('.png'):
                    png_count += 1

            # 打印该子文件夹中 PNG 文件的数量
            print(f"{subfolder}: {png_count} PNG files")

            # 累加到总数
            total_png_count += png_count

    # 打印总计
    print(f"\nTotal PNG files: {total_png_count}")

if __name__ == "__main__":
    # '''计算所有box - PNG的数量'''
    # # 设置你的 base_dir
    base_dir = 'xxx/path'  # 替换为你的文件夹路径
    count_png_files_in_subfolders(base_dir)


    '''计算所有器官的种类 '''
    # # 使用你的 CSV 路径
    # csv_path = 'xxx/path'  # 替换为实际的 CSV 文件路径
    # count_unique_categories_and_counts(csv_path)
    '''
    不同的类别数量: 11
    不同的类别有：
    trachea and bronchie
    abdomen
    heart
    mediastinum
    lung
    others
    bone
    pleura
    esophagus
    thyroid
    breast
    '''
