# import os
#
#
# def count_png_files(directory):
#     png_count = 0
#
#     # 遍历目录及其所有子目录
#     for root, dirs, files in os.walk(directory):
#         # 查找当前目录中的所有 .png 文件
#         png_count += sum(1 for file in files if file.lower().endswith('.png'))
#
#     return png_count
#
#
# # 指定需要检查的文件夹路径
# folder_path = 'xxx/path'
#
# # 调用函数并输出结果
# png_count = count_png_files(folder_path)
# print(f"Total number of .png files: {png_count}")



import os
from pathlib import Path


def count_png_files(root_folder):
    """
    递归统计指定文件夹及其所有子文件夹中的 .png 文件数量

    Args:
        root_folder (str): 根文件夹路径

    Returns:
        int: PNG文件总数
        list: PNG文件完整路径列表
    """
    png_count = 0
    png_files = []

    # 使用 pathlib 进行递归遍历
    root_path = Path(root_folder)

    # 递归查找所有 .png 文件（不区分大小写）
    for png_file in root_path.rglob('*.png'):
        png_count += 1
        png_files.append(str(png_file))

    # 同时也查找大写的 .PNG 文件（如果有的话）
    for png_file in root_path.rglob('*.PNG'):
        png_count += 1
        png_files.append(str(png_file))

    return png_count, png_files


def main():
    # 可以修改这里的文件夹路径
    folder_path = 'xxx/path'

    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return

    if not os.path.isdir(folder_path):
        print(f"错误: '{folder_path}' 不是文件夹")
        return

    try:
        count, files = count_png_files(folder_path)

        print(f"\n统计结果:")
        print(f"文件夹: {folder_path}")
        print(f"PNG文件总数: {count}")

        # # 可选: 显示所有找到的文件路径
        # show_files = input("\n是否显示所有PNG文件路径? (y/n): ").strip().lower()
        # if show_files == 'y':
        #     print("\n找到的PNG文件:")
        #     for i, file in enumerate(files, 1):
        #         print(f"{i:4d}. {file}")
        # 
        # # 可选: 保存结果到文件
        # save_result = input("\n是否将结果保存到文件? (y/n): ").strip().lower()
        # if save_result == 'y':
        #     output_file = "png_files_list.txt"
        #     with open(output_file, 'w', encoding='utf-8') as f:
        #         f.write(f"PNG文件统计结果\n")
        #         f.write(f"文件夹: {folder_path}\n")
        #         f.write(f"PNG文件总数: {count}\n")
        #         f.write("-" * 50 + "\n")
        #         for i, file in enumerate(files, 1):
        #             f.write(f"{i:4d}. {file}\n")
        #     print(f"结果已保存到: {output_file}")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")


# 更简洁的版本（如果只需要计数）
def count_png_simple(root_folder):
    """简洁版本，只返回PNG文件数量"""
    return sum(1 for _ in Path(root_folder).rglob('*.png')) + \
        sum(1 for _ in Path(root_folder).rglob('*.PNG'))


if __name__ == "__main__":
    main()
