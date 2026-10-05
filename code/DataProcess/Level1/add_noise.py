"""
# CT nii --> MIX --> add Noise
# --- left = 原始数据 right = 加噪声数据
# --->  合并成一张img
"""



import csv
import os
from typing import Optional
import nibabel as nib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def y_proj_gauss(nii_path: str, sigma: float = 10, seed: Optional[int] = None) -> Image.Image:
    # 1-2. 读取 + MIP
    nii = nib.load(nii_path)
    data = nii.get_fdata().astype(np.float32)  # shape (X, Y, Z)
    # 2. Y 轴平均投影 → (X, Z)
    proj = np.max(data, axis=2)  # 前后方向压平
    proj -= proj.min()  # 拉到 0-255
    proj /= proj.max()
    proj = (proj * 255).astype(np.uint8)

    # '''show'''
    # # 显示 proj 图像
    # plt.imshow(proj, cmap='gray')  # 使用灰度色图显示
    # plt.colorbar()  # 显示颜色条
    # plt.title('Projection Image (proj)')
    # plt.show()


    # 3. 加高斯噪声
    noise = np.random.normal(0, sigma, proj.shape)
    proj_noisy = proj + noise

    # # 显示噪声本身
    # plt.imshow(noise, cmap='gray')
    # plt.title('Noise Image')
    # plt.colorbar()
    # plt.show()

    # '''show'''
    # # 显示 proj 图像
    # plt.imshow(proj_noisy, cmap='gray')  # 使用灰度色图显示
    # plt.colorbar()  # 显示颜色条
    # plt.title('Projection Image (proj_noisy)')
    # plt.show()


    # 创建一个新图像，宽度为两个图像的宽度之和，高度为两个图像的高度
    width = proj_noisy.shape[1] + proj.shape[1]
    height = max(proj_noisy.shape[0], proj.shape[0])

    # 创建一个新图像，模式为 L（灰度图），尺寸为合并后的宽高
    img = Image.new('L', (width, height))

    # 将 proj_noisy_uint8 和 proj 放到新图像的不同位置
    img.paste(Image.fromarray(proj_noisy), (0, 0))  # 左边放 proj_noisy_uint8
    img.paste(Image.fromarray(proj), (proj_noisy.shape[1], 0))  # 右边放 proj

    # 返回合并后的图像 (确保返回 PIL.Image 对象)
    return img


def find_nii(mainPath):
    # 定义CSV文件的路径
    csv_file_path = 'xxx/path'
    # 读取CSV文件
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行

        # 遍历每个文件名
        for row in reader:
            file_name = row[0]  # 获取每一行的文件名

            # 根据文件名构建对应的路径
            subfolder1 = f"valid_{file_name.split('_')[1]}"  # valid_1、valid_2 等
            subfolder2 = f"valid_{file_name.split('_')[1]}a"  # valid_1a、valid_2a 等
            niiPath = os.path.join(mainPath, subfolder1, subfolder2, file_name)
            # 检查文件是否存在并加载
            if os.path.exists(niiPath):

                img = y_proj_gauss(niiPath, sigma=10)

                savePathMain = 'xxx/path'

                # 使用 os.path.join 来安全拼接路径，确保路径末尾有分隔符
                save_name = file_name.split('.')[0] + '.png'  # 去掉原文件名的 .nii.gz 后缀
                save_path = os.path.join(savePathMain, save_name)  # 使用 os.path.join 拼接路径
                print(type(img))
                # 保存合成图像
                img.save(save_path, format='PNG')
                print(f"已保存 {save_name}")
            else:
                print(f"文件未找到: {niiPath}")



if __name__ == "__main__":

    # 通过 文件名 文件夹里找nii.gz 文件
    mainPath = 'xxx/path'
    niiPath = find_nii(mainPath)

