import csv

# 定义 CSV 文件的保存路径
csv_file_path = 'xxx/path'

# 创建并写入 CSV 文件
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)

    # 写入 CSV 文件的标题行
    writer.writerow(['File Name'])

    # 生成并写入文件名（从 valid_1_a_1.nii.gz 到 valid_1304_a_1.nii.gz）
    for i in range(1, 1305):
        file_name = f'valid_{i}_a_1.nii.gz'
        writer.writerow([file_name])

print(f"CSV 文件已保存为 {csv_file_path}")