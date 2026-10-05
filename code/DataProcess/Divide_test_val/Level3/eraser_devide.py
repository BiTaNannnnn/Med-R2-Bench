import json
import random

# 文件路径
input_jsonl_path = "xxx/path"
test_jsonl_path = 'xxx/path'
val_jsonl_path = 'xxx/path'

# 读取jsonl文件并加载数据
data = []
with open(input_jsonl_path, 'r') as file:
    for line in file:
        data.append(json.loads(line.strip()))

# 十一种不同的问法
# question_variants = [
#     "In this pair of CT images, the original scan is on the left and the AI-edited version is on the right. Can you identify the organ that has been altered or removed in the right image?",
#     "Looking at the two CT images, with the left being the original and the right the AI-modified, can you determine which organ has been edited or removed in the right image?",
#     "The left CT image shows the original scan, and the right one shows the AI-modified version. Can you recognize which organ has been edited or removed in the right image?",
#     "Can you spot which organ has been edited or removed in the AI-modified CT image on the right, compared to the original scan on the left?",
#     "In the given pair of CT images, the original scan is on the left and the AI-modified version is on the right. Which organ is missing or altered in the right image?",
#     "The left CT image is the original, and the right one is AI-modified. Can you identify the organ that has been removed or changed in the right image?",
#     "In the pair of CT images, with the original on the left and the AI-edited on the right, which organ appears to have been removed or altered in the right image?",
#     "Given these two CT images, with the original on the left and the AI-modified on the right, can you identify which organ has been edited or removed in the right image?",
#     "The left CT image is unaltered, and the right image has been modified by AI. Can you recognize which organ has been edited or removed in the right image?",
#     "Can you determine which organ has been edited or removed in the right image of this CT pair, where the left image is the original scan?",
#     "This is a pair of CT images, with the original scan on the left and the AI-edited image on the right. Can you identify which type of organ has been removed or edited in the right image?"
# ]

# 随机选择一个问题并替换每条记录的 'question' 字段
# for record in data:
#     record['question'] = random.choice(question_variants)

# 根据 "modality" 分类数据
Radiograph = [record for record in data if record.get("modality") == "Radiograph"]
Magnetic_Resonance_Imaging = [record for record in data if record.get("modality") == "Magnetic_Resonance_Imaging"]
CT = [record for record in data if record.get("modality") == "CT"]

# 随机从每个类别中选取2250条数据
Radiograph_test = random.sample(Radiograph, 33) if len(Radiograph) >= 33 else Radiograph
CT_test = random.sample(CT, 50) if len(CT) >= 50 else CT
MRI_test = random.sample(Magnetic_Resonance_Imaging, 50) if len(Magnetic_Resonance_Imaging) >= 50 else Magnetic_Resonance_Imaging

# 合并选中的数据
test_data = Radiograph_test + CT_test + MRI_test

# 剩下的数据作为 val 数据
remaining_data = [record for record in data if record not in test_data]

# 保存test_jsonl
with open(test_jsonl_path, 'a') as test_file:
    for record in test_data:
        test_file.write(json.dumps(record) + '\n')

# 保存val_jsonl
with open(val_jsonl_path, 'a') as val_file:
    for record in remaining_data:
        val_file.write(json.dumps(record) + '\n')

print('Files saved as test_jsonl and val_jsonl')
