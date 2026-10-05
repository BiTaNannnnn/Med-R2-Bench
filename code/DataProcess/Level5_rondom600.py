import json
import random
import re

# 读取jsonl文件
input_file = 'xxx/path'  # 输入文件路径
output_file = 'xxx/path'  # 输出文件路径
# input_file = 'xxx/path'  # 输入文件路径
# output_file = 'xxx/path'  # 输出文件路径

# 目标类型列表
target_types = [
    "correct_box",
    "wrong_box",
    "correct_box_no_noise",
    "correct_box_add_noise",
    "correct_box_no_artifact",
    "correct_box_add_artifact"
]

# 存储按类型分类的记录
type_dict = {key: [] for key in target_types}  # 初始化每个类型的空列表

# 读取文件并按type分类
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        metadata = item.get('metadata')
        item_type = metadata.get('type')
        # item_type = item.get('type')

        # 检查type是否在目标类型列表中
        if item_type in target_types:
            ans = item.get('GT_ANSWER', '')
            # 不以 N 开头（忽略大小写 & 前后空格）
            if ans.strip() and not ans.strip().upper().startswith('N'):
                type_dict[item_type].append(item)



# 为每个类型随机选取100个样本
selected_items = []
for item_type, items in type_dict.items(): # 500 = train 100 = qwen
    selected_items.extend(random.sample(items, min(100, len(items))))  # 每种类型取100个，若该类型少于100个则全部取

# 将选中的样本写入新的jsonl文件
with open(output_file, 'w', encoding='utf-8') as f:
    for item in selected_items:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

print(f'Filtered data has been written to {output_file}')
