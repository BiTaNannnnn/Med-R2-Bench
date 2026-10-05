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


# 仅保留答案不以 N（忽略大小写）开头的记录
correct_color_data = [
    r for r in data
    if r.get("type") == "correct_box_no_artifact"
       and str(r.get("answer", "")).strip().upper()[:1] != "N"
]

wrong_color_data = [
    r for r in data
    if r.get("type") == "correct_box_add_artifact"
       and str(r.get("answer", "")).strip().upper()[:1] != "N"
]

# 随机从每个类别中选取80条数据
correct_color_test = random.sample(correct_color_data, 100) if len(correct_color_data) >= 100 else correct_color_data
wrong_color_test = random.sample(wrong_color_data, 100) if len(wrong_color_data) >= 100 else wrong_color_data

# 合并选中的数据
test_data = correct_color_test + wrong_color_test

# 剩下的数据作为 val 数据
remaining_data = [
    r for r in data
    if r not in test_data
       and str(r.get("answer", "")).strip().upper()[:1] != "N"
]

# 保存test_jsonl
with open(test_jsonl_path, 'a') as test_file:
    for record in test_data:
        test_file.write(json.dumps(record) + '\n')

# 保存val_jsonl
with open(val_jsonl_path, 'a') as val_file:
    for record in remaining_data:
        val_file.write(json.dumps(record) + '\n')

print('Files saved as test_jsonl and val_jsonl')
