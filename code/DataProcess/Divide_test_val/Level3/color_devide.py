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

# 根据 "type" 和 "question_type" 分类数据
correct_color_data = [record for record in data if record.get("type") == "correct_color-correct_region"]
wrong_color_data = [record for record in data if record.get("type") == "wrong_color-correct_region"]
open_question_data = [record for record in data if record.get("question_type") == "normal_question"]

# 随机从每个类别中选取80条数据
correct_color_test = random.sample(correct_color_data, 100) if len(correct_color_data) >= 80 else correct_color_data
wrong_color_test = random.sample(wrong_color_data, 100) if len(wrong_color_data) >= 80 else wrong_color_data
normal_question_test = random.sample(open_question_data, 100) if len(open_question_data) >= 80 else open_question_data

# 合并选中的数据
test_data = correct_color_test + wrong_color_test + normal_question_test

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
