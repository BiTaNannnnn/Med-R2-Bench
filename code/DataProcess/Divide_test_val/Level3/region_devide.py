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
correct_box_correct_Q_data = [record for record in data if record.get("type") == "correct_box_correct_Q"]
wrong_box_correct_Q_data = [record for record in data if record.get("type") == "wrong_box_correct_Q"]
# wrong_box_wrong_Q_data = [record for record in data if record.get("type") == "wrong_box_wrong_Q"]
correct_box_wrong_Q_data = [record for record in data if record.get("type") == "correct_box_wrong_Q"]


# 随机从每个类别中选取80条数据
correct_box_correct_Q_test = random.sample(correct_box_correct_Q_data, 2250) if len(correct_box_correct_Q_data) >= 2250 else correct_box_correct_Q_data
wrong_box_correct_Q_test = random.sample(wrong_box_correct_Q_data, 2250) if len(wrong_box_correct_Q_data) >= 2250 else wrong_box_correct_Q_data

# wrong_box_wrong_Q_test = random.sample(wrong_box_wrong_Q_data, 2250) if len(wrong_box_wrong_Q_data) >= 2250 else wrong_box_wrong_Q_data
correct_box_wrong_Q_test = random.sample(correct_box_wrong_Q_data, 2250) if len(correct_box_wrong_Q_data) >= 2250 else correct_box_wrong_Q_data


# 合并选中的数据
test_data = correct_box_correct_Q_test + wrong_box_correct_Q_test  + correct_box_wrong_Q_test

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
