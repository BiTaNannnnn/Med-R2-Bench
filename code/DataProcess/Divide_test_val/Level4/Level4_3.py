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

# 随机选择1230条数据作为 test 数据
test_data = random.sample(data, 1230)  # 随机选取 1230 条记录

# 剩下的数据作为 val 数据
remaining_data = [record for record in data if record not in test_data]

# 保存test_jsonl
with open(test_jsonl_path, 'a') as test_file:
    for record in test_data:
        test_file.write(json.dumps(record, ensure_ascii=False) + '\n')

# 保存val_jsonl
with open(val_jsonl_path, 'a') as val_file:
    for record in remaining_data:
        val_file.write(json.dumps(record, ensure_ascii=False) + '\n')

print('Files saved as test_jsonl and val_jsonl with 1230 records for test.')
