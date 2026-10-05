'''
加载数据：假设 json_data 已加载，您应该从实际的 JSON 文件中读取。
计算记录总数：通过 len(json_data) 来获取记录的总数。
随机选择870条记录：使用 random.sample 从 json_data 中随机选择870条记录。
问题变体：为每条记录的 question 字段随机选择一个问法。
写入 JSONL 文件：将 test_data 和 val_data 写入 test_jsonl 和 val_jsonl 文件。
'''


import json
import random

test_jsonl = 'xxx/path'
val_jsonl = 'xxx/path'

json_data_path = "xxx/path"

# 读取jsonl文件并加载数据
json_data = []
with open(json_data_path, 'r') as file:
    for line in file:
        json_data.append(json.loads(line.strip()))

# 按照 "type" 和 "task" 分类数据
correct_noise = [record for record in json_data if record.get("type") == "correct_question" and record.get("task") == "noise_recognition"]
correct_artifact = [record for record in json_data if record.get("type") == "correct_question" and record.get("task") == "artifact_recognition"]
wrong_noise = [record for record in json_data if record.get("type") == "wrong_question" and record.get("task") == "noise_recognition"]
wrong_artifact = [record for record in json_data if record.get("type") == "wrong_question" and record.get("task") == "artifact_recognition"]
normal_noise = [record for record in json_data if record.get("type") == "normal_question" and record.get("task") == "noise_recognition"]
normal_artifact = [record for record in json_data if record.get("type") == "normal_question" and record.get("task") == "artifact_recognition"]

# 确保每个分类有 440 条数据，且总共不超过 880 条（test/val 各 440 条）
correct_noise_test = random.sample(correct_noise, 440) if len(correct_noise) >= 440 else correct_noise
correct_artifact_test = random.sample(correct_artifact, 440) if len(correct_artifact) >= 440 else correct_artifact
wrong_noise_test = random.sample(wrong_noise, 440) if len(wrong_noise) >= 440 else wrong_noise
wrong_artifact_test = random.sample(wrong_artifact, 440) if len(wrong_artifact) >= 440 else wrong_artifact
normal_noise_test = random.sample(normal_noise, 440) if len(normal_noise) >= 440 else normal_noise
normal_artifact_test = random.sample(normal_artifact, 440) if len(normal_artifact) >= 440 else normal_artifact

# 合并所有测试数据
test_data = correct_noise_test + correct_artifact_test + wrong_noise_test + wrong_artifact_test + normal_noise_test + normal_artifact_test

# 剩下的数据作为 val 数据
remaining_data = [record for record in json_data if record not in test_data]


# 将数据写入 test_jsonl 和 val_jsonl 文件
with open(test_jsonl, 'a') as test_file:
    for record in test_data:
        test_file.write(json.dumps(record) + '\n')

with open(val_jsonl, 'a') as val_file:
    for record in remaining_data:
        val_file.write(json.dumps(record) + '\n')

print(f"Files saved as test_jsonl and val_jsonl with 440 records per category.")
