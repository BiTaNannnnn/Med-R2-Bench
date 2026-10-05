import json


def replace_key_in_jsonl(file_path, old_key, new_key):
    # 读取文件内容
    with open(file_path, 'r') as infile:
        lines = infile.readlines()

    # 处理每一行的内容
    modified_lines = []
    for line in lines:
        # 解析 JSON 数据
        data = json.loads(line)

        # 如果 old_key 存在，根据条件替换为 new_key
        if old_key in data:
            wrong_box_value = data[old_key]

            if wrong_box_value == 'no':
                # 如果是 'no'，根据某个逻辑生成路径，这里假设是一个固定的逻辑
                # 使用 BoxImg_savepath 作为新的 image_paths 的值
                data[new_key] = data.get('BoxImg_savepath', '')
            else:
                # 如果不是 'no'，则 image_paths 直接等于 wrong_box_save_path 的值
                data[new_key] = wrong_box_value

        # 如果 'milti_box_save_path' 存在，直接将键名改为 new_key
        if "milti_box_save_path" in data:
            data[new_key] = data.pop("milti_box_save_path")

        # 如果 'box_path' 存在，直接将键名改为 new_key
        if "box_path" in data:
            data[new_key] = data.pop("box_path")

        # 将修改后的数据转为 JSON 格式并添加到列表
        modified_lines.append(json.dumps(data) + '\n')

    # 覆盖原文件，保存修改后的内容
    with open(file_path, 'w') as outfile:
        outfile.writelines(modified_lines)



def replace_task_name_in_jsonl(file_path):
    # 读取文件内容
    with open(file_path, 'r') as infile:
        lines = infile.readlines()

    # 处理每一行的内容
    modified_lines = []
    for line in lines:
        # 解析 JSON 数据
        data = json.loads(line)

        if 'multi_box' in data.get('type', ''):
            data["task"] = "lesion_multi_box"
            data.pop("type", None)  # Remove 'type' field if it exists

        # 如果 'xxx' 存在，直接将键名改为 new_key
        if 'asked_region' in data and data.get('task') == "lesion_recognition":
            data["task"] = "bodypart_recognition"



        # 将修改后的数据转为 JSON 格式并添加到列表
        modified_lines.append(json.dumps(data) + '\n')

    # 覆盖原文件，保存修改后的内容
    with open(file_path, 'w') as outfile:
        outfile.writelines(modified_lines)

# 使用示例
# file_path = 'xxx/path'  # 替换为实际的文件路径

file_path = 'xxx/path'

'''统一 image_paths'''
replace_key_in_jsonl(file_path, 'wrong_box_save_path', 'image_paths')

'''分开task 名字'''
replace_task_name_in_jsonl(file_path)