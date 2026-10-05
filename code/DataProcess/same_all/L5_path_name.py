import json


def replace_key_in_jsonl(file_path,  new_key):
    # 读取文件内容
    with open(file_path, 'r') as infile:
        lines = infile.readlines()

    # 处理每一行的内容
    modified_lines = []
    for line in lines:
        # 解析 JSON 数据
        data = json.loads(line)

        # 如果 old_key 存在，根据条件替换为 new_key


        # 如果 'milti_box_save_path' 存在，直接将键名改为 new_key
        if "correct_img_path" in data:
            data[new_key] = data.pop("correct_img_path")

        # 如果 'box_path' 存在，直接将键名改为 new_key
        if "wrong_img_path" in data:
            data[new_key] = data.pop("wrong_img_path")

        if "img_path" in data:
            data[new_key] = data.pop("img_path")

        # 将修改后的数据转为 JSON 格式并添加到列表
        modified_lines.append(json.dumps(data) + '\n')

    # 覆盖原文件，保存修改后的内容
    with open(file_path, 'w') as outfile:
        outfile.writelines(modified_lines)


# 使用示例
# file_path = 'xxx/path'  # 替换为实际的文件路径
file_path = 'xxx/path'
replace_key_in_jsonl(file_path,  'image_paths')