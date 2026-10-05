import json

# 输入文件路径
file_path = "xxx/path"

# 读取并修改文件内容
with open(file_path, "r+", encoding="utf-8") as file:
    # 读取所有行
    lines = file.readlines()

    # 移动文件指针到文件开始位置
    file.seek(0)

    # 遍历文件中的每一行
    for line in lines:
        # 解析每一行的 JSON 数据
        item = json.loads(line.strip())
        metadata = item.get('metadata', {})
        task = metadata.get("task", "Unknown")
        type = metadata.get("type")
        answer = metadata.get("answer")

        if "GT_ANSSWER" in item:
            item["GT_ANSWER"] = item.pop("GT_ANSSWER")


            # # 检查 'type' 字段，如果是 'correct_question'，则修改为 'normal_question'
        # if type == 'correct_question':
        #     metadata['type'] = 'normal_question'
        #
        # if type == 'wrong_question':
        #     if answer == 'A. yes':
        #         metadata['type'] = 'correct_question'



        # 将修改后的数据写回到文件
        json.dump(item, file, ensure_ascii=False)
        file.write("\n")  # 每行结束后写入换行符

    # 截断文件内容，去除原始文件中剩余的部分
    file.truncate()

print(f"文件 {file_path} 已成功修改！")
