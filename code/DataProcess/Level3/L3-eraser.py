
'''
json 文件中
模态 = ['Ultrasound', 'Magnetic_Resonance_Imaging', 'CT', 'Radiograph', 'Pathology']
      ['超声', '磁共振成像', 'CT', 'X光片', '病理学']


'''




import json
import os
import random


def find_json(json_Path):


    # 用一个列表来存储所有的task
    tasks = set()

    # 逐行读取文件并解析每行的JSON数据
    with open(json_Path, 'r') as f:
        for line in f:
            # 解析每一行作为一个独立的JSON对象
            try:
                data = json.loads(line.strip())  # 使用strip()去掉换行符
                tasks.add(data['modality'])  # 将task添加到集合中，自动去重
            except json.JSONDecodeError:
                print("Error parsing line:", line)

    # 将集合转换为列表并输出去重后的task列表
    tasks = list(tasks)
    print(tasks)

    # 'artifact_recognition'# 伪影
    # 'organ_recognition'


def filter_jsonl(in_path, out_path):
    modalities = {"Magnetic_Resonance_Imaging", "CT", "Radiograph"}
    tasks = {
        "organ_eraser_identification",
    }

    question_variants = [
        "In this pair of CT images, the original scan is on the left and the AI-edited version is on the right. Can you identify the organ that has been altered or removed in the right image?",
        "Looking at the two CT images, with the left being the original and the right the AI-modified, can you determine which organ has been edited or removed in the right image?",
        "The left CT image shows the original scan, and the right one shows the AI-modified version. Can you recognize which organ has been edited or removed in the right image?",
        "Can you spot which organ has been edited or removed in the AI-modified CT image on the right, compared to the original scan on the left?",
        "In the given pair of CT images, the original scan is on the left and the AI-modified version is on the right. Which organ is missing or altered in the right image?",
        "The left CT image is the original, and the right one is AI-modified. Can you identify the organ that has been removed or changed in the right image?",
        "In the pair of CT images, with the original on the left and the AI-edited on the right, which organ appears to have been removed or altered in the right image?",
        "Given these two CT images, with the original on the left and the AI-modified on the right, can you identify which organ has been edited or removed in the right image?",
        "The left CT image is unaltered, and the right image has been modified by AI. Can you recognize which organ has been edited or removed in the right image?",
        "Can you determine which organ has been edited or removed in the right image of this CT pair, where the left image is the original scan?",
        "This is a pair of CT images, with the original scan on the left and the AI-edited image on the right. Can you identify which type of organ has been removed or edited in the right image?"
    ]

    kept = 0
    total = 0

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            modality = obj.get("modality")
            task = obj.get("task")
            image_paths = obj.get("image_paths")

            # 第一步筛选
            if modality not in modalities:
                continue

            # 第二步筛选
            if task in tasks:
                # ✅ 随机替换 question
                obj["question"] = random.choice(question_variants)

                # 路径前缀处理
                obj["image_paths"] = make_full_image_path(image_paths)

                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1

    print(f"Total: {total}, Kept: {kept}, Output: {out_path}")



def make_full_image_path(p: str) -> str:
    BASE_PREFIX = "xxx/path"
    if not p:
        return p
    # 已是绝对路径则不修改
    if os.path.isabs(p):
        return p
    # 若已有指定前缀，也不重复添加
    if p.startswith((""+BASE_PREFIX, BASE_PREFIX + "/")):
        return p
    # 去掉可能的前导 ./ 再拼接
    p = p.lstrip("./")
    return os.path.normpath(os.path.join(BASE_PREFIX, p))

def append_noise_to_json_array(in_path: str, out_path: str):
    # 读已有 A.json（如果存在且是 JSON 数组）
    TEMPLATE_KEYS = [
        "unique_id",
        "image_paths",
        "question",
        "options",
        "answer",
        "modality",
        "task",
        "options_num",
    ]

    out_records = []
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                out_records = existing
            else:
                print(f"Warning: {out_path} 不是 JSON 数组，将覆盖重写。")
        except json.JSONDecodeError:
            print(f"Warning: {out_path} 不是有效 JSON，将覆盖重写。")

    added = 0
    total = 0
    with open(in_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # 跳过坏行
                continue

            if obj.get("task") != "noise_recognition":
                continue

            # 按模板键顺序重建；值来自输入行
            new_obj = {k: obj.get(k) for k in TEMPLATE_KEYS}
            # 这里加前缀
            new_obj["image_paths"] = make_full_image_path(new_obj.get("image_paths"))

            out_records.append(new_obj)
            added += 1

    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(out_records, fout, ensure_ascii=False, indent=4)

    print(f"扫描行数: {total}, 追加 noise_recognition: {added}, 输出: {out_path}, 总条数: {len(out_records)}")


def filter_and_write_jsonl(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)  # 解析每一行的 JSON 数据
            except json.JSONDecodeError:
                continue  # 如果无法解析 JSON，则跳过此行

            # 检查 task 和 modality 是否符合要求
            if item.get("task") == "organ_eraser_identification" and item.get("modality") in ["Magnetic_Resonance_Imaging", "CT"]:
                json.dump(item, outfile, ensure_ascii=False)
                outfile.write("\n")  # 每行写入一个 JSON 对象

    print(f"筛选后的数据已写入 {output_file_path}")

if __name__ == "__main__":
    json_Path = 'xxx/path'

    ''' 查看DrVD  json 文件内容'''
    # tasks = find_json(json_Path)

    '''筛选 出我要的 json文件'''
    my_json_path = 'xxx/path'

    # 筛选—+写入
    filter_and_write_jsonl(json_Path, my_json_path)

    # 做改变
    filter_jsonl(json_Path, my_json_path)



    ''''''