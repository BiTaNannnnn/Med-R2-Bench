"""
制作 add_noise 的json vqa 结果
"""


import csv
import json
import os
import random

def find_nii(file_name, mainPath):
    """
    根据文件名生成路径，返回完整路径
    """
    subfolder1 = file_name.split('.')[0]  # valid_1、valid_2 等
    file_end = '.png'
    niiPath = os.path.join(mainPath, subfolder1 + file_end)
    # 返回生成的路径
    return niiPath

def generate_json_from_csv(csv_file, mainPath, output_file):
    # 10 种不同问法
    question_variants = [
        "Among these two medical images, one contains an added artifact. Can you identify which one shows the artifact?",
        "Out of these two images, one has an artifact added. Can you figure out which one it is?",
        "Which of these two images displays the added artifact?",
        "Looking at these two images, can you tell which one contains the artifact?",
        "In this pair of medical images, one has an artifact. Can you spot which one shows it?",
        "Can you identify which image has the artifact added, among these two?",
        "In the two images provided, one contains an artifact. Can you determine which one?",
        "Can you figure out which of these two medical images displays the artifact?",
        "Of the two images shown, which one has the artifact added to it?",
        "Here are two medical images. One of them contains an added artifact. Which one shows the artifact?"
    ]

    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行

        with open(output_file, mode='a') as json_file:  # 打开输出文件进行写入
            for row in reader:
                file_name = row[0]
                image_path = find_nii(file_name, mainPath)

                if os.path.exists(image_path):
                    # 随机选项
                    left_option = "left"
                    right_option = "right"

                    if random.random() < 0.5:
                        options = [f"A. {left_option}", f"B. {right_option}"]
                        answer = "A. left"
                    else:
                        options = [f"A. {right_option}", f"B. {left_option}"]
                        answer = "B. left"

                    # 随机选择问题
                    question = random.choice(question_variants)

                    # 单个 JSON 对象
                    json_data = {
                        "unique_id": file_name,
                        "image_paths": image_path,
                        "question": question,
                        "options": options,
                        "answer": answer,
                        "modality": "CT",
                        "task": "artifact_recognition",
                        "options_num": 2,
                        "type": "normal_question",
                    }

                    # 每次写入一个 JSON 对象，保持 jsonl 格式
                    json_file.write(json.dumps(json_data) + '\n')

    print(f"Saved records to {output_file}")



def generate_json_from_csv2(csv_file, mainPath, output_file):
    # 10 种不同问法
    question_variants = [
        # "Here are two medical images. One of them has added artifact. Is the left image added artifact?",
        "Here are two medical images. One of them has added artifact. Is the right image added artifact?"
    ]

    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行

        with open(output_file, mode='a') as json_file:  # 打开输出文件进行写入
            for row in reader:
                file_name = row[0]
                image_path = find_nii(file_name, mainPath)

                if os.path.exists(image_path):
                    # 随机选择一个问题
                    question_id = 0  # 假设这里只选一个问题
                    question = question_variants[question_id]

                    # 随机决定 A 或 B 为 'no'，另外一个为 'yes'
                    answer_option = random.choice(["A", "B"])  # 随机选择 A 或 B
                    if answer_option == "A":
                        options = ["A. no", "B. yes"]  # A 为 no，B 为 yes
                        answer = "A. no"
                    else:
                        options = ["A. yes", "B. no"]  # A 为 yes，B 为 no
                        answer = "B. no"

                    # 创建一个字典来存储每条数据
                    json_data = {
                        "unique_id": file_name,
                        "image_paths": image_path,
                        "question": question,
                        "options": options,  # 假设这是两个选项
                        "answer": answer,  # 答案是 no
                        "modality": "CT",
                        "task": "artifact_recognition",
                        "options_num": 2,
                        "type": "wrong_question"
                    }

                    # 将每个JSON对象写入文件，保持jsonl格式，每行一个JSON对象
                    json_file.write(json.dumps(json_data) + '\n')

    print(f"Saved records to {output_file}")



def generate_json_from_csv3(csv_file, mainPath, output_file):
    # 10 种不同问法
    question_variants = [
        "Here are two medical images. One of them has added artifact. Is the left image added artifact?",
        # "Here are two medical images. One of them has added artifact. Is the right image added artifact?"
    ]

    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行

        with open(output_file, mode='a') as json_file:  # 打开输出文件进行写入
            for row in reader:
                file_name = row[0]
                image_path = find_nii(file_name, mainPath)

                if os.path.exists(image_path):
                    # 随机选择一个问题
                    question_id = 0  # 假设这里只选一个问题
                    question = question_variants[question_id]

                    # 随机决定 A 或 B 为 'no'，另外一个为 'yes'
                    answer_option = random.choice(["A", "B"])  # 随机选择 A 或 B
                    if answer_option == "A":
                        options = ["A. no", "B. yes"]  # A 为 no，B 为 yes
                        answer = "B. yes"
                    else:
                        options = ["A. yes", "B. no"]  # A 为 yes，B 为 no
                        answer = "A. yes"

                    # 创建一个字典来存储每条数据
                    json_data = {
                        "unique_id": file_name,
                        "image_paths": image_path,
                        "question": question,
                        "options": options,  # 假设这是两个选项
                        "answer": answer,  # 答案是 no
                        "modality": "CT",
                        "task": "artifact_recognition",
                        "options_num": 2,
                        "type": "correct_question"
                    }

                    # 将每个JSON对象写入文件，保持jsonl格式，每行一个JSON对象
                    json_file.write(json.dumps(json_data) + '\n')

    print(f"Saved records to {output_file}")

if __name__ == "__main__":
    # 调用函数，指定 CSV 文件和输出 JSON 文件夹
    csv_file = 'xxx/path'  # 你的 CSV 文件路径
    mainPath = 'xxx/path'  # 主路径
    output_folder = 'xxx/path'  # 生成的 JSON 文件保存的文件夹

    # normal-question
    generate_json_from_csv(csv_file, mainPath, output_folder)

    # wrong—question
    generate_json_from_csv2(csv_file, mainPath, output_folder)

    # correct-question
    generate_json_from_csv3(csv_file, mainPath, output_folder)