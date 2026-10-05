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
    with open(output_file, mode='a') as json_file:
        # 10 种不同问法
        question_variants = [
            "Out of these two medical images, one contains added noise. Can you identify which one?",
            "Among the two images, one has been modified with noise. Can you figure out which one?",
            "Which of these two images shows the added noise?",
            "Looking at these two images, can you tell which one has noise added?",
            "In this pair of medical images, one has been altered with noise. Can you spot which one?",
            "Can you identify which image has the noise added, among these two?",
            "In the two images provided, one has noise. Can you identify the noisy image?",
            "Can you figure out which of these two medical images has noise added?",
            "Of the two images shown, which one has been edited with added noise?",
            "Here are two medical images. One of them has added noise. Which one is the noised image?"
        ]

        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过标题行
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

                    # 随机选一个问法
                    question = random.choice(question_variants)

                    # 生成JSON对象
                    json_data = {
                        "unique_id": file_name,
                        "image_paths": image_path,
                        "question": question,
                        "options": options,
                        "answer": answer,
                        "modality": "CT",
                        "task": "noise_recognition",
                        "options_num": 2,
                        "type": "normal_question"
                    }

                    # 将每个JSON对象写入文件，保持jsonl格式，每行一个JSON对象
                    json_file.write(json.dumps(json_data) + '\n')

    print(f"Saved JSONL records to {output_file}")


def generate_json_from_csv2(csv_file, mainPath, output_file):
    with open(output_file, mode='a') as json_file:
        # 10 种不同问法
        question_variants = [
            # "Here are two medical images. One of them has added noise. Is the left image added noise?",
            "Here are two medical images. One of them has added noise. Is the right image added noise?"
        ]

        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过标题行

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
                    # 生成JSON对象
                    json_data = {
                        "unique_id": file_name,
                        "image_paths": image_path,
                        "question": question,
                        "options": options,  # 假设这是两个选项
                        "answer": answer,
                        "modality": "CT",
                        "task": "noise_recognition",
                        "options_num": 2,
                        "type": "wrong_question"
                    }

                    # 将每个JSON对象写入文件，保持jsonl格式，每行一个JSON对象
                    json_file.write(json.dumps(json_data) + '\n')

    print(f"Saved JSONL records to {output_file}")


def generate_json_from_csv3(csv_file, mainPath, output_file):
    with open(output_file, mode='a') as json_file:
        # 10 种不同问法
        question_variants = [
            "Here are two medical images. One of them has added noise. Is the left image added noise?",
            # "Here are two medical images. One of them has added noise. Is the right image added noise?"
        ]

        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过标题行

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
                    # 生成JSON对象
                    json_data = {
                        "unique_id": file_name,
                        "image_paths": image_path,
                        "question": question,
                        "options": options,  # 假设这是两个选项
                        "answer": answer,
                        "modality": "CT",
                        "task": "noise_recognition",
                        "options_num": 2,
                        "type": "correct_question"
                    }

                    # 将每个JSON对象写入文件，保持jsonl格式，每行一个JSON对象
                    json_file.write(json.dumps(json_data) + '\n')

    print(f"Saved JSONL records to {output_file}")



if __name__ == "__main__":
    # 调用函数，指定 CSV 文件和输出 JSON 文件路径
    csv_file = 'xxx/path'  # CSV 文件路径
    mainPath = 'xxx/path'  # 主路径
    output_file = 'xxx/path'  # 生成的 JSON 文件路径

    # normal——question
    generate_json_from_csv(csv_file, mainPath, output_file)

    # wrong—question
    generate_json_from_csv2(csv_file, mainPath, output_file)

    # correct——question
    generate_json_from_csv3(csv_file, mainPath, output_file)

