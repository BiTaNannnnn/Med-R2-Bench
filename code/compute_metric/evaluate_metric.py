import os
# 优先使用国内镜像，避免首次下载卡死
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import evaluate
import json
import argparse
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# 命令行参数解析
parser = argparse.ArgumentParser(description="Evaluate DrVD-Bench tasks")     # joint_qa  # Level1  # qwen2.5vl Hulu4b
parser.add_argument("--RESULT_ROOT", default="xxx/path",
                    help="根目录，用于拼接 <RESULT_ROOT>/<task>/... 路径")
parser.add_argument("--GPU", default="6", help="GPU id, int or 'cpu'")

args = parser.parse_args()

path_dir = os.path.dirname(os.path.normpath(args.RESULT_ROOT))
Level = os.path.basename(os.path.normpath(args.RESULT_ROOT))
Level = Level.split('.')[0]



# 定义任务及其子任务
TASK = {
    'noise_recognition': ['wrong_question', 'correct_question', 'normal_question'],
    'artifact_recognition': ['wrong_question', 'correct_question', 'normal_question'],
    'organ_location': ['wrong_color-correct_region', 'correct_color-correct_region', "normal_question"],
    'organ_recognition': ['wrong_box-correct_Q', 'correct_box-correct_Q', 'correct_box_wrong_Q'],
    'organ_eraser_identification': ['organ_eraser_identification'],
    'bodypart_recognition': ['wrong_box-correct_Q', 'correct_box-correct_Q', 'correct_box-wrong_Q'],
    'lesion_recognition': ['correct_box-correct_Q', 'wrong_box-correct_Q', 'correct_box-wrong_Q'],
    'lesion_multi_box': ['lesion_multi_box'],
    'lesion_size': ['wrong_size-wrong_Q', 'correct_size-correct_Q'],
    'box_report': ['add_box', 'no_box'],
    'noise_report': ['correct_box_no_noise', 'correct_box_add_noise'],
    'artifact_report': ['correct_box_no_artifact', 'correct_box_add_artifact'],
    'joint_qa':['joint_qa']
}

# 统计任务正确率
task_stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))


# 评估开放任务子任务
def evaluate_open_subtask(GT, model_pred ,bertscore, rouge):


    preds = [model_pred]
    answers = [[GT]],

    # Bleu
    smooth = SmoothingFunction().method4
    bleu_result = sentence_bleu([GT.split()],  # 二维：每条参考是一个列表
                          model_pred.split(),  # 一维：预测分词
                          smoothing_function=smooth)


    rouge_result = rouge.compute(predictions=preds, references=[[a] for a in answers])
    rouge_result_rouge1 = rouge_result["rouge1"]

    bert_result = bertscore.compute(predictions=preds, references=[[a] for a in answers], lang="en")
    bert_result_f1 = bert_result["f1"][0]
    return {"bleu": bleu_result, "rouge1": rouge_result_rouge1, "f1": bert_result_f1}


# 评估闭合任务子任务
def evaluate_close_subtask( GT, model_pred):
    answer = GT
    model_result = model_pred
    correct_option = answer.split('.')[0].strip()

    correct = 1 if model_result == correct_option else 0
    total = 1

    return {"correct": correct, "total": total}


# 计算每个任务的准确率（开放任务）

def evaluate_open_task(task_list, task, task_type):
    task_results = {
        "bleu": [],
        "rouge1": [],
        "f1": [],
        "total": 0  # 添加 total 键
    }

    # === 1. 提到循环外 ===

    bertscore = evaluate.load('bertscore')
    rouge = evaluate.load('rouge')

    # 遍历该任务下的所有子任务

    for item in task_list:
        # 获取每个条目的真实答案、模型预测结果以及元数据
        GT = item.get('GT_ANSWER', '')
        model_pred = item.get('text', '').strip()

        # 调用 evaluate_open_subtask 函数获取指标
        result = evaluate_open_subtask(GT, model_pred ,bertscore, rouge)

        # 将每个子任务的结果保存到 task_results
        task_results["bleu"].append(result["bleu"])
        task_results["rouge1"].append(result["rouge1"])
        task_results["f1"].append(result["f1"])
        task_results["total"] += 1

    # 计算各项指标的平均值
    bleu = np.mean(task_results["bleu"]) if task_results["bleu"] else 0

    rouge1 = np.mean(task_results["rouge1"]) if task_results["rouge1"] else 0

    f1 = np.mean(task_results["f1"]) if task_results["f1"] else 0
    total = task_results["total"]


    # 输出保存结果
    results = {
        "task": task,
        "task_type": task_type,
        "bleu": bleu,
        "rouge1": rouge1,
        "f1": f1,
        'total' : total
    }



    return results


# 计算每个任务的准确率（闭合任务）
def evaluate_close_task(task_list, task, task_type):
    task_results = {
        "task": task,
        "task_type": task_type,
        "correct": 0,  # 正确的数量
        "total": 0,  # 总数
        "acc": 0.0,  # 正确率
    }

    def get_is_correct(item) -> bool:
        """
        根据模型输出与正确答案对比，判断是否正确。
        """
        # 获取模型输出和正确答案
        model_result = item.get('text', '').strip()
        answer = item.get('GT_ANSWER', '')

        if not model_result or not answer:
            return False

        # 正确答案的选项
        correct_option = answer.split('.')[0].strip()

        # 如果模型的输出不为空，则比较
        if model_result:
            # 取模型输出的第一个字符（你提到的 model_result[0]）
            model_result = model_result[0]

        # 判断模型输出是否等于正确选项
        return model_result == correct_option

    # 遍历 task_list，统计正确和总数
    for item in task_list:
        task_results["total"] += 1
        if get_is_correct(item):
            task_results["correct"] += 1

    # 计算正确率
    if task_results["total"] > 0:
        task_results["acc"] = task_results["correct"] / task_results["total"]

    return task_results


def ecaluate_join_qa(task_list, task, task_type):
    task_results = {
        "task": task,
        "task_type": task_type,
        "correct": 0,  # 正确的数量
        "total": 0,  # 总数
        "acc": 0.0,  # 正确率
    }

    # def get_correct_score(item) -> float:
    #     """
    #     根据模型输出与正确答案对比，判断正确数量，返回得分。
    #     """
    #     # 获取模型输出和正确答案
    #     model_result = item.get('text', '').strip()
    #     answer = item.get('GT_ANSWER', '')
    #
    #     # 提取答案部分
    #     correct_option = answer.split('.')[0].strip()  # 正确答案的选项
    #
    #     # 如果模型输出或答案为空，返回得分 0
    #     if not model_result or not answer:
    #         return 0.0
    #
    #     # 将模型输出的第一个字符与正确答案进行比较
    #     model_result = model_result[0]  # 假设只取第一个字符
    #
    #     # 判断正确答案与模型输出是否匹配
    #     return 1.0 if model_result == correct_option else 0.0

    def get_total_score(item) -> float:
        """
        计算该条数据的得分，根据多个答案的正确性来计算。
        假设每条数据有四个答案项，每个答案项正确得 0.25，总分为 1。
        """
        # 初始化得分
        score = 0.0

        # 比较四个答案的正确性，累加得分
        correct_count = 0
        for key in ["modality_answer", "organ_answer", "lesion_answer", "diagnosis_answer"]:
            model_answer = item.get(f"model_{key}", "")
            gt_answer = item.get(f"GT_{key}", "")
            answer_id = gt_answer.split('.')[0]

            # 计算该答案的正确性
            if model_answer and gt_answer and model_answer.strip() == answer_id.strip():
                correct_count += 1

        # 根据正确答案的数量来计算得分
        if correct_count == 1:
            score = 0.25
        elif correct_count == 2:
            score = 0.5
        elif correct_count == 3:
            score = 0.75
        elif correct_count == 4:
            score = 1.0

        return score

    # 遍历 task_list，统计正确和总数
    for item in task_list:
        task_results["total"] += 1
        score = get_total_score(item)  # 获取该条数据的得分
        task_results["correct"] += score

    # 计算正确率
    if task_results["total"] > 0:
        task_results["acc"] = task_results["correct"] / task_results["total"]

    return task_results

# 评估每个任务（任务的总正确率）
def evaluate_task(task_list, task, task_type, Level):
    if task in ['lesion_multi_box', 'lesion_size', 'box_report', 'noise_report', 'artifact_report']:
        task_results = evaluate_open_task(task_list, task, task_type)

    elif Level == 'joint_qa':
        task_results = ecaluate_join_qa(task_list, task, task_type)

    else:
        task_results = evaluate_close_task(task_list, task, task_type)
    return task_results


if __name__ == '__main__':
    # 存储任务和子任务
    # 1. 创建 (task, category) 双层 key 的 defaultdict
    task_type_dict = defaultdict(list)
    all_task_results = []  # 用来存储所有任务类型的评估结果

    # 读取JSONL文件并按任务分类子任务
    with open(args.RESULT_ROOT, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            metadata = item.get('metadata', {})
            task = metadata.get("task", "Unknown")
            type = metadata.get("type")

            # 只保留 TASK 字典里出现过的 task & category
            if task in TASK:
                task_type_dict[(task, type)].append(item)
            else:
                task_type_dict[('joint_qa', '4questions')].append(item)
                task = 'joint_qa'
                type = '4questions'

        # 遍历 每一个list，调用 evaluate 方法
        for (task, task_type), task_list in task_type_dict.items():
            print(f"Processing {task} with type {task_type}")

            task_results = evaluate_task(task_list, task, task_type,Level)

            all_task_results.append(task_results)

            print(f"Finished {task} with type {task_type}")

        # 创建目录路径
        result_path = os.path.join(path_dir, Level + "_accrate.jsonl")
        # 将结果保存到文件
        with open(result_path, 'w', encoding='utf-8') as result_file:
            for task_result in all_task_results:
                # 对每个 task_result 写入一行 JSON 对象
                json.dump(task_result, result_file, ensure_ascii=False)
                result_file.write('\n')  # 每个结果后写入换行符，确保 JSONL 格式



        print(f"All results saved to {result_path}")




# BLEU 看 n-gram 精确匹配 → 预测里完全没有 "red" 的任何 1-gram，所以 0。
# ROUGE-1 看召回 → 参考词 "red" 在预测里一次都没出现，所以也是 0。
# BERTScore 走的是 语义向量（PubMedBERT 768 维）：
# "blue and green a little bit" 与 "red" 在颜色语义空间仍有微弱关联（都谈颜色），余弦相似度≈0.79，
# 因此 BERTScore-F1 给出 0.79，不是错误，而是「语义有点靠边」的体现。

