import json
import random

def shuffle_yes_no_options(input_jsonl, output_jsonl, seed=42):
    random.seed(seed)

    with open(input_jsonl, 'r', encoding='utf-8') as fin, \
         open(output_jsonl, 'w', encoding='utf-8') as fout:

        for line in fin:
            data = json.loads(line)

            # ===== 新增安全检查 =====
            # ===== 情况 1：organ_eraser_identification，原样写出 =====
            if data.get("task") == "organ_eraser_identification":
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            # ===== 情况 2：没有 options，原样写出 =====
            if "options" not in data:
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            options = data["options"]          # ["A. yes", "B. no"]
            answer = data["answer"]            # e.g. "A. yes"

            # 解析原始选项
            parsed = []
            for opt in options:
                label, text = opt.split('.', 1)
                parsed.append(text.strip().lower())  # ["yes", "no"]

            # 当前正确答案的语义
            correct_text = answer.split('.', 1)[1].strip().lower()

            # 随机打乱 yes/no
            random.shuffle(parsed)

            # 重新生成 options 和 answer
            new_options = []
            new_answer = None
            labels = ['A', 'B']

            for label, text in zip(labels, parsed):
                option_str = f"{label}. {text}"
                new_options.append(option_str)
                if text == correct_text:
                    new_answer = option_str

            # 写回
            data["options"] = new_options
            data["answer"] = new_answer

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            
            
            
if __name__ == "__main__":
    input_jsonl = "xxx/path"  # 你的 region 标注 JSON
    output_jsonl = "xxx/path"
    shuffle_yes_no_options(input_jsonl, output_jsonl, seed=42)