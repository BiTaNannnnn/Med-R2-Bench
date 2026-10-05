import json
import random
import json
import random
import re

def yn_semantic(s: str):
    """
    从选项/答案文本中抽取 yes/no 语义。
    适配: "Yes, ...", "No, ...", "yes", "no" 等。
    """
    if not s:
        return None
    s = s.strip().lower()
    # 去掉前缀 "A." / "B." 等（如果传进来的是整段）
    s = re.sub(r'^[a-z]\.\s*', '', s)
    if s.startswith('yes'):
        return 'yes'
    if s.startswith('no'):
        return 'no'
    return None

def shuffle_yes_no_options(input_jsonl, output_jsonl, seed=42):
    random.seed(seed)

    with open(input_jsonl, 'r', encoding='utf-8') as fin, \
         open(output_jsonl, 'w', encoding='utf-8') as fout:

        for line in fin:
            data = json.loads(line)

            # 跳过不需要处理的 task：原样写出
            if data.get("task") in {"lesion_multi_box", "lesion_size"}:
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            # 没有 options：原样写出
            if "options" not in data or not data.get("options") or not data.get("answer"):
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            options = data["options"]
            answer = data["answer"]

            # 解析 options：得到 [(semantic, full_text_original_lower), ...]
            parsed = []
            for opt in options:
                if '.' not in opt:
                    parsed = None
                    break
                _, text = opt.split('.', 1)
                text_clean = text.strip()
                sem = yn_semantic(text_clean)
                parsed.append((sem, text_clean))  # 保留原句（不强制lower，避免你想保留大小写可改）

            if parsed is None or len(parsed) != 2:
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            # 只处理真正的 yes/no 二选一
            sems = {p[0] for p in parsed}
            correct_sem = yn_semantic(answer.split('.', 1)[1] if '.' in answer else answer)

            if sems != {'yes', 'no'} or correct_sem not in {'yes', 'no'}:
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            # 打乱顺序（打乱的是整条 option 的文本）
            random.shuffle(parsed)

            # 重新生成 options 和 answer（根据 semantic 对齐）
            labels = ['A', 'B']
            new_options = []
            new_answer = None

            for label, (sem, text_clean) in zip(labels, parsed):
                option_str = f"{label}. {text_clean}"
                new_options.append(option_str)
                if sem == correct_sem:
                    new_answer = option_str

            # 最后保险：如果没找到答案，原样写出，绝不写 null
            if new_answer is None:
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            data["options"] = new_options
            data["answer"] = new_answer

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    input_jsonl = "xxx/path"  # 你的 region 标注 JSON
    output_jsonl = "xxx/path"
    shuffle_yes_no_options(input_jsonl, output_jsonl, seed=42)