import json
import argparse
from collections import defaultdict

def calculate_single_metrics(jsonl_file):

    # model -> task -> stats
    stats = defaultdict(lambda: {
        'overall': {'correct': 0, 'total': 0},

        'organ_location': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'organ_recognition': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'organ_eraser_identification': {
            'overall': {'correct': 0, 'total': 0}
        }
    })

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)

            model = data.get('model_id')
            metadata = data.get('metadata', {})
            task = metadata.get('task')

            if task not in {
                'organ_location',
                'organ_recognition',
                'organ_eraser_identification'
            }:
                continue

            answer = data.get('GT_ANSSWER', '')
            model_result = data.get('text', '').strip()

            correct_option = answer.split('.')[0].strip()

            # normalize model output
            if model_result:
                model_result = model_result[0]

            # ===== overall (model-level) =====
            stats[model]['overall']['total'] += 1

            # ===== task-level overall =====
            stats[model][task]['overall']['total'] += 1

            # ===== type-level (only for two tasks) =====
            if task in {'organ_location', 'organ_recognition'}:
                base_type = metadata.get('type', 'unknown')
                question_type = metadata.get('question_type')

                if question_type == 'open_question':
                    q_type = 'open_question'
                else:
                    q_type = base_type

                stats[model][task][q_type]['total'] += 1

            # ===== correct =====
            if model_result == correct_option:
                stats[model]['overall']['correct'] += 1
                stats[model][task]['overall']['correct'] += 1

                if task in {'organ_location', 'organ_recognition'}:
                    stats[model][task][q_type]['correct'] += 1

    # print results
    for model, tasks in stats.items():
        print(f"Model: {model}")

        # ===== model overall =====
        overall = tasks['overall']
        acc = overall['correct'] / overall['total'] if overall['total'] else 0
        print(f"  Overall: {acc:.2%} ({overall['correct']}/{overall['total']})")

        # ===== per-task =====
        for task_name, task_stats in tasks.items():
            if task_name == 'overall':
                continue

            # task overall
            task_overall = task_stats['overall']
            task_acc = (
                task_overall['correct'] / task_overall['total']
                if task_overall['total'] else 0
            )

            print(f"  {task_name}: {task_acc:.2%} "
                  f"({task_overall['correct']}/{task_overall['total']})")

            # check if this task has type-level stats
            type_keys = [k for k in task_stats.keys() if k != 'overall']

            for t in type_keys:
                t_total = task_stats[t]['total']
                t_correct = task_stats[t]['correct']
                t_acc = t_correct / t_total if t_total else 0

                print(f"    {t}: {t_acc:.2%} ({t_correct}/{t_total})")

        print()



def main():
    parser = argparse.ArgumentParser(
        description='Compute choice metrics in single or joint mode.'
    )
    parser.add_argument(
        '--json_path', '-j',
        # Hulu - 4B
        default= 'xxx/path',

        required=False,
        help='Path to the JSONL results file.'
    )
    parser.add_argument(
        '--type', '-t', default = 'single', choices=['single', 'joint'], required=False,
        help="Type of metric to compute: 'single' for per-task, 'joint' for multi-level joint output."
    )
    args = parser.parse_args()

    if args.type == 'single':
        calculate_single_metrics(args.json_path)



if __name__ == '__main__':
    main()