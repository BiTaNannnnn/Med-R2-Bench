# Med-R2: An Adversarial Benchmark for Evidence-Grounded Reasoning in Medical VLMs

This repository is a cleaned release package prepared from the ACL-2026 working directory for the EMNLP 2026 paper:

**Med-R2: An Adversarial Benchmark for Evidence-Grounded Reasoning in Medical VLMs**

## What Is Included

- `data/test/`: public JSONL test splits with private image paths redacted.
- `data/train/`: public JSONL train splits with private image paths redacted.
- `code/DataProcess/`: benchmark construction scripts, cleaned for public release.
- `code/compute_metric/`: evaluation scripts for multiple-choice tasks and clinical-report tasks.
- `code/compare_models/`: cleaned top-level model-evaluation scripts from the original experiments.
- `docs/Level1-5_Promt.md`: task prompt description from the original working directory.
- `docs/release_audit.md`: preparation notes and excluded/private items.

## Data Note

The original server-local `image_paths` values were intentionally redacted to:

```text
xxx/path/to/<image-file-name>
```

Before running the benchmark, replace the redacted prefix with your local image root or adapt the data loader to resolve image files from your own storage layout.

## Security / Privacy Note

This release package removes model weights, images, checkpoints, raw predictions, cache directories, `.git` folders from third-party repositories, local usernames, server paths, and hard-coded API keys. API-based scripts use placeholder values such as `xxx`; set credentials through environment variables or command-line arguments in your private runtime environment.

## Basic Evaluation

Multiple-choice style tasks can be evaluated with:

```bash
python code/compute_metric/evaluate_metric.py --help
```

Clinical report generation can be evaluated with:

```bash
python code/compute_metric/report_evaluate.py --help
```

Some scripts were originally written for a private cluster and may require path/model arguments to be adapted before use.

## Citation

```bibtex
@inproceedings{medr2-2026,
  title = {Med-R2: An Adversarial Benchmark for Evidence-Grounded Reasoning in Medical VLMs},
  author = {Anonymous},
  booktitle = {Proceedings of EMNLP},
  year = {2026}
}
```
