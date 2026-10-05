# Med-R2 Bench

<p align="center">
  <b>Med-R2: An Adversarial Benchmark for Evidence-Grounded Reasoning in Medical VLMs</b>
</p>

<p align="center">
  <img alt="EMNLP 2026" src="https://img.shields.io/badge/EMNLP-2026%20Main%20Conference-blue">
  <img alt="Task" src="https://img.shields.io/badge/Task-Medical%20VLM%20Reasoning-green">
  <img alt="Data" src="https://img.shields.io/badge/Data-99%2C997%20QA%20Samples-orange">
  <img alt="Release" src="https://img.shields.io/badge/Release-Cleaned%20Public%20Package-purple">
</p>

Med-R2 Bench is a cleaned public release package for the EMNLP 2026 paper:

> **Med-R2: An Adversarial Benchmark for Evidence-Grounded Reasoning in Medical VLMs**

The benchmark is designed to evaluate whether medical vision-language models can reason from visual evidence rather than relying on superficial language priors. It includes multi-level adversarial QA, evidence-grounded image understanding, and clinical-report generation tasks.

---

## ✨ Highlights

| Feature | Description |
| --- | --- |
| 🧠 Evidence-grounded reasoning | Tests whether models can answer by grounding decisions in image evidence. |
| 🧩 Multi-level benchmark | Covers image quality, anatomy, lesion-level reasoning, joint QA, and report generation. |
| ⚔️ Adversarial design | Includes positive, neutral, negative, wrong-box, erased-region, and interference-style cases. |
| 📊 Train/test splits | Provides cleaned JSONL train and test splits for reproducible benchmarking. |
| 🔐 Public-safe release | Removes private image paths, model weights, raw images, checkpoints, logs, and API keys. |

---

## 📦 Repository Layout

```text
Med-R2-Bench/
├── data/
│   ├── train/                 # Redacted training JSONL splits
│   ├── test/                  # Redacted testing JSONL splits
│   └── README.md
├── code/
│   ├── DataProcess/           # Benchmark construction scripts
│   ├── compute_metric/        # Evaluation scripts
│   └── compare_models/        # Cleaned model-evaluation entry scripts
├── docs/
│   ├── Level1-5_Promt.md      # Original level/task prompt notes
│   ├── manifest.json          # Split statistics and release manifest
│   └── release_audit.md       # What was included, redacted, and excluded
├── README.md
└── .gitignore
```

---

## 🚀 Quick Start

Clone the repository:

```bash
git clone https://github.com/BiTaNannnnn/Med-R2-Bench.git
cd Med-R2-Bench
```

Inspect the released data:

```bash
head -n 1 data/test/visual_evidence_Level1.jsonl
cat docs/manifest.json
```

Run metric scripts:

```bash
python code/compute_metric/evaluate_metric.py --help
python code/compute_metric/report_evaluate.py --help
```

---

## 📊 Released Data Splits

| Split | Level1 | Level3 | Level4 | Level5 | Joint QA | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 🧪 Test | 2,640 | 7,316 | 11,140 | 600 | 500 | 22,196 |
| 🏋️ Train | 5,184 | 14,152 | 25,557 | 30,320 | 2,588 | 77,801 |
| ⭐ Total | 7,824 | 21,468 | 36,697 | 30,920 | 3,088 | 99,997 |

All samples are stored as JSONL files. Each line is one benchmark example with fields such as `question`, `options`, `answer`, `task`, `type`, `modality`, and a redacted `image_paths` field.

---

## 🧭 Task Overview

| Level | Focus | Example Capabilities Tested |
| --- | --- | --- |
| Level1 | 🖼️ Image quality reasoning | Noise recognition, artifact recognition, quality perturbation detection |
| Level3 | 🫀 Anatomy-level reasoning | Organ recognition, organ location, erased-region / occlusion reasoning |
| Level4 | 🔍 Lesion-level reasoning | Body-part recognition, lesion recognition, multi-box reasoning, lesion size |
| Level5 | 📝 Clinical report generation | Report generation under box/noise/artifact interference |
| Joint QA | 🧠 Multi-step visual reasoning | Sequential medical reasoning across modality, organ, abnormality, and evidence |

For the original prompt notes, see [`docs/Level1-5_Promt.md`](docs/Level1-5_Promt.md).

---

## 🧪 Evaluation

Multiple-choice tasks can be evaluated with:

```bash
python code/compute_metric/evaluate_metric.py --help
```

Clinical-report tasks can be evaluated with:

```bash
python code/compute_metric/report_evaluate.py --help
```

The `code/compare_models/` directory contains cleaned top-level scripts used for baseline model evaluation. These scripts were originally written for a private cluster, so you may need to adapt model paths, environment names, GPU settings, and image-root resolution before running them.

---

## 🖼️ Image Path Note

The original server-local image paths were intentionally redacted. In this public release, image fields look like:

```text
xxx/path/to/<image-file-name>
```

Before running evaluation, replace the `xxx/path/to/` prefix with your local image root or modify your dataloader to resolve images from your storage layout.

---

## 🔐 Public Release Audit

This repository is a cleaned release package. The following private or heavyweight artifacts were excluded:

| Excluded Item | Reason |
| --- | --- |
| `models/` | Model weights are too large and may have separate licenses. |
| `image/` | Raw image files are not included in this cleaned code/data package. |
| `Results/`, `GPT_Results/` | Raw predictions and intermediate experiment outputs were excluded. |
| `DrVD-Bench/` | Large source dataset directory was excluded. |
| third-party `.git` folders | Cloned upstream repositories were not vendored into this release. |
| API keys / local paths | Hard-coded keys, private usernames, and server paths were redacted. |

See [`docs/release_audit.md`](docs/release_audit.md) for the release checklist.

---

## 📌 Manifest

The release manifest is available at [`docs/manifest.json`](docs/manifest.json). Summary:

| Item | Count |
| --- | ---: |
| Test JSONL files | 5 |
| Train JSONL files | 5 |
| Data construction scripts | 41 |
| Metric scripts | 6 |
| Compare-model scripts | 15 |
| Redacted image path fields | 99,997 |

---

## 🌟 Citation

If you use Med-R2 Bench, please cite our paper:

```bibtex
@inproceedings{medr2-2026,
  title = {Med-R2: An Adversarial Benchmark for Evidence-Grounded Reasoning in Medical VLMs},
  author = {Anonymous},
  booktitle = {Proceedings of EMNLP},
  year = {2026}
}
```

---

## 🤝 Contact

For questions about the benchmark, data format, or evaluation scripts, please open an issue in this repository.
