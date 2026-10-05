# Release Audit

Prepared from: `xxx/path/ACL-2026`

## Included

- Test JSONL files: 5
- Train JSONL files: 5
- DataProcess Python files: 41
- Metric Python files: 6
- Compare-model top-level Python files: 15

## Redacted

- JSONL image path fields redacted: 99997
- Hard-coded API keys replaced with `xxx` placeholders.
- Server-local absolute paths and private usernames replaced with `xxx/path` or `xxx`.

## Excluded

- `models/`: model weights.
- `image/`: raw image files.
- `Results/` and `GPT_Results/`: raw prediction/evaluation artifacts.
- `DrVD-Bench/`: large source dataset directory.
- `compare_models/HuatuoGPT_Vision/` and `compare_models/Janus/`: third-party cloned repositories with `.git` history; please use their official upstream repositories instead.
