from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

import lm_eval
from datasets import load_dataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_packages() -> list[str]:
    return sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="gsm8k")
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--dataset-name", default="main")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--dataset-snapshot", type=Path, required=True)
    parser.add_argument("--framework-tag", default="v0.4.13")
    parser.add_argument("--framework-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-copy", type=Path, required=True)
    parser.add_argument("--lock-output", type=Path, required=True)
    args = parser.parse_args()

    lm_eval_root = Path(lm_eval.__file__).resolve().parent
    task_path = lm_eval_root / "tasks" / "gsm8k" / "gsm8k.yaml"
    if args.task != "gsm8k" or not task_path.exists():
        raise ValueError(f"Unsupported or missing task config: {args.task}")

    snapshot_path = args.dataset_snapshot.resolve()
    if snapshot_path.name != args.dataset_revision:
        raise ValueError(f"Dataset snapshot revision mismatch: {snapshot_path.name} != {args.dataset_revision}")
    data_files = {
        "train": str(snapshot_path / args.dataset_name / "train-00000-of-00001.parquet"),
        "test": str(snapshot_path / args.dataset_name / "test-00000-of-00001.parquet"),
    }
    for split, path in data_files.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing {split} parquet: {path}")
    dataset = load_dataset("parquet", data_files=data_files)
    split_manifest = {
        split: {
            "rows": len(records),
            "fingerprint": records._fingerprint,
            "columns": records.column_names,
        }
        for split, records in dataset.items()
    }
    packages = installed_packages()
    manifest = {
        "schema_version": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "framework": "lm_eval",
        "framework_package": "lm-eval",
        "framework_version": importlib.metadata.version("lm-eval"),
        "framework_tag": args.framework_tag,
        "framework_commit": args.framework_commit,
        "task": args.task,
        "task_config_source": "lm_eval/tasks/gsm8k/gsm8k.yaml",
        "task_config_sha256": sha256_file(task_path),
        "dataset": args.dataset,
        "dataset_name": args.dataset_name,
        "dataset_revision": args.dataset_revision,
        "dataset_snapshot_verified": snapshot_path.name == args.dataset_revision,
        "splits": split_manifest,
        "packages_lock": args.lock_output.name,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.task_copy.parent.mkdir(parents=True, exist_ok=True)
    args.lock_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(task_path, args.task_copy)
    args.lock_output.write_text("\n".join(packages) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
