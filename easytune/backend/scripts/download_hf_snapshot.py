from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--allow-pattern", action="append", default=[])
    args = parser.parse_args()
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        cache_dir=str(args.cache_dir.resolve()),
        allow_patterns=args.allow_pattern or None,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
