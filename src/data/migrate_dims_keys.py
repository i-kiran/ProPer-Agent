"""One-shot rename of `missed_dimensions` -> `implicit_dimensions`.

Artefacts generated before the terminology change store the old key. This
rewrites them in place. It is idempotent, and `src.common.io.normalise_dim_keys`
accepts both names, so running it is optional.

    python -m src.data.migrate_dims_keys --dir data/samples/code/dims/user
    python -m src.data.migrate_dims_keys --dir <release>/dims/user --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OLD, NEW = "missed_dimensions", "implicit_dimensions"


def migrate_file(path, dry_run=False):
    record = json.loads(path.read_text(encoding="utf-8"))
    data = record.get("json_data")
    if not isinstance(data, dict) or OLD not in data:
        return False

    record["json_data"] = {
        (NEW if k == OLD else k): v for k, v in data.items()
    }
    # `raw_text` is the verbatim generation and is kept as-is: it is the
    # provenance record of what the model actually emitted.
    if not dry_run:
        path.write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="a dims/user directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = sorted(Path(args.dir).glob("*.json"))
    changed = sum(migrate_file(p, args.dry_run) for p in files)
    verb = "would migrate" if args.dry_run else "migrated"
    print(f"{verb} {changed}/{len(files)} files in {args.dir}")


if __name__ == "__main__":
    main()
