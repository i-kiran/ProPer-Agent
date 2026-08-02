"""Pack per-example artefact directories into JSONL shards, or unpack them.

The inference stages write one file per example so a crashed run can resume by
skipping what already exists. That is the wrong shape for distribution: the
three collections together run to ~13k small files, past what the Hub is
comfortable with and slow to transfer.

Packing produces:

    labels/teacher/{split}.jsonl   {id, input, gpt_output}
    dims/user.jsonl                the user-pass records verbatim
    dims/system.jsonl              {id, system_explicit_dimensions}

Readers in src/common/io.py accept either form and prefer the shard, so nothing
downstream changes.

    python -m src.data.shard pack   --config configs/code.yaml
    python -m src.data.shard unpack --config configs/code.yaml
    python -m src.data.shard pack   --config configs/code.yaml --remove-source
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil

from src.common.io import (
    load_config,
    load_jsonl,
    parse_system_dims,
    read_text,
    split_teacher_label,
    write_jsonl,
    write_text,
)

log = logging.getLogger(__name__)

SPLITS = ("train", "test")


def pack(paths, remove_source=False):
    packed = []

    for split in SPLITS:
        directory = paths.teacher_labels(split)
        if not directory.is_dir():
            continue
        rows = []
        for path in sorted(directory.glob("*.txt")):
            text_in, output = split_teacher_label(read_text(path))
            if output is None:
                log.warning("%s: no output marker; skipping", path.name)
                continue
            rows.append({"id": path.stem, "input": text_in, "gpt_output": output})
        write_jsonl(paths.teacher_labels_shard(split), rows)
        packed.append((paths.teacher_labels_shard(split), len(rows), directory))

    if paths.dims_user.is_dir():
        rows = []
        for path in sorted(paths.dims_user.glob("*.json")):
            record = json.loads(read_text(path))
            record.setdefault("id", path.stem)
            rows.append(record)
        write_jsonl(paths.dims_user_shard, rows)
        packed.append((paths.dims_user_shard, len(rows), paths.dims_user))

    if paths.dims_system.is_dir():
        rows = []
        for path in sorted(paths.dims_system.glob("*.txt")):
            record = parse_system_dims(read_text(path), path.stem)
            if record is not None:
                rows.append(record)
        write_jsonl(paths.dims_system_shard, rows)
        packed.append((paths.dims_system_shard, len(rows), paths.dims_system))

    for shard, count, source in packed:
        print(f"{count:6d} -> {shard}")
        if remove_source:
            shutil.rmtree(source)
            print(f"       removed {source}")

    return packed


def unpack(paths):
    for split in SPLITS:
        shard = paths.teacher_labels_shard(split)
        if not shard.is_file():
            continue
        directory = paths.teacher_labels(split)
        for row in load_jsonl(shard):
            write_text(
                directory / f"{row['id']}.txt",
                f"\n=== INPUT ===\n{row['input']}\n\n"
                f"=== GPT OUTPUT ===\n{row['gpt_output']}\n",
            )
        print(f"{shard} -> {directory}")

    if paths.dims_user_shard.is_file():
        for row in load_jsonl(paths.dims_user_shard):
            write_text(
                paths.dims_user / f"{row['id']}.json",
                json.dumps(row, ensure_ascii=False),
            )
        print(f"{paths.dims_user_shard} -> {paths.dims_user}")

    if paths.dims_system_shard.is_file():
        for row in load_jsonl(paths.dims_system_shard):
            write_text(
                paths.dims_system / f"{row['id']}.txt",
                json.dumps(row, ensure_ascii=False, indent=2),
            )
        print(f"{paths.dims_system_shard} -> {paths.dims_system}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pack", "unpack"])
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--remove-source",
        action="store_true",
        help="delete the per-example directories after packing",
    )
    args = parser.parse_args()

    paths = load_config(args.config)["paths"]
    if args.action == "pack":
        pack(paths, args.remove_source)
    else:
        unpack(paths)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
