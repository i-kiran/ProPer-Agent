"""raw/{split}.jsonl  ->  raw/{domain}_rga_{split}.jsonl

Alpaca-format SFT data for the Response Generating Agent. Instruction is the
domain framing plus the ===START===/===END=== format rule; input is the user
query; output is the gold response wrapped in those markers.

    python -m src.data.build_rga --config configs/code.yaml
"""

from __future__ import annotations

import argparse
import logging

from src.common import domains
from src.common.io import START, END, load_config, load_jsonl, write_jsonl

log = logging.getLogger(__name__)


def build_split(records, domain):
    instruction = domains.rga_instruction(domain)
    rows = []
    for rec in records:
        query = domains.query(rec, domain).replace("\n", "")
        gold = domains.gold(rec, domain).replace("\n", "")
        rows.append(
            {
                "instruction": instruction,
                "input": query,
                "output": f"\n{START}\n{gold}\n{END}\n",
                "id": rec["id"],
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths, domain = cfg["paths"], cfg["domain"]

    for split in args.splits:
        records = load_jsonl(paths.raw / f"{split}.jsonl")
        rows = build_split(records, domain)
        write_jsonl(paths.rga_sft(split), rows)
        log.info("%s: wrote %d rows -> %s", split, len(rows), paths.rga_sft(split))
        print(f"{split}: {len(rows)} rows -> {paths.rga_sft(split)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
