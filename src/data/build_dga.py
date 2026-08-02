"""labels/teacher/{split}/*.txt + raw/{split}.jsonl  ->  raw/{domain}_dga_{split}.jsonl

Alpaca-format SFT data for the Dimension Generating Agent, distilled from the
teacher labels. No API calls -- this stage is a pure join on ``id``.

Two rows per example, both trained into the same checkpoint:

  user row    instruction = teacher_label.txt rendered with (persona, query)
              output      = {"user_aspects": [...]}
  system row  instruction = dga_system.txt rendered with (query, gold response)
              output      = {"solution_aspects": [...]}

Note the deliberate train/inference prompt gap: training targets
``user_aspects``/``solution_aspects``, while at inference the model is prompted
with dga_explicit_implicit.txt and asked for
``explicit_dimensions``/``implicit_dimensions``. SFT teaches the model what a
dimension *is*; the inference prompt reshapes the output. This is intentional.

    python -m src.data.build_dga --config configs/code.yaml
"""

from __future__ import annotations

import argparse
import json
import logging

from src.common import domains
from src.common.io import (
    END_JSON,
    START_JSON,
    extract_json_block,
    iter_teacher_labels,
    load_config,
    load_jsonl,
    prompt,
    render,
    write_jsonl,
)

log = logging.getLogger(__name__)


def wrap(obj):
    return f"{START_JSON}\n{json.dumps(obj, ensure_ascii=False, indent=2)}\n{END_JSON}"


def build_split(records, labels, domain, teacher_tpl, system_tpl):
    by_id = {rec["id"]: rec for rec in records}
    rows, missing, unparsed = [], 0, 0

    for uid, gpt_output in labels:
        rec = by_id.get(uid)
        if rec is None:
            missing += 1
            continue

        label = extract_json_block(gpt_output)
        if not isinstance(label, dict):
            log.warning("%s: teacher output did not parse as JSON", uid)
            unparsed += 1
            continue

        fields = domains.teacher_fields(rec, domain)

        user_aspects = label.get("user_aspects")
        if user_aspects:
            rows.append(
                {
                    "instruction": render(
                        teacher_tpl,
                        persona=fields["persona"],
                        query=fields["query"],
                        gold_response=fields["gold_response"],
                    ),
                    "input": "",
                    "output": wrap({"user_aspects": user_aspects}),
                    "id": uid,
                    "head": "user",
                }
            )

        solution_aspects = label.get("solution_aspects")
        if solution_aspects:
            rows.append(
                {
                    "instruction": render(
                        system_tpl,
                        user_query=fields["query"],
                        assistant_response=fields["gold_response"],
                    ),
                    "input": "",
                    "output": wrap({"solution_aspects": solution_aspects}),
                    "id": uid,
                    "head": "system",
                }
            )

    if missing:
        log.warning("%d label files had no matching raw record", missing)
    if unparsed:
        log.warning("%d label files did not parse", unparsed)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths, domain = cfg["paths"], cfg["domain"]
    teacher_tpl = prompt(cfg, "teacher_label")
    system_tpl = prompt(cfg, "dga_system")

    for split in args.splits:
        labels = list(iter_teacher_labels(paths, split))
        if not labels:
            log.warning("no teacher labels found for %s; skipping", split)
            continue
        records = load_jsonl(paths.raw / f"{split}.jsonl")
        rows = build_split(records, labels, domain, teacher_tpl, system_tpl)
        write_jsonl(paths.dga_sft(split), rows)
        n_user = sum(r["head"] == "user" for r in rows)
        print(
            f"{split}: {len(rows)} rows "
            f"({n_user} user / {len(rows) - n_user} system) -> {paths.dga_sft(split)}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
