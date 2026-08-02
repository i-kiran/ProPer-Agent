"""Teacher labelling: raw/{split}.jsonl -> labels/teacher/{split}/{id}.txt

Distils dimension labels out of a frontier model (GPT-5-nano in the paper). The
teacher sees the gold response, which the student never will -- this is
privileged-information distillation.

Provenance stage: the released labels were produced by this script. Rerunning it
costs API calls and will not reproduce the labels token-for-token.

    OPENAI_API_KEY=... python -m src.data.label_gpt --config configs/code.yaml
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.common import domains
from src.common.gpt_api import call_gpt
from src.common.io import load_config, load_jsonl, prompt, write_text

log = logging.getLogger(__name__)


def label_one(rec, domain, system_prompt, out_dir, model):
    user_text = domains.teacher_input_text(rec, domain)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    output = call_gpt(messages, model=model)
    if output is None:
        return False

    write_text(
        out_dir / f"{rec['id']}.txt",
        f"\n=== INPUT ===\n{user_text}\n\n=== GPT OUTPUT ===\n{output}\n",
    )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths, domain = cfg["paths"], cfg["domain"]
    system_prompt = prompt(cfg, "teacher_label")
    model = cfg["gpt"]["model"]

    for split in args.splits:
        out_dir = paths.teacher_labels(split)
        out_dir.mkdir(parents=True, exist_ok=True)
        records = load_jsonl(paths.raw / f"{split}.jsonl")
        if not args.overwrite:
            records = [r for r in records if not (out_dir / f"{r['id']}.txt").exists()]

        ok = 0
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = {
                pool.submit(label_one, rec, domain, system_prompt, out_dir, model): rec
                for rec in records
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=split):
                rec = futures[future]
                try:
                    ok += bool(future.result())
                except Exception:
                    log.exception("failed to label %s", rec["id"])

        print(f"{split}: labelled {ok}/{len(records)} -> {out_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
