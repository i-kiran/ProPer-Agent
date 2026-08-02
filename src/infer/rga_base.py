"""Stage 1 -- RGA base pass.

Runs the RGA checkpoint over the test split and records its unaided response.
Everything downstream (system dimensions, the rewrite, the judge's Response A)
is measured against this.

    python -m src.infer.rga_base --config configs/code.yaml
"""

from __future__ import annotations

import argparse
import json
import logging

from tqdm import tqdm

from src.common.io import END, START, extract_between, load_config, load_jsonl
from src.common.llm import generate, load_causal_lm

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", help="override models.rga from the config")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    model_path = args.model or cfg["models"]["rga"]

    records = load_jsonl(paths.rga_sft("test"))
    if args.limit:
        records = records[: args.limit]

    model, tokenizer = load_causal_lm(model_path, cfg["models"].get("base"))
    out_path = paths.rga_preds
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in tqdm(records, desc="rga_base"):
            prompt = f"Instruction: {rec['instruction']}\nQuery:{rec['input']}"
            text = generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=cfg["generation"]["rga_base_max_new_tokens"],
            )

            response = extract_between(text, START, END)
            if response is None:
                log.warning("%s: no ===START===/===END=== span; skipping", rec["id"])
                continue

            gold = extract_between(rec["output"], START, END) or ""
            f.write(
                json.dumps(
                    {
                        "query": rec["input"],
                        "true": gold.replace("\n", "").strip(),
                        "llm": response,
                        "id": rec["id"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1

    print(f"wrote {kept}/{len(records)} predictions -> {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
