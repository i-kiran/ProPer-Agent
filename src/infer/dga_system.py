"""Stage 3 -- DGA system pass.

Same checkpoint as the user pass, different prompt: given the query *and* the
base response from stage 1, extract what the response explicitly did -- how it
framed, scoped, or deferred. The ranker subtracts these from the user
dimensions, which is what turns "everything the query left unsaid" into "the
gaps the answer actually left open".

Writes dims/system/{id}.txt (the JSON body, no markers).

    python -m src.infer.dga_system --config configs/code.yaml
"""

from __future__ import annotations

import argparse
import json
import logging

from tqdm import tqdm

from src.common.io import extract_json_block, load_config, load_jsonl, prompt, render
from src.common.llm import generate, load_causal_lm

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", help="override models.dga from the config")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    template = prompt(cfg, "dga_system")

    records = load_jsonl(paths.rga_preds)
    out_dir = paths.dims_system
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        records = [r for r in records if not (out_dir / f"{r['id']}.txt").exists()]
    if args.limit:
        records = records[: args.limit]

    model, tokenizer = load_causal_lm(
        args.model or cfg["models"]["dga"], cfg["models"].get("base")
    )
    kept = 0

    for rec in tqdm(records, desc="dga_system"):
        rendered = render(
            template, user_query=rec["query"], assistant_response=rec["llm"]
        )
        text = generate(
            model,
            tokenizer,
            rendered,
            max_new_tokens=cfg["generation"]["dga_max_new_tokens"],
        )

        payload = extract_json_block(text)
        if not isinstance(payload, dict) or "system_explicit_dimensions" not in payload:
            log.warning("%s: no valid system_explicit_dimensions; skipping", rec["id"])
            continue

        (out_dir / f"{rec['id']}.txt").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        kept += 1

    print(f"wrote {kept}/{len(records)} system-dimension files -> {out_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
