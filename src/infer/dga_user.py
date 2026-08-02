"""Stage 2 -- DGA user pass.

Prompts the DGA checkpoint with the *query alone* (plus persona, where the
domain has one) and asks what the query states and what it leaves unspecified.
No response is in the loop: implicit dimensions are a gap in the *question*.

Per-token logprobs are stored alongside the generation -- the ranker recovers a
confidence score per dimension by matching value spans back onto them, so the
two must stay index-aligned (see src.common.llm.generate).

Writes dims/user/{id}.json:

    {"id", "persona" (pwab only), "query", "raw_text", "logprobs",
     "json_data": {"explicit_dimensions": [...], "implicit_dimensions": [...]}}

    python -m src.infer.dga_user --config configs/code.yaml
"""

from __future__ import annotations

import argparse
import json
import logging

from tqdm import tqdm

from src.common import domains
from src.common.io import (
    extract_json_block,
    load_config,
    load_jsonl,
    normalise_dim_keys,
    prompt,
    render,
)
from src.common.llm import generate, load_causal_lm

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", help="override models.dga from the config")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths, domain = cfg["paths"], cfg["domain"]
    template = prompt(cfg, "dga_explicit_implicit")

    records = load_jsonl(paths.raw / f"{args.split}.jsonl")
    out_dir = paths.dims_user
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        records = [r for r in records if not (out_dir / f"{r['id']}.json").exists()]
    if args.limit:
        records = records[: args.limit]

    model, tokenizer = load_causal_lm(
        args.model or cfg["models"]["dga"], cfg["models"].get("base")
    )
    kept = 0

    for rec in tqdm(records, desc="dga_user"):
        query = domains.query(rec, domain)
        persona = domains.persona(rec, domain)
        rendered = render(template, user_query=query, persona=persona)

        raw_text, logprobs = generate(
            model,
            tokenizer,
            rendered,
            max_new_tokens=cfg["generation"]["dga_max_new_tokens"],
            with_logprobs=True,
        )

        payload = extract_json_block(raw_text)
        if not isinstance(payload, dict):
            log.warning("%s: generation did not parse as JSON; skipping", rec["id"])
            continue
        payload = normalise_dim_keys(payload)
        payload.setdefault("explicit_dimensions", [])
        payload.setdefault("implicit_dimensions", [])

        record = {"id": rec["id"]}
        if persona:
            record["persona"] = persona
        record.update(
            {
                "query": query,
                "raw_text": raw_text,
                "json_data": payload,
                "logprobs": logprobs,
            }
        )

        (out_dir / f"{rec['id']}.json").write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )
        kept += 1

    print(f"wrote {kept}/{len(records)} user-dimension files -> {out_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
