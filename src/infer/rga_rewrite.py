"""Stage 5 -- rank, then rewrite.

For every test id that has both user and system dimensions: score the user
dimensions with the ranker, take the top-k, and hand them back to the RGA along
with the query and its own base response.

The rewrite runs on the **RGA** checkpoint, not the DGA one. The rewrite prompt
demands ===START===/===END===, and the DGA was SFT'd to emit
===START_JSON===/===END_JSON=== -- running this on the DGA is what made the
original marker extraction unstable.

Writes outputs/llm_outputs.pkl: {id: {query, true, llm, rewrite, ranked_dims}}.

    python -m src.infer.rga_rewrite --config configs/code.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle

from tqdm import tqdm

from src.common.io import (
    END,
    START,
    extract_between,
    iter_dims_user,
    load_config,
    load_dims_system,
    load_jsonl,
    prompt,
    render,
    user_dims as parse_user_dims,
)
from src.common.llm import generate, load_causal_lm, load_embedder, load_tokenizer
from src.rank.ranker import attach_logprobs, embed_dims, rank_user_dims

log = logging.getLogger(__name__)


def format_dims(dims):
    return "\n".join(
        f"{i + 1}. {d['name']}: {d['value']} ({d.get('justification', '')})"
        for i, d in enumerate(dims)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rga-model", help="override models.rga")
    parser.add_argument("--dga-model", help="override models.dga (tokenizer only)")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    rank_cfg = cfg["ranker"]
    template = prompt(cfg, "rga_rewrite")

    base = {
        rec["id"]: {"query": rec["query"], "true": rec["true"], "llm": rec["llm"]}
        for rec in load_jsonl(paths.rga_preds)
    }

    # The RGA writes the rewrite; the DGA's tokenizer is still needed to map
    # logprobs back onto the dimension values it generated.
    base = cfg["models"].get("base")
    rga_model, rga_tokenizer = load_causal_lm(args.rga_model or cfg["models"]["rga"], base)
    dga_tokenizer = load_tokenizer(args.dga_model or cfg["models"]["dga"], base)
    emb_model, emb_tokenizer = load_embedder(cfg["models"]["embedder"])

    system_records = load_dims_system(paths)
    user_records = list(iter_dims_user(paths))
    if args.limit:
        user_records = user_records[: args.limit]

    results, skipped = {}, 0
    for user_record in tqdm(user_records, desc="rewrite"):
        uid = user_record["id"]
        system_record = system_records.get(uid)
        if system_record is None or uid not in base:
            skipped += 1
            continue

        try:
            dims = parse_user_dims(user_record)
            system_dims = system_record.get("system_explicit_dimensions", [])
            if not dims:
                skipped += 1
                continue

            attach_logprobs(
                dims,
                user_record["raw_text"],
                user_record["logprobs"],
                dga_tokenizer,
                min_ratio=rank_cfg["min_ratio"],
            )
            embed_dims(dims, emb_model, emb_tokenizer)
            embed_dims(system_dims, emb_model, emb_tokenizer)

            ranked = [
                dims[i]
                for i in rank_user_dims(
                    dims,
                    system_dims,
                    rank_cfg["lambda1"],
                    rank_cfg["lambda2"],
                    rank_cfg["top_k"],
                )
            ]

            rendered = render(
                template,
                user_query=base[uid]["query"],
                base_response=base[uid]["llm"],
                implicit_dimensions=format_dims(ranked),
            )
            text = generate(
                rga_model,
                rga_tokenizer,
                rendered,
                max_new_tokens=cfg["generation"]["rewrite_max_new_tokens"],
            )
            rewrite = extract_between(text, START, END)
            if rewrite is None:
                log.warning("%s: rewrite had no ===START===/===END=== span", uid)
                skipped += 1
                continue

            results[uid] = {
                **base[uid],
                "rewrite": rewrite,
                "ranked_dims": [
                    {k: v for k, v in d.items() if k != "embed"} for d in ranked
                ],
            }

        except Exception:
            log.exception("%s: rewrite failed", uid)
            skipped += 1

    paths.outputs.mkdir(parents=True, exist_ok=True)
    with open(paths.rewrites_pkl, "wb") as f:
        pickle.dump(results, f)

    print(
        f"rewrote {len(results)} examples "
        f"(skipped {skipped}) -> {paths.rewrites_pkl}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
