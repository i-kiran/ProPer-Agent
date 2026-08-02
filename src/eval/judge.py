"""Stage 6 -- LLM-as-judge.

Scores the base response (A) against the rewrite (B) on calibrated proactivity
and personalization, 0-5 each. The judge is told nothing about which response
is which, and prompts/judge.txt instructs it to score them independently rather
than pick a winner.

Writes outputs/results.pkl, adding A_score / B_score / A_just / B_just per id.

    OPENAI_API_KEY=... python -m src.eval.judge --config configs/code.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.common.gpt_api import call_gpt
from src.common.io import extract_json_block, load_config, prompt

log = logging.getLogger(__name__)

USER_TEMPLATE = """
Query:
{user_query}

Response A:
{response_a}

Response B:
{response_b}
"""


def judge_one(example, system_prompt, model):
    user_prompt = (
        USER_TEMPLATE.replace("{user_query}", example["query"])
        .replace("{response_a}", example["llm"])
        .replace("{response_b}", example["rewrite"])
    )
    content = call_gpt(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
    )
    if content is None:
        return None

    payload = extract_json_block(content)
    if not isinstance(payload, dict):
        log.warning("judge returned unparseable output: %.200s", content)
        return None

    return {
        "A_score": payload.get("response_A_score"),
        "B_score": payload.get("response_B_score"),
        "A_just": payload.get("response_A_justification"),
        "B_just": payload.get("response_B_justification"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    system_prompt = prompt(cfg, "judge")
    model = cfg["gpt"]["model"]

    with open(paths.rewrites_pkl, "rb") as f:
        outputs = pickle.load(f)

    items = [(uid, ex) for uid, ex in outputs.items() if ex.get("rewrite")]
    if args.limit:
        items = items[: args.limit]

    scored = 0
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {
            pool.submit(judge_one, ex, system_prompt, model): uid for uid, ex in items
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="judge"):
            uid = futures[future]
            try:
                verdict = future.result()
            except Exception:
                log.exception("%s: judging failed", uid)
                continue
            if verdict is None:
                continue
            outputs[uid].update(verdict)
            scored += 1

    with open(paths.results_pkl, "wb") as f:
        pickle.dump(outputs, f)

    print(f"scored {scored}/{len(items)} examples -> {paths.results_pkl}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
