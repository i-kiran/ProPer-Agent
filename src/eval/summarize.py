"""Stage 7 -- headline numbers.

Reads outputs/results.pkl and prints mean judge score and win rate for the base
response (A) against the rewrite (B).

    python -m src.eval.summarize --config configs/code.yaml
    python -m src.eval.summarize --config configs/code.yaml configs/pwab.yaml
"""

from __future__ import annotations

import argparse
import pickle

from src.common.io import load_config


def summarize(results):
    total_a = total_b = wins_a = wins_b = ties = n = 0

    for example in results.values():
        a, b = example.get("A_score"), example.get("B_score")
        if not (isinstance(a, int) and isinstance(b, int)):
            continue
        total_a += a
        total_b += b
        n += 1
        if a > b:
            wins_a += 1
        elif b > a:
            wins_b += 1
        else:
            ties += 1

    if n == 0:
        return None

    return {
        "n": n,
        "avg_A": total_a / n,
        "avg_B": total_b / n,
        "A_wins": wins_a,
        "B_wins": wins_b,
        "ties": ties,
        "B_win_rate": wins_b / n,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, nargs="+")
    args = parser.parse_args()

    header = f"{'domain':10s} {'n':>6s} {'avg A':>7s} {'avg B':>7s} {'A wins':>7s} {'B wins':>7s} {'ties':>6s} {'B win %':>8s}"
    print(header)
    print("-" * len(header))

    for config_path in args.config:
        cfg = load_config(config_path)
        results_pkl = cfg["paths"].results_pkl
        if not results_pkl.exists():
            print(f"{cfg['domain']:10s} (no results at {results_pkl})")
            continue

        with open(results_pkl, "rb") as f:
            stats = summarize(pickle.load(f))

        if stats is None:
            print(f"{cfg['domain']:10s} (no scored examples)")
            continue

        print(
            f"{cfg['domain']:10s} {stats['n']:6d} {stats['avg_A']:7.3f} "
            f"{stats['avg_B']:7.3f} {stats['A_wins']:7d} {stats['B_wins']:7d} "
            f"{stats['ties']:6d} {stats['B_win_rate'] * 100:7.2f}%"
        )


if __name__ == "__main__":
    main()
