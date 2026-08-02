#!/usr/bin/env bash
# LoRA SFT for both agents of one domain.
#
#   bash src/train/train.sh code [DATA_DIR] [SAVE_DIR]
#
# Requires LLaMA-Factory on PATH (`pip install llamafactory[torch,metrics]`).
# DATA_DIR must contain {domain}/raw/{domain}_{rga,dga}_train.jsonl -- i.e. the
# release pulled by data/download.sh, or data/samples for a smoke test.
set -euo pipefail

DOMAIN="${1:?usage: train.sh <code|medical|pwab> [DATA_DIR] [SAVE_DIR]}"
DATA_DIR="${2:-data/release}"
SAVE_DIR="${3:-saves}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LLaMA-Factory resolves dataset names against dataset_info.json inside
# dataset_dir, so put ours where the data lives.
cp "$HERE/dataset_info.json" "$DATA_DIR/dataset_info.json"

# Adapters land at saves/{domain}/{rga,dga} -- the same layout as the Hub repo
# (one subfolder per agent), which is what configs/{domain}.yaml points at.
# There is no merge step: src/common/llm.py applies the adapter to the base
# model and merges it in memory at load time.
train_agent () {
  local agent="$1"
  local out="$SAVE_DIR/$DOMAIN/$agent"

  llamafactory-cli train "$HERE/$agent.yaml" \
    model_name_or_path="$BASE_MODEL" \
    dataset="${agent}_${DOMAIN}" \
    dataset_dir="$DATA_DIR" \
    output_dir="$out"

  echo "adapter -> $out"
}

train_agent rga
train_agent dga
