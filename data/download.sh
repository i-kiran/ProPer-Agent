#!/usr/bin/env bash
# Pull the release from the Hugging Face Hub.
#
#   bash data/download.sh                       # dataset only
#   bash data/download.sh --with-models         # dataset + all six adapters
#   bash data/download.sh --with-models code    # dataset + one domain's pair
#
# Then point configs/{domain}.yaml at data_root: data/release/{domain}.
#
#   dataset  https://huggingface.co/datasets/itsgupta/proper-agents-data
#   models   https://huggingface.co/itsgupta/proper-agents-models
#
# The released checkpoints are LoRA adapters (~50 MB each), not merged 8B
# weights: the base model is public and src/common/llm.py merges the adapter
# in memory at load time. All six live in one repo, one subfolder per agent:
#
#   code/rga/  code/dga/  medical/rga/  medical/dga/  pwab/rga/  pwab/dga/
set -euo pipefail

DATASET_REPO="${DATASET_REPO:-itsgupta/proper-agents-data}"
MODEL_REPO="${MODEL_REPO:-itsgupta/proper-agents-models}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

HF="$(command -v hf || command -v huggingface-cli)" || {
  echo "hf CLI not found; pip install -r requirements.txt" >&2
  exit 1
}

echo "downloading $DATASET_REPO -> $ROOT/data/release"
"$HF" download "$DATASET_REPO" --repo-type dataset --local-dir "$ROOT/data/release"

if [ "${1:-}" = "--with-models" ]; then
  if [ -n "${2:-}" ]; then DOMAINS=("$2"); else DOMAINS=(code medical pwab); fi

  for domain in "${DOMAINS[@]}"; do
    # Subfolders land at saves/{domain}/{rga,dga} -- exactly what
    # configs/{domain}.yaml points at, so no moving afterwards.
    echo "downloading $MODEL_REPO :: $domain/* -> $ROOT/saves/$domain"
    "$HF" download "$MODEL_REPO" --include "$domain/*" --local-dir "$ROOT/saves"
  done
fi

cat <<'EOF'

Dataset layout:
  data/release/{code,medical,pwab}/
    raw/     train.jsonl  test.jsonl
             {domain}_rga_{train,test}.jsonl    RGA SFT data
             {domain}_dga_{train,test}.jsonl    DGA SFT data
             {domain}_test_rga_preds.jsonl      base responses
    labels/teacher/{train,test}/{id}.txt        teacher dimension labels
    dims/user/{id}.json                         DGA user pass + logprobs
    dims/system/{id}.txt                        DGA system pass

Model layout:
  saves/{code,medical,pwab}/{rga,dga}/          LoRA adapters
EOF
