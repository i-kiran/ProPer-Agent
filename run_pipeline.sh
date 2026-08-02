#!/usr/bin/env bash
# End-to-end inference + evaluation for one domain.
#
#   bash run_pipeline.sh code                    # all stages
#   bash run_pipeline.sh code rga_base dga_user  # selected stages
#
# Data preparation (src/data/) and training (src/train/) are deliberately not
# here: they are provenance, run once, and need API keys / GPUs respectively.
set -euo pipefail

DOMAIN="${1:?usage: run_pipeline.sh <code|medical|pwab> [stage ...]}"
shift || true
CONFIG="configs/${DOMAIN}.yaml"
STAGES=("$@")
if [ ${#STAGES[@]} -eq 0 ]; then
  STAGES=(rga_base dga_user dga_system rga_rewrite judge summarize)
fi

run () {
  echo ""
  echo "=== $1 ==="
  shift
  python -m "$@" --config "$CONFIG"
}

for stage in "${STAGES[@]}"; do
  case "$stage" in
    rga_base)     run "1/6 RGA base pass"        src.infer.rga_base ;;
    dga_user)     run "2/6 DGA user pass"        src.infer.dga_user ;;
    dga_system)   run "3/6 DGA system pass"      src.infer.dga_system ;;
    rga_rewrite)  run "4/6 rank + RGA rewrite"   src.infer.rga_rewrite ;;
    judge)        run "5/6 GPT judge"            src.eval.judge ;;
    summarize)    run "6/6 results"              src.eval.summarize ;;
    *) echo "unknown stage: $stage" >&2; exit 1 ;;
  esac
done
