#!/usr/bin/env bash
# One-off baseline smoke test:  OLMo-3-7B-Instruct + lm-eval-harness + chat
# template + greedy decoding on mbpp_plus_instruct.  Should reproduce the
# reported ~60% MBPP+ pass@1 if our pipeline is sound.
set -euo pipefail
cd "$(dirname "$0")/../.."

export HF_ALLOW_CODE_EVAL=1

TS=$(date +%Y%m%d_%H%M%S)
OUT="finetuning/outputs/harness_smoke/${TS}"
mkdir -p "$OUT"

uv run lm_eval \
    --model vllm \
    --model_args "pretrained=allenai/OLMo-3-7B-Instruct,dtype=bfloat16,trust_remote_code=True,gpu_memory_utilization=0.9,max_model_len=8192" \
    --tasks mbpp_instruct_fixed \
    --include_path finetuning/mbpp/lm_eval_tasks \
    --batch_size auto \
    --apply_chat_template \
    --output_path "$OUT" \
    --log_samples \
    --confirm_run_unsafe_code 2>&1 | tee "$OUT/smoke.log"

echo
echo "--- summary ---"
find "$OUT" -name "results_*.json" -exec uv run python -c "
import json, sys
d = json.load(open(sys.argv[1]))
for t, m in d.get('results', {}).items():
    print(t, m)
" {} \;
