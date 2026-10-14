#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$TASK_ROOT/third_party/lingbot-vla-v2:$TASK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
TASK_PYTHON="${LINGBOT_PYTHON:-python}"
TASK_CONFIG="${LINGBOT_CONFIG:-$TASK_ROOT/training/configs/default.json}"
if [[ -z "${LINGBOT_CONFIG:-}" && -f "$TASK_ROOT/configs/runtime.json" ]]; then TASK_CONFIG="$TASK_ROOT/configs/runtime.json"; fi
"$TASK_PYTHON" -u -m training.preflight --config "$TASK_CONFIG"
exec "$TASK_PYTHON" -u -m torch.distributed.run --standalone --nnodes=1 \
  --nproc-per-node="${NUM_GPUS:-4}" -m training.train_server --config "$TASK_CONFIG" "$@"
