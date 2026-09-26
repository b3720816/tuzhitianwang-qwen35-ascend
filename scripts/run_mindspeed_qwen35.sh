#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINDSPEED_MM_ROOT="${MINDSPEED_MM_ROOT:-$ROOT/third_party/MindSpeed-MM}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/cann/set_env.sh}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-6000}"
ASCEND_DEVICE_ID="${ASCEND_DEVICE_ID:-0}"
RUN_ID="${RUN_ID:-qwen35_0.8B_$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/ascend_qwen35/runs/$RUN_ID}"
RESULT_DIR="$("$PYTHON_BIN" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$RESULT_DIR")"
MINDSPEED_MM_ROOT="$("$PYTHON_BIN" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$MINDSPEED_MM_ROOT")"
PYTHON_BIN="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"

: "${QWEN35_HF_DIR:?Set QWEN35_HF_DIR to the downloaded Qwen3.5-0.8B Hugging Face weight directory}"
: "${QWEN35_DCP_DIR:?Set QWEN35_DCP_DIR to the converted DCP checkpoint directory}"
: "${QWEN35_DATASET:?Set QWEN35_DATASET to the official annotations_slim.json path}"

if [[ ! -f "$CANN_ENV" ]]; then
  printf 'CANN environment script not found: %s\n' "$CANN_ENV" >&2
  exit 2
fi

# shellcheck disable=SC1090
set +u
source "$CANN_ENV"
set -u
export ASCEND_DEVICE_ID
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$ASCEND_DEVICE_ID}"
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE="${MULTI_STREAM_MEMORY_REUSE:-2}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"
export ACLNN_CACHE_LIMIT="${ACLNN_CACHE_LIMIT:-100000}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

if [[ -e "$RESULT_DIR/train.log" || -e "$RESULT_DIR/qwen3_5_0.8B_run.yaml" ]]; then
  printf 'Existing run directory; choose a new RUN_ID or RESULT_DIR: %s\n' "$RESULT_DIR" >&2
  exit 2
fi
mkdir -p "$RESULT_DIR" "$RESULT_DIR/cache" "$RESULT_DIR/checkpoints"
CONFIG_PATH="$RESULT_DIR/qwen3_5_0.8B_run.yaml"
TRAIN_LOG="$RESULT_DIR/train.log"

printf '[1/5] Strict Ascend and bundle preflight\n'
"$PYTHON_BIN" "$ROOT/scripts/check_mindspeed_qwen35_bundle.py" \
  --mindspeed-root "$MINDSPEED_MM_ROOT" \
  --hf-model-dir "$QWEN35_HF_DIR" \
  --dcp-dir "$QWEN35_DCP_DIR" \
  --dataset "$QWEN35_DATASET" \
  --output "$RESULT_DIR/preflight.json" \
  2>&1 | tee "$RESULT_DIR/preflight.log"

printf '[2/5] Render the immutable 100-step validation contract with server paths\n'
"$PYTHON_BIN" "$ROOT/scripts/prepare_mindspeed_qwen35_config.py" \
  --template "$MINDSPEED_MM_ROOT/examples/qwen3_5/qwen3_5_0.8B_config.yaml" \
  --hf-model-dir "$QWEN35_HF_DIR" \
  --dcp-dir "$QWEN35_DCP_DIR" \
  --dataset "$QWEN35_DATASET" \
  --cache-dir "$RESULT_DIR/cache" \
  --save-dir "$RESULT_DIR/checkpoints" \
  --train-iters 100 \
  --num-workers "${QWEN35_NUM_WORKERS:-0}" \
  --preprocessing-num-workers "${QWEN35_PREPROCESSING_NUM_WORKERS:-1}" \
  --output "$CONFIG_PATH" \
  --metadata-output "$RESULT_DIR/config_metadata.json" \
  2>&1 | tee "$RESULT_DIR/config_generation.log"

printf '[3/5] Capture environment evidence\n'
{
  printf 'RUN_ID=%s\n' "$RUN_ID"
  printf 'UTC_TIME=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uname -a
  "$PYTHON_BIN" --version
  "$PYTHON_BIN" -c 'import torch; print("torch=" + torch.__version__)'
  "$PYTHON_BIN" -c 'import torch_npu; print("torch_npu=" + torch_npu.__version__); import torch; print("npu_available=" + str(torch.npu.is_available()))'
  "$PYTHON_BIN" -c 'import transformers; print("transformers=" + transformers.__version__)'
  "$PYTHON_BIN" -c 'import accelerate; print("accelerate=" + accelerate.__version__)'
  npu-smi info
} 2>&1 | tee "$RESULT_DIR/environment.log"

printf '[4/5] Run the official single-NPU 100-step FSDP training flow\n'
cd "$MINDSPEED_MM_ROOT"
"$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node 1 \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  mindspeed_mm/fsdp/train/trainer.py \
  "$CONFIG_PATH" \
  2>&1 | tee "$TRAIN_LOG"

printf '[5/5] Parse performance and compare loss/grad norm with the bundled reference\n'
cd "$ROOT"
"$PYTHON_BIN" scripts/analyze_qwen35_training_log.py \
  "$TRAIN_LOG" \
  --reference-log "$MINDSPEED_MM_ROOT/logs/qwen35_0.8B_20260608_110844.log" \
  --start-step 50 \
  --end-step 100 \
  --relative-tolerance 0.02 \
  --require-complete \
  --output "$RESULT_DIR/validation.json" \
  2>&1 | tee "$RESULT_DIR/validation.log"

printf 'Validated run evidence saved under: %s\n' "$RESULT_DIR"
