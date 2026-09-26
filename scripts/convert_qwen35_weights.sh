#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINDSPEED_MM_ROOT="${MINDSPEED_MM_ROOT:-$ROOT/third_party/MindSpeed-MM}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/cann/set_env.sh}"

: "${QWEN35_HF_DIR:?Set QWEN35_HF_DIR to the downloaded Qwen3.5-0.8B Hugging Face weight directory}"
: "${QWEN35_DCP_DIR:?Set QWEN35_DCP_DIR to the output DCP checkpoint directory}"

if [[ ! -f "$CANN_ENV" ]]; then
  printf 'CANN environment script not found: %s\n' "$CANN_ENV" >&2
  exit 2
fi
if [[ ! -d "$QWEN35_HF_DIR" ]]; then
  printf 'Qwen3.5 Hugging Face weight directory not found: %s\n' "$QWEN35_HF_DIR" >&2
  exit 3
fi
if [[ ! -f "$MINDSPEED_MM_ROOT/checkpoint/convert_cli.py" ]]; then
  printf 'MindSpeed-MM converter not found under: %s\n' "$MINDSPEED_MM_ROOT" >&2
  exit 4
fi

# shellcheck disable=SC1090
source "$CANN_ENV"
mkdir -p "$QWEN35_DCP_DIR" "$ROOT/results/ascend_qwen35/conversion"

cd "$MINDSPEED_MM_ROOT"
"$PYTHON_BIN" checkpoint/convert_cli.py GenericDCPConverter hf_to_dcp \
  --hf_dir "$QWEN35_HF_DIR" \
  --dcp_dir "$QWEN35_DCP_DIR" \
  2>&1 | tee "$ROOT/results/ascend_qwen35/conversion/convert_console.log"

if [[ ! -e "$QWEN35_DCP_DIR/release" ]]; then
  printf 'Conversion command completed but DCP release entry is missing: %s\n' "$QWEN35_DCP_DIR/release" >&2
  exit 5
fi
printf 'DCP checkpoint ready: %s\n' "$QWEN35_DCP_DIR"
