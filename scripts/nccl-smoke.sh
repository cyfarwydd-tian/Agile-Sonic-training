#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
  cat <<'EOF'
Run a single-host NCCL all-reduce smoke test.

Usage:
  scripts/nccl-smoke.sh [options]

Options:
  --gpu-count N    Number of GPUs/ranks (default: GPU_COUNT or all visible)
  --size-mb N      Tensor size per rank in MiB (default: 64)
  --iterations N   Timed all-reduce iterations (default: 5)
  --timeout N      Collective timeout in seconds (default: 180)
  -h, --help       Show this help
EOF
}

gpu_count="${GPU_COUNT:-}"
size_mb="${NCCL_SMOKE_MB:-64}"
iterations="${NCCL_SMOKE_ITERATIONS:-5}"
timeout_seconds="${NCCL_SMOKE_TIMEOUT:-180}"

while (( $# > 0 )); do
  case "$1" in
    --gpu-count)
      [[ $# -ge 2 ]] || die "--gpu-count requires a value"
      gpu_count="$2"
      shift 2
      ;;
    --size-mb)
      [[ $# -ge 2 ]] || die "--size-mb requires a value"
      size_mb="$2"
      shift 2
      ;;
    --iterations)
      [[ $# -ge 2 ]] || die "--iterations requires a value"
      iterations="$2"
      shift 2
      ;;
    --timeout)
      [[ $# -ge 2 ]] || die "--timeout requires a value"
      timeout_seconds="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

activate_training_env
require_command nvidia-smi
if [[ -z "${gpu_count}" ]]; then
  gpu_count="$(nvidia-smi --list-gpus | wc -l)"
  gpu_count="${gpu_count//[[:space:]]/}"
fi
require_positive_integer GPU_COUNT "${gpu_count}"
require_positive_integer NCCL_SMOKE_MB "${size_mb}"
require_positive_integer NCCL_SMOKE_ITERATIONS "${iterations}"
require_positive_integer NCCL_SMOKE_TIMEOUT "${timeout_seconds}"

visible_gpu_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( visible_gpu_count < gpu_count )); then
  die "${gpu_count} GPUs requested, but PyTorch sees only ${visible_gpu_count}"
fi

export NCCL_SMOKE_MB="${size_mb}"
export NCCL_SMOKE_ITERATIONS="${iterations}"
export NCCL_SMOKE_TIMEOUT="${timeout_seconds}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

log "Starting NCCL smoke test with ${gpu_count} local ranks"
exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${gpu_count}" \
  "${SCRIPT_DIR}/_nccl_smoke.py"
