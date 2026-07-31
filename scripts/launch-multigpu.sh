#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
load_runtime_state

usage() {
  cat <<'EOF'
Launch SONIC training through Hugging Face Accelerate on one multi-GPU host.

Usage:
  scripts/launch-multigpu.sh [options] [-- training arguments...]

Options:
  --gpu-count N          Local process/GPU count (default: GPU_COUNT or all)
  --config FILE          Accelerate config file
  --mixed-precision MODE no, fp16, or bf16 (default: bf16)
  --main-port PORT       Rendezvous port (default: 29500)
  --entrypoint FILE      Training script (default: gear_sonic/train_agent_trl.py)
  --no-preflight         Skip the fast environment/GPU preflight
  -h, --help             Show this help

RUN_ID is generated when absent and exported with RUN_DIR/WANDB_RUN_ID. Arguments
after -- are passed unchanged to train_agent_trl.py. If no Hydra base_dir
override is supplied, outputs default to the unique RUN_DIR under /runs.
EOF
}

gpu_count="${GPU_COUNT:-}"
config_file="${ACCELERATE_CONFIG_FILE:-}"
mixed_precision="${MIXED_PRECISION:-bf16}"
main_port="${MASTER_PORT:-29500}"
entrypoint="${TRAINING_ENTRYPOINT:-gear_sonic/train_agent_trl.py}"
run_preflight="${RUN_PREFLIGHT:-1}"
training_args=()

while (( $# > 0 )); do
  case "$1" in
    --gpu-count)
      [[ $# -ge 2 ]] || die "--gpu-count requires a value"
      gpu_count="$2"
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || die "--config requires a value"
      config_file="$2"
      shift 2
      ;;
    --mixed-precision)
      [[ $# -ge 2 ]] || die "--mixed-precision requires a value"
      mixed_precision="$2"
      shift 2
      ;;
    --main-port)
      [[ $# -ge 2 ]] || die "--main-port requires a value"
      main_port="$2"
      shift 2
      ;;
    --entrypoint)
      [[ $# -ge 2 ]] || die "--entrypoint requires a value"
      entrypoint="$2"
      shift 2
      ;;
    --no-preflight)
      run_preflight=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      training_args=("$@")
      break
      ;;
    *)
      training_args+=("$1")
      shift
      ;;
  esac
done

if [[ -n "${config_file}" && "${config_file}" != /* ]]; then
  config_file="$(realpath -m "${config_file}")"
fi

activate_training_env
require_command accelerate
require_command nvidia-smi

workdir="${SONIC_WORKDIR:-/workspace/sonic-training}"
[[ -d "${workdir}" ]] || die "SONIC workdir does not exist: ${workdir}"
cd "${workdir}"
[[ -f "${entrypoint}" ]] || die "Training entrypoint does not exist: ${workdir}/${entrypoint}"

if [[ -z "${gpu_count}" ]]; then
  gpu_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
fi
require_positive_integer GPU_COUNT "${gpu_count}"
case "${mixed_precision}" in
  no|fp16|bf16) ;;
  *) die "--mixed-precision must be no, fp16, or bf16" ;;
esac
[[ "${main_port}" =~ ^[0-9]+$ ]] || die "--main-port must be an integer"
(( main_port >= 1024 && main_port <= 65535 )) || die "--main-port is outside 1024-65535"

visible_gpu_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( visible_gpu_count < gpu_count )); then
  die "${gpu_count} worker(s) requested, but PyTorch sees ${visible_gpu_count} GPU(s)"
fi

export GPU_COUNT="${gpu_count}"
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%S.%NZ)}"
safe_run_id="${RUN_ID//[^a-zA-Z0-9_.-]/-}"
export RUNS_DIR="${RUNS_DIR:-/runs}"
export RUN_DIR="${RUN_DIR:-${RUNS_DIR}/${safe_run_id}}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_ID}}"
export WANDB_DIR="${WANDB_DIR:-${RUN_DIR}/wandb}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
mkdir -p "${RUN_DIR}" "${WANDB_DIR}"

base_dir_supplied=0
for training_arg in "${training_args[@]}"; do
  case "${training_arg}" in
    base_dir=*|+base_dir=*|++base_dir=*)
      base_dir_supplied=1
      break
      ;;
  esac
done
if (( base_dir_supplied == 0 )) \
  && [[ "${entrypoint}" == "gear_sonic/train_agent_trl.py" ]]; then
  training_args+=("base_dir=${RUN_DIR}")
elif (( base_dir_supplied == 0 )); then
  warn "Custom entrypoint selected; pass its persistent output directory explicitly"
fi

if [[ "${run_preflight}" == "1" ]]; then
  preflight_args=(--gpu-count "${gpu_count}")
  project_preflight="${workdir}/scripts/env_checks/check_training_container.py"
  if [[ -f "${project_preflight}" ]]; then
    if python -c 'import tensorrt; from cuda.bindings import runtime' >/dev/null 2>&1; then
      preflight_args+=(--project-check)
    else
      warn "Mounted SONIC project check needs the tensorrt/full image target; running the base preflight only"
    fi
  fi
  "${SCRIPT_DIR}/preflight.sh" "${preflight_args[@]}"
fi

launch_args=(
  launch
  --num_machines 1
  --num_processes "${gpu_count}"
  --machine_rank 0
  --main_process_ip 127.0.0.1
  --main_process_port "${main_port}"
  --mixed_precision "${mixed_precision}"
)
if (( gpu_count > 1 )); then
  launch_args+=(--multi_gpu)
fi
if [[ -n "${config_file}" ]]; then
  [[ -f "${config_file}" ]] || die "Accelerate config does not exist: ${config_file}"
  launch_args+=(--config_file "${config_file}")
fi

log "Launching RUN_ID=${RUN_ID} with ${gpu_count} process(es), precision=${mixed_precision}"
exec accelerate "${launch_args[@]}" "${entrypoint}" "${training_args[@]}"
