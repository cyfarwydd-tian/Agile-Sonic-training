#!/usr/bin/env bash

if [[ -r /usr/local/lib/agile-sonic/nccl-runtime.sh ]]; then
  # shellcheck source=files/nccl-runtime.sh
  source /usr/local/lib/agile-sonic/nccl-runtime.sh
fi

log() {
  printf '[agile-sonic] %s\n' "$*"
}

warn() {
  printf '[agile-sonic] WARNING: %s\n' "$*" >&2
}

die() {
  printf '[agile-sonic] ERROR: %s\n' "$*" >&2
  exit 1
}

load_runtime_state() {
  if [[ -r /run/agile-sonic/runtime.env ]]; then
    # shellcheck source=/dev/null
    source /run/agile-sonic/runtime.env
  fi
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer (got '${value}')"
}

activate_named_env() {
  local env_name="$1"
  local candidate
  local conda_sh=""

  [[ -n "${env_name}" ]] || die "Conda environment name cannot be empty"

  for candidate in \
    "${CONDA_EXE:-}" \
    "${HOME}/miniforge3/bin/conda" \
    "/opt/miniforge3/bin/conda" \
    "/opt/conda/bin/conda"
  do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      conda_sh="$(dirname "$(dirname "${candidate}")")/etc/profile.d/conda.sh"
      break
    fi
  done

  if [[ -z "${conda_sh}" ]] && command -v conda >/dev/null 2>&1; then
    candidate="$(command -v conda)"
    conda_sh="$(dirname "$(dirname "${candidate}")")/etc/profile.d/conda.sh"
  fi

  [[ -f "${conda_sh}" ]] || die \
    "Cannot find Conda initialization. Run this inside the training image or activate '${env_name}'."

  # shellcheck disable=SC1090
  source "${conda_sh}"
  conda activate "${env_name}"
}

activate_training_env() {
  # GPU training, NCCL, and dependency validation must never inherit the
  # interactive SONIC_ENV selection (tools/data/inference use incompatible
  # Python and PyTorch versions).
  activate_named_env agile-sonic
  if declare -F agile_sonic_enable_nccl_runtime >/dev/null 2>&1; then
    agile_sonic_enable_nccl_runtime
  fi
}
