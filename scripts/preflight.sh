#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
load_runtime_state

usage() {
  cat <<'EOF'
Validate a running Agile SONIC training container.

Usage:
  scripts/preflight.sh [--gpu-count N] [--project-check] [--strict]

Options:
  --gpu-count N    Require at least N visible GPUs (default: GPU_COUNT or all)
  --project-check  Also run SONIC's project check (requires tensorrt/full target)
  --strict         Treat host/container limit warnings as errors
  -h, --help       Show this help

Set PREFLIGHT_IMPORTS to a comma-separated module list to override the fast
Python import checks. EXPECTED_PYTHON, EXPECTED_TORCH, and EXPECTED_CUDA default
to 3.11, 2.7.0, and 12.8. GPU/import checks always use agile-sonic regardless
of SONIC_ENV; installed tools/data/inference environments receive pip checks.
EOF
}

gpu_count="${GPU_COUNT:-}"
project_check=0
strict="${PREFLIGHT_STRICT:-0}"

while (( $# > 0 )); do
  case "$1" in
    --gpu-count)
      [[ $# -ge 2 ]] || die "--gpu-count requires a value"
      gpu_count="$2"
      shift 2
      ;;
    --project-check)
      project_check=1
      shift
      ;;
    --strict)
      strict=1
      shift
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

require_command nvidia-smi
visible_gpu_count="$(nvidia-smi --list-gpus | wc -l)"
visible_gpu_count="${visible_gpu_count//[[:space:]]/}"
if [[ -z "${gpu_count}" ]]; then
  gpu_count="${visible_gpu_count}"
fi
require_positive_integer GPU_COUNT "${gpu_count}"

source_dir="${SONIC_WORKDIR:-/workspace/sonic-training}"
datasets_dir="${DATASETS_DIR:-/datasets}"
runs_dir="${RUNS_DIR:-/runs}"
run_dir="${RUN_DIR:-${runs_dir}/nvidia-omniverse}"
cache_dir="${CACHE_DIR:-/cache}"
cache_probe_dir="${XDG_CACHE_HOME:-${cache_dir}/xdg}"
failures=0

expected_revision="${SONIC_EXPECTED_REVISION:-}"
require_clean="${SONIC_REQUIRE_CLEAN:-0}"
case "${require_clean}" in
  0|false|FALSE|no|NO)
    require_clean=0
    ;;
  1|true|TRUE|yes|YES)
    require_clean=1
    ;;
  *)
    die "SONIC_REQUIRE_CLEAN must be 0 or 1"
    ;;
esac

check_directory() {
  local access="$1"
  local path="$2"
  if [[ ! -d "${path}" ]]; then
    printf '[FAIL] directory is not mounted: %s\n' "${path}" >&2
    failures=$((failures + 1))
  elif [[ "${access}" == "write" && ! -w "${path}" ]]; then
    printf '[FAIL] directory is not writable: %s\n' "${path}" >&2
    failures=$((failures + 1))
  elif [[ ! -r "${path}" ]]; then
    printf '[FAIL] directory is not readable: %s\n' "${path}" >&2
    failures=$((failures + 1))
  else
    printf '[ OK ] %s directory: %s\n' "${access}" "${path}"
  fi
}

check_directory read "${source_dir}"
check_directory read "${datasets_dir}"
check_directory read "${runs_dir}"
check_directory write "${run_dir}"
check_directory read "${cache_dir}"
check_directory write "${cache_probe_dir}"

atomic_probe=""
atomic_ready=""
if atomic_probe="$(mktemp "${run_dir}/.agile-sonic-preflight.XXXXXX")"; then
  atomic_ready="${atomic_probe}.ready"
  if printf 'preflight\n' > "${atomic_probe}" && mv "${atomic_probe}" "${atomic_ready}"; then
    printf '[ OK ] atomic write/rename under %s\n' "${run_dir}"
    rm -f "${atomic_ready}"
  else
    printf '[FAIL] atomic write/rename failed under %s\n' "${run_dir}" >&2
    failures=$((failures + 1))
    rm -f "${atomic_probe}" "${atomic_ready}"
  fi
else
  printf '[FAIL] temporary file creation failed under %s\n' "${run_dir}" >&2
  failures=$((failures + 1))
fi

if [[ ! -f "${source_dir}/gear_sonic/train_agent_trl.py" ]]; then
  printf '[FAIL] SONIC training entrypoint is missing under %s\n' "${source_dir}" >&2
  failures=$((failures + 1))
else
  printf '[ OK ] SONIC training entrypoint found\n'
fi

if [[ -n "${expected_revision}" || "${require_clean}" == "1" ]]; then
  require_command git
  git -C "${source_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || die "SONIC source revision checks require a Git checkout at ${source_dir}"
fi
if [[ -n "${expected_revision}" ]]; then
  [[ "${expected_revision}" =~ ^[0-9a-fA-F]{40}$ ]] \
    || die "SONIC_EXPECTED_REVISION must be a full 40-character commit SHA"
  actual_revision="$(git -C "${source_dir}" rev-parse --verify HEAD)"
  if [[ "${actual_revision}" != "${expected_revision,,}" ]]; then
    die "SONIC source revision mismatch: expected ${expected_revision,,}, found ${actual_revision}"
  fi
  printf '[ OK ] SONIC source revision: %s\n' "${actual_revision}"
fi
if [[ "${require_clean}" == "1" ]]; then
  dirty_source="$(git -C "${source_dir}" status --porcelain --untracked-files=normal)"
  [[ -z "${dirty_source}" ]] \
    || die "SONIC source checkout is dirty while SONIC_REQUIRE_CLEAN=1"
  printf '[ OK ] SONIC source checkout is clean\n'
fi

if [[ -f "${source_dir}/.gitattributes" ]] \
  && git -C "${source_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  && command -v git-lfs >/dev/null 2>&1
then
  lfs_listing="$(git -C "${source_dir}" lfs ls-files)"
  lfs_pointer_count="$(
    awk '$2 == "-" { count += 1 } END { print count + 0 }' <<<"${lfs_listing}"
  )"
  if (( lfs_pointer_count > 0 )); then
    printf '[FAIL] %s Git LFS file(s) are still pointer text; run git lfs pull on the host:\n' \
      "${lfs_pointer_count}" >&2
    awk '$2 == "-" && shown < 20 {
      $1=""; $2=""; sub(/^  */, ""); print "       " $0; shown += 1
    }' <<<"${lfs_listing}" >&2
    failures=$((failures + 1))
  else
    lfs_file_count="$(awk 'NF >= 3 { count += 1 } END { print count + 0 }' <<<"${lfs_listing}")"
    printf '[ OK ] Git LFS working-tree objects materialized: %s file(s)\n' \
      "${lfs_file_count}"
  fi
fi

memlock_limit="$(ulimit -l)"
nofile_limit="$(ulimit -n)"
printf 'Limits: memlock=%s, nofile=%s\n' "${memlock_limit}" "${nofile_limit}"
if [[ "${memlock_limit}" != "unlimited" && "${memlock_limit}" != "-1" ]]; then
  warn "memlock is not unlimited; NCCL/GPUDirect performance may suffer"
  if [[ "${strict}" == "1" ]]; then
    failures=$((failures + 1))
  fi
fi
if [[ "${nofile_limit}" =~ ^[0-9]+$ ]] && (( nofile_limit < 524288 )); then
  warn "nofile is below 524288; large Isaac workloads may exhaust descriptors"
  if [[ "${strict}" == "1" ]]; then
    failures=$((failures + 1))
  fi
fi

printf 'Shared memory:\n'
df -h /dev/shm
printf 'NVIDIA driver/GPU inventory:\n'
nvidia-smi

if (( failures > 0 )); then
  die "Filesystem/runtime preflight failed with ${failures} error(s)"
fi

activate_training_env
cd "${source_dir}"
log "Checking fixed training environment agile-sonic (independent of SONIC_ENV)"
python -m pip check

environment_is_installed() {
  local env_name="$1"
  conda env list --json | python -c \
    'import json, pathlib, sys; target=sys.argv[1]; envs=json.load(sys.stdin).get("envs", []); raise SystemExit(0 if any(pathlib.Path(path).name == target for path in envs) else 1)' \
    "${env_name}"
}

check_named_environment() {
  local env_name="$1"
  local allow_depthai_warning="${2:-0}"
  local check_log
  local check_status=0
  local ignored_count=0
  local unexpected=""
  local depthai_unsupported_regex='^depthai .* is not supported on this platform$'

  log "Checking ${env_name} dependency consistency"
  check_log="$(mktemp)"
  if conda run --no-capture-output -n "${env_name}" \
    python -m pip check >"${check_log}" 2>&1
  then
    check_status=0
  else
    check_status=$?
  fi

  if (( check_status == 0 )); then
    cat "${check_log}"
    rm -f "${check_log}"
    return
  fi

  if [[ "${allow_depthai_warning}" == "1" ]]; then
    ignored_count="$(grep -Ec "${depthai_unsupported_regex}" "${check_log}" || true)"
    unexpected="$(grep -Ev "${depthai_unsupported_regex}" "${check_log}" || true)"
    if [[ -z "${unexpected}" && "${ignored_count}" == "1" ]]; then
      warn "Accepted known tools metadata warning: $(grep -E "${depthai_unsupported_regex}" "${check_log}")"
      rm -f "${check_log}"
      return
    fi
  fi

  cat "${check_log}" >&2
  rm -f "${check_log}"
  die "${env_name} failed pip check"
}

if environment_is_installed agile-sonic-tools; then
  # Keep this filter identical to the image build gate: exactly one DepthAI
  # unsupported-platform metadata line is accepted; every other line is fatal.
  check_named_environment agile-sonic-tools 1
else
  log "Optional agile-sonic-tools environment is not installed; skipping its pip check"
fi

for optional_env in agile-sonic-data agile-sonic-inference; do
  if environment_is_installed "${optional_env}"; then
    check_named_environment "${optional_env}"
  else
    log "Optional ${optional_env} environment is not installed; skipping its pip check"
  fi
done

python "${SCRIPT_DIR}/_preflight.py" \
  --gpu-count "${gpu_count}" \
  --imports "${PREFLIGHT_IMPORTS:-torch,accelerate,numpy,hydra,omegaconf,tensordict,wandb,isaacsim,isaaclab,gear_sonic.train_agent_trl}"

if (( project_check == 1 )); then
  project_preflight="${source_dir}/scripts/env_checks/check_training_container.py"
  [[ -f "${project_preflight}" ]] || die "Project preflight not found: ${project_preflight}"
  log "Running SONIC project-level dependency and asset checks"
  python "${project_preflight}"
fi
