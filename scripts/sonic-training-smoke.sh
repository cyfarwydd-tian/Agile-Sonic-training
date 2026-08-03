#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
  cat <<'EOF'
Run the self-contained, real SONIC H20 training smoke test on one GPU.

Usage:
  scripts/sonic-training-smoke.sh --source PATH [options]

Options:
  --source PATH       Private sonic-training checkout (required)
  --image IMAGE       Training image reference
  --runs PATH         Persistent output directory (default: ./runs)
  --cache PATH        Persistent cache directory (default: ./.cache/agile-sonic)
  --gpu DEVICE        Docker GPU device index or UUID (default: 0)
  --keep-fixture      Preserve the decoded temporary fixture after the run
  -h, --help          Show this help

The test uses 24 Isaac environments, 8 rollout steps and 2 PPO updates. It
fails unless H20/teleop/SOMA encoders, H20 dynamic/kinematic decoders, the
critic and optimizer all show a real finite update between checkpoints.
EOF
}

source_dir=""
image="${IMAGE:-ghcr.io/cyfarwydd-tian/agile-sonic-training@sha256:6ab2d3aac46c74bb97cae40611262c16163ef1b3aae9cf503ce7e97e2f6b7b59}"
runs_dir="${RUNS_DIR:-${PWD}/runs}"
cache_dir="${CACHE_DIR:-${PWD}/.cache/agile-sonic}"
gpu_device="${GPU_DEVICE:-0}"
keep_fixture=0

while (( $# > 0 )); do
  case "$1" in
    --source) [[ $# -ge 2 ]] || die "--source requires a value"; source_dir="$2"; shift 2 ;;
    --image) [[ $# -ge 2 ]] || die "--image requires a value"; image="$2"; shift 2 ;;
    --runs) [[ $# -ge 2 ]] || die "--runs requires a value"; runs_dir="$2"; shift 2 ;;
    --cache) [[ $# -ge 2 ]] || die "--cache requires a value"; cache_dir="$2"; shift 2 ;;
    --gpu) [[ $# -ge 2 ]] || die "--gpu requires a value"; gpu_device="$2"; shift 2 ;;
    --keep-fixture) keep_fixture=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ -n "${source_dir}" ]] || die "--source is required"
[[ -f "${source_dir}/gear_sonic/train_agent_trl.py" ]] \
  || die "not a sonic-training checkout: ${source_dir}"
require_command python3
require_command docker

fixture_dir="$(mktemp -d -t agile-sonic-smoke.XXXXXXXX)"
cleanup() {
  if (( keep_fixture == 1 )); then
    log "Decoded fixture preserved at ${fixture_dir}"
  else
    rm -rf -- "${fixture_dir}"
  fi
}
trap cleanup EXIT

python3 "${PROJECT_DIR}/smoke/sonic_h20/materialize_fixture.py" "${fixture_dir}"
chmod 0755 "${fixture_dir}" "${fixture_dir}/robot" "${fixture_dir}/soma"
install -m 0444 "${PROJECT_DIR}/smoke/sonic_h20/run_inside.py" "${fixture_dir}/run_inside.py"
install -m 0444 "${PROJECT_DIR}/smoke/sonic_h20/verify_training.py" "${fixture_dir}/verify_training.py"

run_id="sonic-h20-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${run_id}" "${SCRIPT_DIR}/run.sh" \
  --image "${image}" \
  --name "agile-sonic-smoke-${run_id##*-}" \
  --source "${source_dir}" \
  --datasets "${fixture_dir}" \
  --runs "${runs_dir}" \
  --cache "${cache_dir}" \
  --gpu-request "device=${gpu_device}" \
  --gpu-count 1 \
  --ssh-port 0 \
  -- python /datasets/run_inside.py

log "Smoke artifacts: ${runs_dir}/${run_id}/training"
