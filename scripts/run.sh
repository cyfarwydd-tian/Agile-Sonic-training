#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
  cat <<'EOF'
Start the Agile SONIC training container.

Usage:
  scripts/run.sh [options] [-- command [args...]]

Options:
  --image IMAGE        Image reference (default: agile-sonic-training:latest)
  --name NAME          Container name
  --gpu-request VALUE  Docker GPU request: all, N, or device=0,1
  --gpu-count N        Number of worker processes expected inside the container
  --uid UID            Host UID exposed to the entrypoint
  --gid GID            Host GID exposed to the entrypoint
  --env NAME           Conda environment (default: agile-sonic)
  --source PATH        SONIC source tree mounted at /workspace/sonic-training
  --datasets PATH      Dataset directory mounted at /datasets
  --datasets-rw        Mount /datasets read-write (default: read-only)
  --runs PATH          Run/checkpoint directory mounted at /runs
  --cache PATH         Cache directory mounted at /cache
  --ssh-port PORT      Host SSH port; 0 disables publishing (default: 2222)
  -d, --detach         Run in the background
  --keep               Keep the stopped container (default: --rm)
  -h, --help           Show this help

Environment equivalents include IMAGE, CONTAINER_NAME, GPU_REQUEST, GPU_COUNT,
HOST_UID/HOST_GID, SONIC_ENV, RUN_ID, SONIC_SOURCE, DATASETS_DIR, RUNS_DIR,
CACHE_DIR, SSH_PORT, SONIC_EXPECTED_REVISION, and SONIC_REQUIRE_CLEAN. With
--detach and no command the container runs `sleep infinity`; otherwise the
default command is `bash -l`. UID/GID default to the invoking user; direct root
operation uses SUDO_UID/SUDO_GID when present and otherwise 1000:1000.
ENABLE_SSHD/START_SSHD may explicitly override automatic key-based SSH startup.
EOF
}

invoking_uid="$(id -u)"
invoking_gid="$(id -g)"
default_uid="${invoking_uid}"
default_gid="${invoking_gid}"
if [[ "${invoking_uid}" == "0" ]]; then
  default_uid="${SUDO_UID:-1000}"
  default_gid="${SUDO_GID:-1000}"
fi

image="${IMAGE:-agile-sonic-training:latest}"
run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%S.%NZ)}"
safe_run_id="${run_id//[^a-zA-Z0-9_.-]/-}"
container_name="${CONTAINER_NAME:-agile-sonic-training}"
gpu_count="${GPU_COUNT:-}"
gpu_request="${GPU_REQUEST:-}"
host_uid="${HOST_UID:-${default_uid}}"
host_gid="${HOST_GID:-${default_gid}}"
sonic_env="${SONIC_ENV:-${CONDA_ENV_NAME:-agile-sonic}}"
source_dir="${SONIC_SOURCE:-${PWD}}"
datasets_dir="${DATASETS_DIR:-${PWD}/datasets}"
datasets_mode="ro"
runs_dir="${RUNS_DIR:-${PWD}/runs}"
cache_dir="${CACHE_DIR:-${PWD}/.cache/agile-sonic}"
ssh_port="${SSH_PORT:-2222}"
ssh_mode="${ENABLE_SSHD:-${START_SSHD:-auto}}"
detach=0
remove=1
command_args=()

while (( $# > 0 )); do
  case "$1" in
    --image)
      [[ $# -ge 2 ]] || die "--image requires a value"
      image="$2"
      shift 2
      ;;
    --name)
      [[ $# -ge 2 ]] || die "--name requires a value"
      container_name="$2"
      shift 2
      ;;
    --gpu-request)
      [[ $# -ge 2 ]] || die "--gpu-request requires a value"
      gpu_request="$2"
      shift 2
      ;;
    --gpu-count)
      [[ $# -ge 2 ]] || die "--gpu-count requires a value"
      gpu_count="$2"
      shift 2
      ;;
    --uid)
      [[ $# -ge 2 ]] || die "--uid requires a value"
      host_uid="$2"
      shift 2
      ;;
    --gid)
      [[ $# -ge 2 ]] || die "--gid requires a value"
      host_gid="$2"
      shift 2
      ;;
    --env)
      [[ $# -ge 2 ]] || die "--env requires a value"
      sonic_env="$2"
      shift 2
      ;;
    --source)
      [[ $# -ge 2 ]] || die "--source requires a value"
      source_dir="$2"
      shift 2
      ;;
    --datasets)
      [[ $# -ge 2 ]] || die "--datasets requires a value"
      datasets_dir="$2"
      shift 2
      ;;
    --datasets-rw)
      datasets_mode="rw"
      shift
      ;;
    --runs)
      [[ $# -ge 2 ]] || die "--runs requires a value"
      runs_dir="$2"
      shift 2
      ;;
    --cache)
      [[ $# -ge 2 ]] || die "--cache requires a value"
      cache_dir="$2"
      shift 2
      ;;
    --ssh-port)
      [[ $# -ge 2 ]] || die "--ssh-port requires a value"
      ssh_port="$2"
      shift 2
      ;;
    -d|--detach)
      detach=1
      shift
      ;;
    --keep)
      remove=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      command_args=("$@")
      break
      ;;
    *)
      die "Unknown option '$1'; put the container command after --"
      ;;
  esac
done

require_command docker
require_positive_integer UID "${host_uid}"
require_positive_integer GID "${host_gid}"
[[ "${container_name}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || \
  die "Invalid Docker container name '${container_name}'"
[[ "${ssh_port}" =~ ^[0-9]+$ ]] || die "SSH port must be 0 or an integer"
(( ssh_port >= 0 && ssh_port <= 65535 )) || die "SSH port is outside 0-65535"
[[ -d "${source_dir}" ]] || die "SONIC source directory does not exist: ${source_dir}"
[[ -f "${source_dir}/gear_sonic/train_agent_trl.py" ]] \
  || die "SONIC_SOURCE is not a sonic-training checkout: ${source_dir}"
[[ -d "${datasets_dir}" ]] || die "Dataset directory does not exist: ${datasets_dir}"

runs_dir_existed=0
run_dir_existed=0
cache_dir_existed=0
[[ -d "${runs_dir}" ]] && runs_dir_existed=1
[[ -d "${runs_dir}/${safe_run_id}" ]] && run_dir_existed=1
[[ -d "${cache_dir}" ]] && cache_dir_existed=1
mkdir -p "${runs_dir}/${safe_run_id}" "${cache_dir}"
if [[ "${invoking_uid}" == "0" ]]; then
  # A root-operated server should not hand the container a newly-created,
  # root-owned output/cache path. Existing shared paths are never changed.
  (( runs_dir_existed == 1 )) || chown "${host_uid}:${host_gid}" "${runs_dir}"
  (( run_dir_existed == 1 )) || chown "${host_uid}:${host_gid}" "${runs_dir}/${safe_run_id}"
  (( cache_dir_existed == 1 )) || chown "${host_uid}:${host_gid}" "${cache_dir}"
fi
source_dir="$(realpath "${source_dir}")"
datasets_dir="$(realpath "${datasets_dir}")"
runs_dir="$(realpath "${runs_dir}")"
cache_dir="$(realpath "${cache_dir}")"

if [[ -z "${gpu_request}" ]]; then
  if [[ -n "${gpu_count}" ]]; then
    gpu_request="${gpu_count}"
  else
    gpu_request="all"
  fi
fi

requested_count=""
case "${gpu_request}" in
  all)
    if command -v nvidia-smi >/dev/null 2>&1; then
      requested_count="$(nvidia-smi --list-gpus | wc -l)"
      requested_count="${requested_count//[[:space:]]/}"
    fi
    ;;
  device=*)
    visible_devices="${gpu_request#device=}"
    [[ "${visible_devices}" =~ ^[^,[:space:]]+(,[^,[:space:]]+)*$ ]] || \
      die "Invalid GPU device list '${visible_devices}'"
    IFS=',' read -r -a devices <<< "${visible_devices}"
    requested_count="${#devices[@]}"
    ;;
  *)
    if [[ "${gpu_request}" =~ ^[1-9][0-9]*$ ]]; then
      requested_count="${gpu_request}"
    elif [[ "${gpu_request}" =~ ^[^,[:space:]]+(,[^,[:space:]]+)+$ ]]; then
      IFS=',' read -r -a devices <<< "${gpu_request}"
      requested_count="${#devices[@]}"
      gpu_request="device=${gpu_request}"
    else
      die "Unsupported GPU request '${gpu_request}'"
    fi
    ;;
esac

if [[ -z "${gpu_count}" ]]; then
  gpu_count="${requested_count}"
elif [[ -n "${requested_count}" ]]; then
  require_positive_integer GPU_COUNT "${gpu_count}"
  if [[ "${gpu_request}" == "all" ]]; then
    (( gpu_count <= requested_count )) || \
      die "GPU_COUNT=${gpu_count}, but only ${requested_count} host GPUs were detected"
  elif (( gpu_count != requested_count )); then
    die "GPU_COUNT=${gpu_count} conflicts with GPU_REQUEST=${gpu_request} (${requested_count} devices)"
  fi
fi
[[ -n "${gpu_count}" ]] || die \
  "Could not infer GPU_COUNT. Set --gpu-count explicitly on a GPU host."
require_positive_integer GPU_COUNT "${gpu_count}"

# Docker parses --gpus as CSV. Without literal inner quotes, `device=0,1`
# is accepted but silently records only device 0. Keep the unquoted form in
# GPU_REQUEST for diagnostics while quoting the CLI value that contains CSV.
docker_gpu_request="${gpu_request}"
if [[ "${gpu_request}" == device=* && "${gpu_request}" == *,* ]]; then
  docker_gpu_request="\"${gpu_request}\""
fi

docker_args=(
  run
  --name "${container_name}"
  --hostname "${container_name//_/-}"
  --gpus "${docker_gpu_request}"
  --init
  --ipc host
  --cap-add IPC_LOCK
  --ulimit memlock=-1:-1
  --ulimit "nofile=${NOFILE_LIMIT:-1048576}:${NOFILE_LIMIT:-1048576}"
  --ulimit "stack=${STACK_LIMIT:-67108864}:${STACK_LIMIT:-67108864}"
  --log-driver json-file
  --log-opt "max-size=${DOCKER_LOG_MAX_SIZE:-100m}"
  --log-opt "max-file=${DOCKER_LOG_MAX_FILE:-5}"
  --stop-timeout "${STOP_TIMEOUT:-120}"
  --workdir /workspace/sonic-training
  --volume "${source_dir}:/workspace/sonic-training:rw"
  --volume "${datasets_dir}:/datasets:${datasets_mode}"
  --volume "${runs_dir}:/runs:rw"
  --volume "${cache_dir}:/cache:rw"
  --env "HOST_UID=${host_uid}"
  --env "HOST_GID=${host_gid}"
  --env "GPU_COUNT=${gpu_count}"
  --env "GPU_REQUEST=${gpu_request}"
  --env "RUN_ID=${run_id}"
  --env "RUNS_DIR=/runs"
  --env "RUN_DIR=/runs/${safe_run_id}"
  --env "SONIC_ENV=${sonic_env}"
  --env "CONDA_ENV_NAME=${sonic_env}"
  --env "NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
  --env "TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  --env "XDG_CACHE_HOME=/cache/xdg"
  --env "HF_HOME=/cache/huggingface"
  --env "TORCH_HOME=/cache/torch"
  --env "TORCH_EXTENSIONS_DIR=/cache/torch_extensions"
  --env "WANDB_CACHE_DIR=/cache/wandb"
  --env "WANDB_DIR=/runs/${safe_run_id}/wandb"
  --env "FIX_MOUNT_OWNERSHIP=${FIX_MOUNT_OWNERSHIP:-0}"
)

if (( remove == 1 )); then
  docker_args+=(--rm)
fi
if (( detach == 1 )); then
  docker_args+=(--detach)
elif [[ -t 0 && -t 1 ]]; then
  docker_args+=(--interactive --tty)
fi
if (( ssh_port > 0 )); then
  docker_args+=(--publish "${ssh_port}:22")
elif [[ -z "${ENABLE_SSHD+x}" && -z "${START_SSHD+x}" ]]; then
  ssh_mode=0
fi
docker_args+=(--env "ENABLE_SSHD=${ssh_mode}")

for variable in \
  SSH_AUTHORIZED_KEYS SSH_USER_PUBLIC_KEY \
  AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_SECURITY_TOKEN \
  AWS_DEFAULT_REGION \
  AWS_KEY_ID AWS_KEY AWS_REGION \
  ALIBABA_CLOUD_ACCESS_KEY_ID ALIBABA_CLOUD_ACCESS_KEY_SECRET \
  ALIBABA_CLOUD_SECURITY_TOKEN ALIBABA_CLOUD_REGION_ID \
  ALICLOUD_KEY_ID ALICLOUD_KEY ALICLOUD_SECURITY_TOKEN ALICLOUD_REGION \
  AWS_PROFILE \
  HF_TOKEN WANDB_API_KEY WANDB_MODE \
  NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_DISABLE \
  SONIC_EXPECTED_REVISION SONIC_REQUIRE_CLEAN \
  HTTP_PROXY HTTPS_PROXY NO_PROXY \
  ACCEPT_EULA PRIVACY_CONSENT
do
  if [[ -n "${!variable:-}" ]]; then
    docker_args+=(--env "${variable}")
  fi
done

for file_variable in \
  SSH_AUTHORIZED_KEYS_FILE \
  AWS_ACCESS_KEY_ID_FILE AWS_SECRET_ACCESS_KEY_FILE AWS_SESSION_TOKEN_FILE \
  AWS_REGION_FILE AWS_DEFAULT_REGION_FILE \
  AWS_KEY_ID_FILE AWS_KEY_FILE AWS_SECURITY_TOKEN_FILE \
  ALIBABA_CLOUD_ACCESS_KEY_ID_FILE ALIBABA_CLOUD_ACCESS_KEY_SECRET_FILE \
  ALIBABA_CLOUD_SECURITY_TOKEN_FILE ALIBABA_CLOUD_REGION_ID_FILE \
  ALICLOUD_KEY_ID_FILE ALICLOUD_KEY_FILE ALICLOUD_SECURITY_TOKEN_FILE \
  ALICLOUD_REGION_FILE
do
  host_file="${!file_variable:-}"
  if [[ -n "${host_file}" ]]; then
    [[ -f "${host_file}" ]] || die "${file_variable} does not exist: ${host_file}"
    host_file="$(realpath "${host_file}")"
    container_file="/run/secrets/${file_variable,,}"
    docker_args+=(
      --volume "${host_file}:${container_file}:ro"
      --env "${file_variable}=${container_file}"
    )
  fi
done

mount_canonical_config() {
  local canonical_path="$1"
  shift
  local selected_variable=""
  local file_variable
  local host_file

  for file_variable in "$@"; do
    if [[ -n "${!file_variable:-}" ]]; then
      [[ -z "${selected_variable}" ]] \
        || die "set only one of ${selected_variable} and ${file_variable}"
      selected_variable="${file_variable}"
    fi
  done
  [[ -n "${selected_variable}" ]] || return 0

  host_file="${!selected_variable}"
  [[ -f "${host_file}" ]] || die "${selected_variable} does not exist: ${host_file}"
  host_file="$(realpath "${host_file}")"
  # Do not persist this path in Docker Config.Env. The entrypoint discovers the
  # canonical secret, copies it to the user's home and docker exec then uses the
  # CLI's normal default location instead of an unreadable root-owned mount.
  docker_args+=(--volume "${host_file}:${canonical_path}:ro")
}

mount_canonical_config \
  /run/secrets/aws_credentials \
  AWS_SHARED_CREDENTIALS_FILE AWS_CREDENTIALS_FILE
mount_canonical_config /run/secrets/aws_config AWS_CONFIG_FILE
mount_canonical_config \
  /run/secrets/alicloud_config \
  ALIBABA_CLOUD_CONFIG_FILE ALICLOUD_CONFIG_FILE

if (( ${#command_args[@]} == 0 )); then
  if (( detach == 1 )); then
    command_args=(sleep infinity)
  else
    command_args=(bash -l)
  fi
fi

log "Starting ${container_name}: image=${image}, GPUs=${gpu_request} (${gpu_count} workers)"
log "Mounts: source=${source_dir}, datasets=${datasets_dir}, runs=${runs_dir}, cache=${cache_dir}"
exec docker "${docker_args[@]}" "${image}" "${command_args[@]}"
