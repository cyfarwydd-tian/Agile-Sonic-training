#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck source=files/nccl-runtime.sh
source /usr/local/lib/agile-sonic/nccl-runtime.sh

die() {
    echo "agile-sonic entrypoint: $*" >&2
    exit 2
}

is_non_root_id() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]] && (( 10#$1 <= 2147483647 ))
}

resolve_secret() {
    local direct_name="$1"
    local file_name="$2"
    local legacy_name="${3:-}"
    local legacy_file_name="${4:-}"
    local direct_value="${!direct_name:-}"
    local file_path="${!file_name:-}"

    if [[ -n "${direct_value}" && -n "${file_path}" ]]; then
        die "set only one of ${direct_name} and ${file_name}"
    fi
    if [[ -n "${file_path}" ]]; then
        [[ -f "${file_path}" && ! -L "${file_path}" ]] \
            || die "${file_name} must name a regular, non-symlink file"
        direct_value="$(<"${file_path}")"
    fi
    if [[ -z "${direct_value}" && -n "${legacy_name}" ]]; then
        direct_value="${!legacy_name:-}"
    fi
    if [[ -z "${direct_value}" && -n "${legacy_file_name}" ]]; then
        file_path="${!legacy_file_name:-}"
        if [[ -n "${file_path}" ]]; then
            [[ -f "${file_path}" && ! -L "${file_path}" ]] \
                || die "${legacy_file_name} must name a regular, non-symlink file"
            direct_value="$(<"${file_path}")"
        fi
    fi
    if [[ "${direct_value}" == *$'\n'* || "${direct_value}" == *$'\r'* ]]; then
        die "${direct_name} must contain exactly one line"
    fi
    printf '%s' "${direct_value}"
}

ensure_private_directory() {
    local path="$1"
    local owner="$2"
    local group="$3"

    [[ ! -L "${path}" ]] || die "refusing symlink credential directory: ${path}"
    [[ ! -e "${path}" || -d "${path}" ]] \
        || die "credential directory path is not a directory: ${path}"
    install -d -m 0700 -o "${owner}" -g "${group}" "${path}"
}

ensure_private_destination() {
    local path="$1"

    [[ ! -L "${path}" ]] || die "refusing symlink credential destination: ${path}"
    [[ ! -e "${path}" || -f "${path}" ]] \
        || die "credential destination is not a regular file: ${path}"
    if [[ -e "${path}" && "$(stat -c %h "${path}")" != 1 ]]; then
        die "refusing multiply-linked credential destination: ${path}"
    fi
}

install_private_file() {
    local source="$1"
    local destination="$2"
    local owner="$3"
    local group="$4"

    [[ -f "${source}" && ! -L "${source}" ]] \
        || die "credential source must be a regular, non-symlink file: ${source}"
    ensure_private_destination "${destination}"
    if [[ "$(readlink -f "${source}")" != "$(readlink -m "${destination}")" ]]; then
        install -m 0600 -o "${owner}" -g "${group}" "${source}" "${destination}"
    else
        chmod 0600 "${destination}"
        chown "${owner}:${group}" "${destination}"
    fi
}

remap_runtime_identity() {
    local user="$1"
    local requested_uid="${HOST_UID:-$(id -u "${user}")}"
    local requested_gid="${HOST_GID:-$(id -g "${user}")}"
    local current_uid current_gid current_group uid_owner gid_owner

    is_non_root_id "${requested_uid}" \
        || die "HOST_UID must be a non-zero decimal ID no larger than 2147483647"
    is_non_root_id "${requested_gid}" \
        || die "HOST_GID must be a non-zero decimal ID no larger than 2147483647"

    current_uid="$(id -u "${user}")"
    current_gid="$(id -g "${user}")"
    current_group="$(id -gn "${user}")"

    if [[ "${requested_gid}" != "${current_gid}" ]]; then
        gid_owner="$(getent group "${requested_gid}" | cut -d: -f1 || true)"
        if [[ -n "${gid_owner}" && "${gid_owner}" != "${current_group}" ]]; then
            # Reuse an existing group rather than renumbering or deleting it.
            usermod --gid "${gid_owner}" "${user}"
        else
            groupmod --gid "${requested_gid}" "${current_group}"
        fi
    fi

    if [[ "${requested_uid}" != "${current_uid}" ]]; then
        uid_owner="$(getent passwd "${requested_uid}" | cut -d: -f1 || true)"
        if [[ -n "${uid_owner}" && "${uid_owner}" != "${user}" ]]; then
            die "HOST_UID ${requested_uid} is already owned by ${uid_owner}"
        fi
        usermod --uid "${requested_uid}" "${user}"
    fi
}

select_sonic_environment() {
    local requested="${SONIC_ENV:-agile-sonic}"
    local canonical prefix path_part filtered_path=""

    case "${requested}" in
        agile-sonic|training)
            canonical="agile-sonic"
            ;;
        agile-sonic-tools|tools|sim|teleop|camera|robocasa)
            canonical="agile-sonic-tools"
            ;;
        agile-sonic-data|data|data_collection)
            canonical="agile-sonic-data"
            ;;
        agile-sonic-inference|inference)
            canonical="agile-sonic-inference"
            ;;
        *)
            die "unknown SONIC_ENV '${requested}'"
            ;;
    esac

    prefix="${CONDA_DIR}/envs/${canonical}"
    [[ -x "${prefix}/bin/python" ]] \
        || die "environment ${canonical} is not installed in this image target"

    # Remove any baked environment bin directory before prepending the selected
    # one, otherwise a missing command could silently fall through to training.
    IFS=: read -r -a path_parts <<<"${PATH}"
    for path_part in "${path_parts[@]}"; do
        case "${path_part}" in
            "${CONDA_DIR}"/envs/*/bin|"${CONDA_DIR}"/condabin)
                continue
                ;;
        esac
        if [[ -z "${filtered_path}" ]]; then
            filtered_path="${path_part}"
        else
            filtered_path="${filtered_path}:${path_part}"
        fi
    done

    export SONIC_ENV="${canonical}"
    export CONDA_ENV_NAME="${canonical}"
    export CONDA_DEFAULT_ENV="${canonical}"
    export CONDA_PREFIX="${prefix}"
    export PATH="${prefix}/bin:${CONDA_DIR}/condabin:${filtered_path}"
    if [[ "${canonical}" == "agile-sonic" ]]; then
        agile_sonic_enable_nccl_runtime
    else
        agile_sonic_disable_nccl_runtime
    fi
    install -d -m 0755 /run/agile-sonic
    printf '%s\n' "${canonical}" > /run/agile-sonic/environment
}

safe_cache_link() {
    local target="$1"
    local link_path="$2"
    local user="$3"
    local group="$4"
    local parent_dir

    if [[ ! -d "${target}" ]]; then
        install -d -m 0755 -o "${user}" -g "${group}" "${target}"
    fi
    if [[ -L "${link_path}" ]]; then
        if [[ "$(readlink "${link_path}")" != "${target}" ]]; then
            echo "agile-sonic entrypoint: keeping existing symlink ${link_path}" >&2
        fi
        return 0
    fi
    if [[ -e "${link_path}" ]]; then
        echo "agile-sonic entrypoint: keeping existing cache path ${link_path}" >&2
        return 0
    fi

    parent_dir="$(dirname "${link_path}")"
    if [[ ! -d "${parent_dir}" ]]; then
        install -d -m 0755 -o "${user}" -g "${group}" "${parent_dir}"
    fi
    ln -s "${target}" "${link_path}"
    chown -h "${user}:${group}" "${link_path}"
}

configure_runtime_cache_links() {
    local user="$1"
    local group="$2"
    local home="$3"
    local writable_path
    local fix_ownership="${FIX_MOUNT_OWNERSHIP:-0}"
    local runtime_run_dir="${RUN_DIR:-/runs/${RUN_ID:-manual}}"

    case "${fix_ownership}" in
        0|false|FALSE|no|NO)
            fix_ownership=0
            ;;
        1|true|TRUE|yes|YES)
            fix_ownership=1
            ;;
        *)
            die "FIX_MOUNT_OWNERSHIP must be 0 or 1"
            ;;
    esac
    runtime_run_dir="$(readlink -m "${runtime_run_dir}")"
    case "${runtime_run_dir}" in
        /runs/*)
            ;;
        *)
            die "RUN_DIR must be a child of /runs"
            ;;
    esac
    export RUN_DIR="${runtime_run_dir}"

    # Do not change ownership of an existing bind mount by default: it may be
    # backed by NFS/root-squash or ACL-managed shared storage. Root creates only
    # missing, exact cache/run subdirectories for the runtime user. Simple local
    # mounts may opt in to a non-recursive ownership fix.
    for writable_path in \
        /cache \
        /runs \
        /cache/pip \
        /cache/xdg \
        /cache/xdg/ov \
        /cache/isaac-portable \
        /cache/huggingface \
        /cache/torch \
        /cache/torch_extensions \
        /cache/wandb \
        /cache/nvidia \
        /cache/nvidia/ComputeCache \
        /cache/nvidia/GLCache \
        /cache/ov-data \
        /runs/nvidia-omniverse \
        /runs/wandb \
        "${runtime_run_dir}"; do
        if [[ ! -e "${writable_path}" ]]; then
            # Try as the final identity first so ACL-enabled root-squash mounts
            # can create their own directories. Fall back to container root for
            # ordinary local paths whose parent is root-owned.
            if ! gosu "${user}" install -d -m 0755 "${writable_path}"; then
                install -d -m 0755 -o "${user}" -g "${group}" "${writable_path}" \
                    || die "cannot create runtime directory: ${writable_path}"
            fi
        elif [[ ! -d "${writable_path}" ]]; then
            die "runtime path is not a directory: ${writable_path}"
        elif [[ "${fix_ownership}" == "1" ]]; then
            chown "${user}:${group}" "${writable_path}"
        fi
    done

    for writable_path in \
        /cache/pip \
        /cache/xdg \
        /cache/xdg/ov \
        /cache/isaac-portable \
        /cache/huggingface \
        /cache/torch \
        /cache/torch_extensions \
        /cache/wandb \
        /cache/nvidia/ComputeCache \
        /cache/nvidia/GLCache \
        /cache/ov-data \
        /runs/nvidia-omniverse \
        "${runtime_run_dir}"; do
        if ! gosu "${user}" test -w "${writable_path}"; then
            die "${writable_path} is not writable by ${user}; fix host ACL/ownership or explicitly set FIX_MOUNT_OWNERSHIP=1 for a simple local bind mount"
        fi
    done

    if [[ ! -e "${home}/.cache" && ! -L "${home}/.cache" ]]; then
        safe_cache_link /cache/xdg "${home}/.cache" "${user}" "${group}"
    else
        safe_cache_link /cache/xdg/ov "${home}/.cache/ov" "${user}" "${group}"
        safe_cache_link \
            /cache/nvidia/GLCache \
            "${home}/.cache/nvidia/GLCache" \
            "${user}" "${group}"
    fi
    safe_cache_link \
        /cache/nvidia/ComputeCache \
        "${home}/.nv/ComputeCache" \
        "${user}" "${group}"
    safe_cache_link \
        /cache/ov-data \
        "${home}/.local/share/ov/data" \
        "${user}" "${group}"
    safe_cache_link \
        /runs/nvidia-omniverse \
        "${home}/.nvidia-omniverse/logs" \
        "${user}" "${group}"
}

write_runtime_state() {
    local state_dir=/run/agile-sonic
    local state_file="${state_dir}/runtime.env"
    local state_tmp

    install -d -m 0755 "${state_dir}"
    state_tmp="$(mktemp "${state_dir}/.runtime.env.XXXXXX")"
    {
        printf '# Generated at container start; contains paths only, never credentials.\n'
        # These references are intentionally written literally for later shells.
        # shellcheck disable=SC2016
        printf 'if [[ -z "${RUN_ID:-}" && -z "${RUN_DIR:-}" ]]; then\n'
        printf '  export RUN_ID=%q\n' "${RUN_ID}"
        printf '  export RUN_DIR=%q\n' "${RUN_DIR}"
        # shellcheck disable=SC2016
        printf '  if [[ -z "${WANDB_DIR:-}" ]]; then\n'
        printf '    export WANDB_DIR=%q\n' "${WANDB_DIR}"
        printf '  fi\n'
        printf 'fi\n'
    } > "${state_tmp}"
    chmod 0644 "${state_tmp}"
    mv -f "${state_tmp}" "${state_file}"
}

configure_ssh() {
    local user="$1"
    local group="$2"
    local home="$3"
    local ssh_dir="${home}/.ssh"
    local authorized_keys="${ssh_dir}/authorized_keys"
    local key_source="${SSH_AUTHORIZED_KEYS_FILE:-}"
    local key_value="${SSH_AUTHORIZED_KEYS:-${SSH_USER_PUBLIC_KEY:-}}"
    local has_key=0
    local ssh_mode="${ENABLE_SSHD:-${START_SSHD:-auto}}"

    if [[ -n "${key_source}" || -n "${key_value}" ]]; then
        ensure_private_directory "${ssh_dir}" "${user}" "${group}"
        if [[ -n "${key_source}" ]]; then
            install_private_file \
                "${key_source}" "${authorized_keys}" "${user}" "${group}"
        else
            ensure_private_destination "${authorized_keys}"
            umask 077
            printf '%s\n' "${key_value}" > "${authorized_keys}"
            chown "${user}:${group}" "${authorized_keys}"
        fi
        [[ -s "${authorized_keys}" ]] || die "SSH authorized_keys is empty"
        ssh-keygen -l -f "${authorized_keys}" >/dev/null \
            || die "SSH authorized_keys contains no valid public key"
        has_key=1
    fi

    case "${ssh_mode}" in
        auto)
            [[ "${has_key}" = 1 ]] || return 0
            ;;
        1|true|TRUE|yes|YES)
            [[ "${has_key}" = 1 ]] \
                || die "ENABLE_SSHD=1 requires an authorized key"
            ;;
        0|false|FALSE|no|NO)
            return 0
            ;;
        *)
            die "ENABLE_SSHD must be auto, 1 or 0"
            ;;
    esac

    ssh-keygen -A
    /usr/sbin/sshd -t
    /usr/sbin/sshd
}

configure_aws() {
    local user="$1"
    local group="$2"
    local home="$3"
    local aws_dir="${home}/.aws"
    local credentials="${aws_dir}/credentials"
    local config="${aws_dir}/config"
    local key_id secret_key session_token region credentials_source config_source

    key_id="$(resolve_secret \
        AWS_ACCESS_KEY_ID AWS_ACCESS_KEY_ID_FILE AWS_KEY_ID AWS_KEY_ID_FILE)"
    secret_key="$(resolve_secret \
        AWS_SECRET_ACCESS_KEY AWS_SECRET_ACCESS_KEY_FILE AWS_KEY AWS_KEY_FILE)"
    session_token="$(resolve_secret \
        AWS_SESSION_TOKEN AWS_SESSION_TOKEN_FILE AWS_SECURITY_TOKEN AWS_SECURITY_TOKEN_FILE)"
    region="$(resolve_secret \
        AWS_REGION AWS_REGION_FILE AWS_DEFAULT_REGION AWS_DEFAULT_REGION_FILE)"

    credentials_source="${AWS_SHARED_CREDENTIALS_FILE:-${AWS_CREDENTIALS_FILE:-}}"
    if [[ -z "${credentials_source}" && -f /run/secrets/aws_credentials ]]; then
        credentials_source=/run/secrets/aws_credentials
    fi
    config_source="${AWS_CONFIG_FILE:-}"
    if [[ -z "${config_source}" && -f /run/secrets/aws_config ]]; then
        config_source=/run/secrets/aws_config
    fi

    if [[ -n "${credentials_source}" \
          && ( -n "${key_id}" || -n "${secret_key}" || -n "${session_token}" ) ]]; then
        die "do not combine an AWS credentials file with field-level AWS credential values"
    fi
    if [[ -n "${config_source}" && -n "${region}" ]]; then
        die "do not combine an AWS config file with a field-level AWS region"
    fi
    if [[ -n "${session_token}" && ( -z "${key_id}" || -z "${secret_key}" ) ]]; then
        die "an AWS session token requires a field-level access key ID and secret key"
    fi

    if [[ -n "${key_id}" || -n "${secret_key}" ]]; then
        [[ -n "${key_id}" && -n "${secret_key}" ]] \
            || die "AWS access key ID and secret key must be supplied together"
        ensure_private_directory "${aws_dir}" "${user}" "${group}"
        ensure_private_destination "${credentials}"
        umask 077
        {
            printf '[default]\n'
            printf 'aws_access_key_id = %s\n' "${key_id}"
            printf 'aws_secret_access_key = %s\n' "${secret_key}"
            if [[ -n "${session_token}" ]]; then
                printf 'aws_session_token = %s\n' "${session_token}"
            fi
        } > "${credentials}"
        chmod 0600 "${credentials}"
        chown "${user}:${group}" "${credentials}"
        export AWS_ACCESS_KEY_ID="${key_id}"
        export AWS_SECRET_ACCESS_KEY="${secret_key}"
        [[ -z "${session_token}" ]] || export AWS_SESSION_TOKEN="${session_token}"
        export AWS_SHARED_CREDENTIALS_FILE="${credentials}"
    elif [[ -n "${credentials_source}" ]]; then
        ensure_private_directory "${aws_dir}" "${user}" "${group}"
        install_private_file \
            "${credentials_source}" "${credentials}" "${user}" "${group}"
        export AWS_SHARED_CREDENTIALS_FILE="${credentials}"
    fi

    if [[ -n "${config_source}" ]]; then
        ensure_private_directory "${aws_dir}" "${user}" "${group}"
        install_private_file "${config_source}" "${config}" "${user}" "${group}"
        export AWS_CONFIG_FILE="${config}"
    elif [[ -n "${region}" ]]; then
        ensure_private_directory "${aws_dir}" "${user}" "${group}"
        ensure_private_destination "${config}"
        umask 077
        printf '[default]\nregion = %s\noutput = json\n' "${region}" > "${config}"
        chmod 0600 "${config}"
        chown "${user}:${group}" "${config}"
        export AWS_REGION="${region}"
        export AWS_DEFAULT_REGION="${region}"
        export AWS_CONFIG_FILE="${config}"
    fi
}

configure_alicloud() {
    local user="$1"
    local group="$2"
    local home="$3"
    local aliyun_dir="${home}/.aliyun"
    local config="${aliyun_dir}/config.json"
    local key_id secret_key security_token region config_source mode

    key_id="$(resolve_secret \
        ALIBABA_CLOUD_ACCESS_KEY_ID ALIBABA_CLOUD_ACCESS_KEY_ID_FILE \
        ALICLOUD_KEY_ID ALICLOUD_KEY_ID_FILE)"
    secret_key="$(resolve_secret \
        ALIBABA_CLOUD_ACCESS_KEY_SECRET ALIBABA_CLOUD_ACCESS_KEY_SECRET_FILE \
        ALICLOUD_KEY ALICLOUD_KEY_FILE)"
    security_token="$(resolve_secret \
        ALIBABA_CLOUD_SECURITY_TOKEN ALIBABA_CLOUD_SECURITY_TOKEN_FILE \
        ALICLOUD_SECURITY_TOKEN ALICLOUD_SECURITY_TOKEN_FILE)"
    region="$(resolve_secret \
        ALIBABA_CLOUD_REGION_ID ALIBABA_CLOUD_REGION_ID_FILE \
        ALICLOUD_REGION ALICLOUD_REGION_FILE)"
    config_source="${ALIBABA_CLOUD_CONFIG_FILE:-${ALICLOUD_CONFIG_FILE:-}}"
    if [[ -z "${config_source}" && -f /run/secrets/alicloud_config ]]; then
        config_source=/run/secrets/alicloud_config
    fi

    if [[ -n "${config_source}" \
          && ( -n "${key_id}" || -n "${secret_key}" \
               || -n "${security_token}" || -n "${region}" ) ]]; then
        die "do not combine an Alibaba Cloud config file with field-level values"
    fi
    if [[ -n "${security_token}" \
          && ( -z "${key_id}" || -z "${secret_key}" ) ]]; then
        die "an Alibaba Cloud security token requires a field-level access key ID and secret"
    fi

    if [[ -n "${key_id}" || -n "${secret_key}" ]]; then
        [[ -n "${key_id}" && -n "${secret_key}" ]] \
            || die "Alibaba Cloud access key ID and secret must be supplied together"
        ensure_private_directory "${aliyun_dir}" "${user}" "${group}"
        ensure_private_destination "${config}"
        mode=AK
        [[ -z "${security_token}" ]] || mode=StsToken
        jq -n \
            --arg mode "${mode}" \
            --arg key_id "${key_id}" \
            --arg secret_key "${secret_key}" \
            --arg security_token "${security_token}" \
            --arg region "${region}" \
            '{
                current: "default",
                profiles: [{
                    name: "default",
                    mode: $mode,
                    access_key_id: $key_id,
                    access_key_secret: $secret_key,
                    sts_token: $security_token,
                    region_id: $region,
                    output_format: "json",
                    language: "en"
                }],
                meta_path: ""
            }' > "${config}"
        chmod 0600 "${config}"
        chown "${user}:${group}" "${config}"
        export ALIBABA_CLOUD_ACCESS_KEY_ID="${key_id}"
        export ALIBABA_CLOUD_ACCESS_KEY_SECRET="${secret_key}"
        [[ -z "${security_token}" ]] \
            || export ALIBABA_CLOUD_SECURITY_TOKEN="${security_token}"
        [[ -z "${region}" ]] || export ALIBABA_CLOUD_REGION_ID="${region}"
    elif [[ -n "${config_source}" ]]; then
        ensure_private_directory "${aliyun_dir}" "${user}" "${group}"
        install_private_file \
            "${config_source}" "${config}" "${user}" "${group}"
    elif [[ -n "${region}" ]]; then
        export ALIBABA_CLOUD_REGION_ID="${region}"
    fi
}

if [[ "$(id -u)" != 0 ]]; then
    die "the image entrypoint must start as root so it can safely drop privileges"
fi

container_user="${CONTAINER_USER:-fangzhengtian}"
id "${container_user}" >/dev/null 2>&1 \
    || die "configured user does not exist: ${container_user}"

remap_runtime_identity "${container_user}"
container_group="$(id -gn "${container_user}")"
container_home="$(getent passwd "${container_user}" | cut -d: -f6)"
chown "${container_user}:${container_group}" "${container_home}"
find "${container_home}" -xdev -mindepth 1 -maxdepth 1 \
    -exec chown -h "${container_user}:${container_group}" {} +
export HOME="${container_home}"
export USER="${container_user}"
export LOGNAME="${container_user}"

if [[ -z "${RUN_ID:-}" ]]; then
    RUN_ID="$(date -u +%Y%m%dT%H%M%S.%NZ)"
    export RUN_ID
fi
safe_run_id="${RUN_ID//[^a-zA-Z0-9_.-]/-}"
export RUN_DIR="${RUN_DIR:-/runs/${safe_run_id}}"
export WANDB_DIR="${WANDB_DIR:-${RUN_DIR}/wandb}"

select_sonic_environment
configure_runtime_cache_links \
    "${container_user}" "${container_group}" "${container_home}"
write_runtime_state
configure_ssh "${container_user}" "${container_group}" "${container_home}"
configure_aws "${container_user}" "${container_group}" "${container_home}"
configure_alicloud "${container_user}" "${container_group}" "${container_home}"

if [[ ! -e /usr/lib/x86_64-linux-gnu/libcuda.so \
      && -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]]; then
    ln -s libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
fi

if [[ "$#" -eq 0 ]]; then
    set -- bash
fi
exec gosu "${container_user}" "$@"
