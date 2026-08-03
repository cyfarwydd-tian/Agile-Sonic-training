#!/usr/bin/env bash
# shellcheck shell=bash

# shellcheck source=files/nccl-runtime.sh
source /usr/local/lib/agile-sonic/nccl-runtime.sh

if [[ -r /run/agile-sonic/runtime.env ]]; then
    # shellcheck source=/dev/null
    source /run/agile-sonic/runtime.env
fi

sonic_env="${SONIC_ENV:-}"
if [[ -z "${sonic_env}" && -r /run/agile-sonic/environment ]]; then
    sonic_env="$(< /run/agile-sonic/environment)"
fi
sonic_env="${sonic_env:-agile-sonic}"

case "${sonic_env}" in
    training) sonic_env=agile-sonic ;;
    tools|sim|teleop|camera|robocasa) sonic_env=agile-sonic-tools ;;
    data|data_collection) sonic_env=agile-sonic-data ;;
    inference) sonic_env=agile-sonic-inference ;;
esac

sonic_prefix="/opt/miniforge3/envs/${sonic_env}"
if [[ -d "${sonic_prefix}/bin" ]]; then
    sonic_filtered_path=""
    IFS=: read -r -a sonic_path_parts <<<"${PATH}"
    for sonic_path_part in "${sonic_path_parts[@]}"; do
        case "${sonic_path_part}" in
            /opt/miniforge3/envs/*/bin|/opt/miniforge3/condabin) continue ;;
        esac
        if [[ -z "${sonic_filtered_path}" ]]; then
            sonic_filtered_path="${sonic_path_part}"
        else
            sonic_filtered_path="${sonic_filtered_path}:${sonic_path_part}"
        fi
    done
    export SONIC_ENV="${sonic_env}"
    export CONDA_ENV_NAME="${sonic_env}"
    export CONDA_DEFAULT_ENV="${sonic_env}"
    export CONDA_PREFIX="${sonic_prefix}"
    export PATH="${sonic_prefix}/bin:/opt/miniforge3/condabin:${sonic_filtered_path}"
    if [[ "${sonic_env}" == "agile-sonic" ]]; then
        agile_sonic_enable_nccl_runtime
    else
        agile_sonic_disable_nccl_runtime
    fi
fi
unset sonic_env sonic_prefix sonic_filtered_path sonic_path_part sonic_path_parts
