#!/usr/bin/env bash
# shellcheck shell=bash

agile_sonic_enable_nccl_runtime() {
    local library="${AGILE_SONIC_NCCL_LIBRARY:-}"
    [[ -n "${library}" && -r "${library}" ]] || return 0
    case ":${LD_PRELOAD:-}:" in
        *":${library}:"*) ;;
        *) export LD_PRELOAD="${library}${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
    esac
}

agile_sonic_disable_nccl_runtime() {
    local library="${AGILE_SONIC_NCCL_LIBRARY:-}"
    local padded
    [[ -n "${library}" && -n "${LD_PRELOAD:-}" ]] || return 0
    padded=":${LD_PRELOAD}:"
    padded="${padded//:${library}:/:}"
    padded="${padded#:}"
    padded="${padded%:}"
    if [[ -n "${padded}" ]]; then
        export LD_PRELOAD="${padded}"
    else
        unset LD_PRELOAD
    fi
}
