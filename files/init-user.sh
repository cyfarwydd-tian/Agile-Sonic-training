#!/usr/bin/env bash
set -Eeuo pipefail

: "${USERNAME:?USERNAME is required}"
: "${REQUESTED_UID:?REQUESTED_UID is required}"
: "${REQUESTED_GID:?REQUESTED_GID is required}"

is_non_root_id() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]] && (( 10#$1 <= 2147483647 ))
}

if ! [[ "${USERNAME}" =~ ^[a-z_][a-z0-9_-]*$ ]]; then
    echo "Invalid USERNAME: ${USERNAME}" >&2
    exit 2
fi
if ! is_non_root_id "${REQUESTED_UID}" || ! is_non_root_id "${REQUESTED_GID}"; then
    echo "UID and GID must be non-zero decimal IDs no larger than 2147483647" >&2
    exit 2
fi

# NVIDIA's Ubuntu base may reserve UID/GID 1000 for an unused ubuntu account.
if [[ "${USERNAME}" != "ubuntu" ]] && id ubuntu >/dev/null 2>&1; then
    userdel --remove ubuntu 2>/dev/null || true
    if id ubuntu >/dev/null 2>&1; then
        userdel ubuntu
    fi
fi
if [[ "${USERNAME}" != "ubuntu" ]] && getent group ubuntu >/dev/null 2>&1; then
    groupdel ubuntu
fi

uid_owner="$(getent passwd "${REQUESTED_UID}" | cut -d: -f1 || true)"
if [[ -n "${uid_owner}" && "${uid_owner}" != "${USERNAME}" ]]; then
    echo "Requested UID ${REQUESTED_UID} is already owned by ${uid_owner}" >&2
    exit 2
fi

primary_group="$(getent group "${REQUESTED_GID}" | cut -d: -f1 || true)"
if [[ -z "${primary_group}" ]]; then
    primary_group="${USERNAME}"
    groupadd --gid "${REQUESTED_GID}" "${primary_group}"
fi

useradd \
    --uid "${REQUESTED_UID}" \
    --gid "${primary_group}" \
    --create-home \
    --shell /bin/bash \
    --no-log-init \
    "${USERNAME}"
passwd --lock "${USERNAME}"

for supplemental_group in video render; do
    if getent group "${supplemental_group}" >/dev/null 2>&1; then
        usermod --append --groups "${supplemental_group}" "${USERNAME}"
    fi
done

case "${ENABLE_PASSWORDLESS_SUDO:-0}" in
    1|true|TRUE|yes|YES)
        printf '%s ALL=(ALL) NOPASSWD:ALL\n' "${USERNAME}" \
            > "/etc/sudoers.d/${USERNAME}"
        chmod 0440 "/etc/sudoers.d/${USERNAME}"
        ;;
    0|false|FALSE|no|NO)
        ;;
    *)
        echo "ENABLE_PASSWORDLESS_SUDO must be 0 or 1" >&2
        exit 2
        ;;
esac

printf '\nexport LANG=en_US.UTF-8\nexport LC_ALL=en_US.UTF-8\n' \
    >> "/home/${USERNAME}/.bashrc"
chown "${USERNAME}:${primary_group}" "/home/${USERNAME}/.bashrc"
