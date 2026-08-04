#!/usr/bin/env bash
set -Eeuo pipefail

: "${MINIFORGE_VERSION:?MINIFORGE_VERSION is required}"
: "${MINIFORGE_SHA256:?MINIFORGE_SHA256 is required}"
: "${CONDA_DIR:?CONDA_DIR is required}"

installer="/tmp/Miniforge3-Linux-x86_64.sh"
url="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh"

curl --fail --location --retry 5 --retry-all-errors \
    --output "${installer}" "${url}"
printf '%s  %s\n' "${MINIFORGE_SHA256}" "${installer}" | sha256sum --check --strict -
bash "${installer}" -b -p "${CONDA_DIR}"
rm -f "${installer}"

"${CONDA_DIR}/bin/conda" config --system --set auto_activate_base false
"${CONDA_DIR}/bin/conda" config --system --set always_yes true
"${CONDA_DIR}/bin/conda" config --system --set channel_priority strict
"${CONDA_DIR}/bin/conda" config --system --add pkgs_dirs /opt/conda-pkgs
"${CONDA_DIR}/bin/conda" clean --all --yes
