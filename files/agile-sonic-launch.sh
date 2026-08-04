#!/usr/bin/env bash
set -Eeuo pipefail

scripts_dir="${AGILE_SONIC_SCRIPTS_DIR:-/opt/agile-sonic/scripts}"
exec "${scripts_dir}/launch-multigpu.sh" "$@"
