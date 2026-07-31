#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
  cat <<'EOF'
Build an Agile SONIC container image with Docker Buildx.

Usage:
  scripts/build.sh [options]

Options:
  --image NAME       Image repository (default: agile-sonic-training)
  --tag TAG          Image tag (default: latest)
  --target TARGET    Docker target: training, tensorrt, or full (default: full)
  --platform LIST    Build platform(s) (default: linux/amd64)
  --uid UID          Container user ID (default: current host UID)
  --gid GID          Container group ID (default: current host GID)
  --username NAME    Container username (default: fangzhengtian)
  --builder NAME     Use a specific Buildx builder
  --cache-ref REF    Registry cache reference
  --push             Push the result and supply SBOM/provenance
  --load             Load a single-platform result into local Docker (default)
  --no-cache         Disable layer cache
  -h, --help         Show this help

The IMAGE, TAG, BUILD_TARGET, PLATFORM, CONTAINER_UID/CONTAINER_GID,
CONTAINER_USER, BUILDER, and CACHE_REF environment variables provide the same
defaults.
EOF
}

image="${IMAGE:-agile-sonic-training}"
tag="${TAG:-latest}"
target="${BUILD_TARGET:-full}"
platform="${PLATFORM:-linux/amd64}"
invoking_uid="$(id -u)"
invoking_gid="$(id -g)"
default_uid="${invoking_uid}"
default_gid="${invoking_gid}"
if [[ "${invoking_uid}" == "0" ]]; then
  default_uid="${SUDO_UID:-1000}"
  default_gid="${SUDO_GID:-1000}"
fi
build_uid="${CONTAINER_UID:-${default_uid}}"
build_gid="${CONTAINER_GID:-${default_gid}}"
username="${CONTAINER_USER:-fangzhengtian}"
builder="${BUILDER:-}"
cache_ref="${CACHE_REF:-}"
output_mode="load"
no_cache=0

while (( $# > 0 )); do
  case "$1" in
    --image)
      [[ $# -ge 2 ]] || die "--image requires a value"
      image="$2"
      shift 2
      ;;
    --tag)
      [[ $# -ge 2 ]] || die "--tag requires a value"
      tag="$2"
      shift 2
      ;;
    --target)
      [[ $# -ge 2 ]] || die "--target requires a value"
      target="$2"
      shift 2
      ;;
    --platform)
      [[ $# -ge 2 ]] || die "--platform requires a value"
      platform="$2"
      shift 2
      ;;
    --uid)
      [[ $# -ge 2 ]] || die "--uid requires a value"
      build_uid="$2"
      shift 2
      ;;
    --gid)
      [[ $# -ge 2 ]] || die "--gid requires a value"
      build_gid="$2"
      shift 2
      ;;
    --username)
      [[ $# -ge 2 ]] || die "--username requires a value"
      username="$2"
      shift 2
      ;;
    --builder)
      [[ $# -ge 2 ]] || die "--builder requires a value"
      builder="$2"
      shift 2
      ;;
    --cache-ref)
      [[ $# -ge 2 ]] || die "--cache-ref requires a value"
      cache_ref="$2"
      shift 2
      ;;
    --push)
      output_mode="push"
      shift
      ;;
    --load)
      output_mode="load"
      shift
      ;;
    --no-cache)
      no_cache=1
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

case "${target}" in
  training|tensorrt|full) ;;
  *) die "Unsupported target '${target}'; use training, tensorrt, or full" ;;
esac
require_positive_integer UID "${build_uid}"
require_positive_integer GID "${build_gid}"
[[ "${username}" =~ ^[a-z_][a-z0-9_-]*$ ]] || die "Invalid container username '${username}'"
[[ -n "${image}" && -n "${tag}" ]] || die "Image name and tag cannot be empty"
if [[ "${output_mode}" == "load" && "${platform}" == *,* ]]; then
  die "--load supports one platform only; use --push for multi-platform builds"
fi

require_command docker
docker info >/dev/null
docker buildx version >/dev/null

build_args=(
  buildx build
  --file "${REPO_ROOT}/Dockerfile"
  --target "${target}"
  --platform "${platform}"
  --tag "${image}:${tag}"
  --build-arg "UID=${build_uid}"
  --build-arg "GID=${build_gid}"
  --build-arg "USERNAME=${username}"
)
if git -C "${REPO_ROOT}" rev-parse --verify HEAD >/dev/null 2>&1; then
  build_args+=(--build-arg "IMAGE_REVISION=$(git -C "${REPO_ROOT}" rev-parse HEAD)")
fi
for proxy_variable in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  if [[ -n "${!proxy_variable:-}" ]]; then
    build_args+=(--build-arg "${proxy_variable}")
  fi
done

if [[ -n "${builder}" ]]; then
  build_args+=(--builder "${builder}")
fi
if [[ -n "${cache_ref}" ]]; then
  build_args+=(
    --cache-from "type=registry,ref=${cache_ref}"
    --cache-to "type=registry,ref=${cache_ref},mode=max,image-manifest=true,oci-mediatypes=true"
  )
fi
if (( no_cache == 1 )); then
  build_args+=(--no-cache)
fi
if [[ "${output_mode}" == "push" ]]; then
  build_args+=(--push --provenance=mode=max --sbom=true)
else
  build_args+=(--load)
fi

log "Building ${image}:${tag} (target=${target}, platform=${platform}, output=${output_mode})"
exec docker "${build_args[@]}" "${REPO_ROOT}"
