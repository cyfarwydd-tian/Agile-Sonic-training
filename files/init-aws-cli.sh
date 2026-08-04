#!/usr/bin/env bash
set -Eeuo pipefail

: "${AWS_CLI_VERSION:?AWS_CLI_VERSION is required}"
: "${AWS_CLI_SHA256:?AWS_CLI_SHA256 is required}"

archive="/tmp/awscliv2.zip"
extract_dir="/tmp/awscliv2"
url="https://awscli.amazonaws.com/awscli-exe-linux-x86_64-${AWS_CLI_VERSION}.zip"

curl --fail --location --retry 5 --retry-all-errors \
    --output "${archive}" "${url}"
printf '%s  %s\n' "${AWS_CLI_SHA256}" "${archive}" | sha256sum --check --strict -
unzip -q "${archive}" -d "${extract_dir}"
"${extract_dir}/aws/install" \
    --install-dir /usr/local/aws-cli \
    --bin-dir /usr/local/bin
aws --version
rm -rf "${archive}" "${extract_dir}"
