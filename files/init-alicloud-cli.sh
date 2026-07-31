#!/usr/bin/env bash
set -Eeuo pipefail

: "${ALICLOUD_CLI_VERSION:?ALICLOUD_CLI_VERSION is required}"
: "${ALICLOUD_CLI_SHA256:?ALICLOUD_CLI_SHA256 is required}"

archive="/tmp/aliyun-cli.tgz"
url="https://github.com/aliyun/aliyun-cli/releases/download/v${ALICLOUD_CLI_VERSION}/aliyun-cli-linux-${ALICLOUD_CLI_VERSION}-amd64.tgz"

curl --fail --location --retry 5 --retry-all-errors \
    --output "${archive}" "${url}"
printf '%s  %s\n' "${ALICLOUD_CLI_SHA256}" "${archive}" | sha256sum --check --strict -
tar -xzf "${archive}" -C /tmp aliyun
install -m 0755 /tmp/aliyun /usr/local/bin/aliyun
aliyun version
rm -f "${archive}" /tmp/aliyun
