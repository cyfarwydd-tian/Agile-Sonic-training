#!/usr/bin/env python3
"""Extract validated values from an OCI/Docker image manifest on stdin."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def load_document() -> dict[str, Any]:
    document = json.load(sys.stdin)
    if not isinstance(document, dict):
        raise ValueError("manifest document must be a JSON object")
    return document


def platform_digest(document: dict[str, Any], os_name: str, architecture: str) -> str:
    manifests = document.get("manifests")
    if not isinstance(manifests, list):
        return ""
    for descriptor in manifests:
        if not isinstance(descriptor, dict):
            continue
        platform = descriptor.get("platform")
        annotations = descriptor.get("annotations", {})
        if (
            isinstance(platform, dict)
            and platform.get("os") == os_name
            and platform.get("architecture") == architecture
            and (
                not isinstance(annotations, dict)
                or annotations.get("vnd.docker.reference.type")
                != "attestation-manifest"
            )
        ):
            digest = descriptor.get("digest")
            if isinstance(digest, str) and digest:
                return digest
    return ""


def max_layer_size(document: dict[str, Any]) -> int:
    layers = document.get("layers")
    if not isinstance(layers, list):
        raise ValueError("image manifest has no layers array")
    sizes: list[int] = []
    for descriptor in layers:
        if not isinstance(descriptor, dict):
            raise ValueError("layer descriptor must be a JSON object")
        size = descriptor.get("size")
        if not isinstance(size, int) or size < 0:
            raise ValueError("layer descriptor has an invalid size")
        sizes.append(size)
    return max(sizes, default=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    platform_parser = subparsers.add_parser("platform-digest")
    platform_parser.add_argument("--os", default="linux")
    platform_parser.add_argument("--architecture", default="amd64")
    subparsers.add_parser("max-layer-size")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document = load_document()
    if args.command == "platform-digest":
        print(platform_digest(document, args.os, args.architecture))
    else:
        print(max_layer_size(document))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (json.JSONDecodeError, ValueError) as error:
        print(f"invalid image manifest: {error}", file=sys.stderr)
        raise SystemExit(2) from error
