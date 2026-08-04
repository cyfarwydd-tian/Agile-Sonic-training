#!/usr/bin/env python3
"""Decode the checked-in H20 robot/SOMA smoke fixture using only stdlib."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent


def decode_one(source: Path, destination: Path, expected_sha256: str) -> None:
    encoded = "".join(source.read_text(encoding="ascii").split())
    payload = gzip.decompress(base64.b64decode(encoded, validate=True))
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"fixture checksum mismatch for {source.name}: expected {expected_sha256}, got {actual_sha256}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary.write(payload)
        temporary.flush()
        temporary_path = Path(temporary.name)
    temporary_path.chmod(0o444)
    temporary_path.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="directory to create")
    args = parser.parse_args()

    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    output = args.output.resolve()
    decode_one(
        ROOT / "fixtures/robot.pkl.gz.b64",
        output / "robot/neutral_walk_180_R_002__A075.pkl",
        manifest["robot"]["sha256"],
    )
    decode_one(
        ROOT / "fixtures/soma.pkl.gz.b64",
        output / "soma/neutral_walk_180_R_002__A075.pkl",
        manifest["soma"]["sha256"],
    )
    print(f"Materialized paired 120-frame fixture at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
