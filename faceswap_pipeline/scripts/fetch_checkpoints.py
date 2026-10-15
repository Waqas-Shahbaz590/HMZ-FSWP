"""Fetch/verify pinned pretrained FaceSwap checkpoints without external code.

Publisher CRC32 detects corruption; the pinned SHA256 values identify the
already tested baseline assets. They are observed hashes, not publisher-signed
proofs of origin. Downloads and ZIP extraction use validated temporary files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time
import urllib.request
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[1]


def file_hashes(path):
    sha, crc, size = hashlib.sha256(), 0, 0
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            sha.update(chunk)
            crc = zlib.crc32(chunk, crc)
            size += len(chunk)
    return sha.hexdigest(), f"{crc:08x}", size


def verify(path, spec):
    sha, crc, size = file_hashes(path)
    if size != spec["bytes"] or sha != spec["sha256"]:
        raise ValueError(f"Pinned baseline size/SHA256 mismatch: {path}")
    # CRC is checked only when the spec carries one; plain pinned downloads
    # (e.g. the Apache-2.0 hand cascade) have no publisher hash sidecar.
    expected_crc = spec.get("publisher_crc32", spec.get("zip_crc32"))
    if expected_crc is not None and crc != expected_crc:
        raise ValueError(f"Checkpoint CRC32 mismatch: {path}")
    return {**spec, "bytes": size, "sha256": sha, "verified": True}


def valid(path, spec):
    try:
        verify(path, spec)
        return True
    except (OSError, ValueError):
        return False


def download(url, target):
    target = Path(target)
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "WQS-FaceSwap-Evaluation/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream, 1024 * 1024)
            os.replace(partial, target)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2)


def extract_buffalo(archive, output, specs):
    """Only exact expected flat ONNX ZIP members are permitted."""
    pack = output / "buffalo_l"
    pack.mkdir(parents=True, exist_ok=True)
    expected = {Path(spec["file"]).name: spec for spec in specs}
    with zipfile.ZipFile(archive) as container:
        infos = container.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("Checkpoint ZIP members do not match the pinned flat model pack")
        for info in infos:
            spec = expected[info.filename]
            file_mode = info.external_attr >> 16
            if info.is_dir() or stat.S_ISLNK(file_mode) or Path(info.filename).name != info.filename:
                raise ValueError(f"Unsafe checkpoint ZIP member: {info.filename}")
            if info.file_size != spec["bytes"] or f"{info.CRC:08x}" != spec["zip_crc32"]:
                raise ValueError(f"Unexpected ZIP size/CRC: {info.filename}")
            target = pack / info.filename
            if valid(target, spec):
                continue
            temporary = target.with_suffix(".onnx.part")
            with container.open(info) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, 1024 * 1024)
            verify(temporary, spec)
            temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--manifest", type=Path, default=ROOT / "model_manifest.json")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    specs = json.loads((ROOT / "checkpoints.json").read_text(encoding="utf-8-sig"))
    output = args.models.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        relative = Path(spec["file"])
        target = (output / relative).resolve()
        if not target.is_relative_to(output) or relative.is_absolute():
            raise ValueError(f"Invalid pinned checkpoint name: {relative}")
    if not args.verify_only:
        for spec in specs:
            if "publisher_crc32" not in spec:
                continue
            target = output / spec["file"]
            hashfile = target.with_suffix(".hash")
            download(spec["url"].removesuffix(".onnx") + ".hash", hashfile)
            published_crc = hashfile.read_text(encoding="ascii").strip().lower()
            if not re.fullmatch(r"[0-9a-f]{8}", published_crc) or published_crc != spec["publisher_crc32"]:
                raise ValueError(f"Publisher CRC differs from tested baseline: {hashfile}")
            if not valid(target, spec):
                print(f"Downloading {target.name}", flush=True)
                temporary = target.with_suffix(".download")
                download(spec["url"], temporary)
                verify(temporary, spec)
                temporary.replace(target)
        buffalo_specs = [spec for spec in specs if "zip_crc32" in spec]
        if any(not valid(output / spec["file"], spec) for spec in buffalo_specs):
            archive = output / "buffalo_l.zip"
            print("Downloading buffalo_l", flush=True)
            download(buffalo_specs[0]["url"], archive)
            extract_buffalo(archive, output, buffalo_specs)
        # Plain pinned downloads without a publisher hash sidecar (observed
        # SHA256 identifies the tested asset; nothing more is claimed).
        for spec in specs:
            if "publisher_crc32" in spec or "zip_crc32" in spec:
                continue
            target = output / spec["file"]
            if valid(target, spec):
                continue
            print(f"Downloading {target.name}", flush=True)
            temporary = target.with_suffix(".download")
            download(spec["url"], temporary)
            verify(temporary, spec)
            temporary.replace(target)
    manifest = [verify(output / spec["file"], spec) for spec in specs]
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.manifest)
    print(json.dumps({"verified_models": len(manifest), "manifest": str(args.manifest)}), flush=True)


if __name__ == "__main__":
    main()
