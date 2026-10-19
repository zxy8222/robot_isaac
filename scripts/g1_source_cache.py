"""Verify and atomically cache immutable upstream files and source manifests."""

import hashlib
import json
import os
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

from g1_outputs import write_json


def git_blob_digest(content: bytes) -> str:
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def repository_tree(repository: str, commit: str) -> dict:
    request = Request(f"https://api.github.com/repos/{repository}/git/trees/{commit}?recursive=1",
                      headers={"User-Agent": "RSS-G1-Mode15"})
    with urlopen(request, timeout=60) as response:
        tree = json.load(response)
    if tree.get("truncated"):
        raise RuntimeError("GitHub returned an incomplete source tree")
    return {entry["path"]: entry for entry in tree["tree"] if entry["type"] == "blob"}


def cache_file(repository: str, commit: str, root: Path, relative: str, entry: dict) -> dict:
    """Validate cached or downloaded bytes; never replace an existing file."""
    destination = root / relative
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("Source destination escapes the cache")
    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{relative}"
    if destination.exists():
        content = destination.read_bytes()
    else:
        with urlopen(url, timeout=120) as response:
            content = response.read(entry["size"]+1)
    if len(content) != entry["size"] or git_blob_digest(content) != entry["sha"]:
        raise ValueError(f"Pinned source checksum mismatch: {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.download")
        try:
            with temporary.open("xb") as output:
                output.write(content)
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {"path": relative, "url": url, "bytes": len(content), "git_blob_sha1": entry["sha"],
            "sha256": hashlib.sha256(content).hexdigest()}


def save_manifest(path: Path, manifest: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Existing source manifest differs from verified files")
    else:
        write_json(path, manifest)
