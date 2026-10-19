"""Fetch a pinned official G1 URDF and only its referenced meshes.

Existing cache files are verified against GitHub's immutable Git blob hashes;
downloads are published without overwriting files. Conversion belongs to
g1_urdf_import.py, and the source license is cached alongside the assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

from g1_outputs import file_digest, write_json
from g1_policy import LEG_NAMES
from g1_urdf_config import URDF_MODELS, UrdfModel


def mesh_paths(urdf: Path, expected_joints=LEG_NAMES) -> list[str]:
    """Validate the supported joint set and source-relative mesh paths."""
    robot = ET.parse(urdf).getroot()
    movable = [joint.attrib["name"] for joint in robot.findall("joint") if joint.attrib["type"] != "fixed"]
    if len(movable) != len(expected_joints) or set(movable) != set(expected_joints):
        raise ValueError("The source joint set differs from the selected official G1 model")
    paths = sorted({mesh.attrib["filename"] for mesh in robot.findall(".//mesh")})
    for value in paths:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "meshes" or ":" in value:
            raise ValueError(f"Unsupported source mesh path: {value}")
    return paths


def git_blob_digest(content: bytes) -> str:
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def fetch_assets(model: UrdfModel = URDF_MODELS["unitree12"]) -> Path:
    raw_root = f"https://raw.githubusercontent.com/{model.repository}/{model.commit}"
    url = f"https://api.github.com/repos/{model.repository}/git/trees/{model.commit}?recursive=1"
    with urlopen(Request(url, headers={"User-Agent": "RSS-G1-URDF-import"}), timeout=60) as response:
        tree = json.load(response)
    if tree.get("truncated"):
        raise RuntimeError("GitHub returned an incomplete source tree")
    entries = {entry["path"]: entry for entry in tree["tree"] if entry["type"] == "blob"}

    def fetch(relative: str) -> dict:
        entry = entries[relative]
        destination = model.source_root / relative
        if not destination.resolve().is_relative_to(model.source_root.resolve()):
            raise ValueError("Source destination escapes the cache")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            content = destination.read_bytes()
        else:
            with urlopen(f"{raw_root}/{relative}", timeout=120) as response:
                content = response.read(entry["size"] + 1)
        if len(content) != entry["size"] or git_blob_digest(content) != entry["sha"]:
            raise ValueError(f"Pinned source checksum mismatch: {relative}")
        if not destination.exists():
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.download")
            try:
                with temporary.open("xb") as output:
                    output.write(content)
                os.link(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return {"path": relative, "url": f"{raw_root}/{relative}", "bytes": len(content),
                "git_blob_sha1": entry["sha"], "sha256": hashlib.sha256(content).hexdigest()}

    urdf_info = fetch(str(model.description / model.urdf_path.name))
    if file_digest(model.urdf_path) != model.urdf_sha256:
        raise ValueError("Unexpected G1 URDF SHA256")
    remaining = ["LICENSE", *[str(model.description / path) for path in mesh_paths(model.urdf_path, model.joint_names)]]
    if model.key == "unitree29":
        remaining.append(str(model.description / "README.md"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        source_files = [urdf_info, *pool.map(fetch, remaining)]
    manifest = {"repository": model.repository, "commit": model.commit,
                "urdf_sha256": model.urdf_sha256, "mesh_count": len(mesh_paths(model.urdf_path, model.joint_names)),
                "files": sorted(source_files, key=lambda item: item["path"])}
    if model.manifest_path.exists():
        if json.loads(model.manifest_path.read_text()) != manifest:
            raise ValueError("Existing source manifest differs from verified files")
    else:
        write_json(model.manifest_path, manifest)
    return model.urdf_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=URDF_MODELS, default="unitree12")
    print(fetch_assets(URDF_MODELS[parser.parse_args().model]))
