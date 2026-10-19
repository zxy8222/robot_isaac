"""Fetch the official Mode-15 URDF, referenced meshes, description and license."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from g1_outputs import file_digest
from g1_source_cache import cache_file, repository_tree, save_manifest
import g1_urdf_config as asset


def mesh_paths(urdf: Path) -> list[str]:
    """Validate the supported joint set and source-relative mesh paths."""
    robot = ET.parse(urdf).getroot()
    movable = [joint.attrib["name"] for joint in robot.findall("joint") if joint.attrib["type"] != "fixed"]
    if len(movable) != len(asset.JOINT_NAMES) or set(movable) != set(asset.JOINT_NAMES):
        raise ValueError("The source joint set differs from official Mode 15")
    paths = sorted({mesh.attrib["filename"] for mesh in robot.findall(".//mesh")})
    for value in paths:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "meshes" or ":" in value:
            raise ValueError(f"Unsupported source mesh path: {value}")
    return paths


def fetch_assets() -> Path:
    entries = repository_tree(asset.SOURCE_REPOSITORY, asset.SOURCE_COMMIT)

    def fetch(relative: str) -> dict:
        return cache_file(asset.SOURCE_REPOSITORY, asset.SOURCE_COMMIT, asset.SOURCE_ROOT,
                          relative, entries[relative])

    urdf_info = fetch(str(asset.DESCRIPTION / asset.URDF_PATH.name))
    if file_digest(asset.URDF_PATH) != asset.URDF_SHA256:
        raise ValueError("Unexpected G1 URDF SHA256")
    meshes = mesh_paths(asset.URDF_PATH)
    remaining = ["LICENSE", *[str(asset.DESCRIPTION / path) for path in meshes],
                 str(asset.DESCRIPTION / "README.md")]
    with ThreadPoolExecutor(max_workers=4) as pool:
        source_files = [urdf_info, *pool.map(fetch, remaining)]
    manifest = {"repository": asset.SOURCE_REPOSITORY, "commit": asset.SOURCE_COMMIT,
                "urdf_sha256": asset.URDF_SHA256, "mesh_count": len(meshes),
                "files": sorted(source_files, key=lambda item: item["path"])}
    save_manifest(asset.MANIFEST_PATH, manifest)
    return asset.URDF_PATH


if __name__ == "__main__":
    print(fetch_assets())
