"""Download the pinned official G1 29-DOF velocity ONNX deployment and references."""

import hashlib
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

from g1_outputs import write_json
from g1_onnx_policy import (ASSET_CONFIG_RELATIVE, CONFIG_RELATIVE, POLICY_RELATIVE,
                            SOURCE_COMMIT, SOURCE_ROOT, Deployment, POLICY_SHA256, verified_file)
from fetch_g1_urdf import git_blob_digest

FILES = (POLICY_RELATIVE, CONFIG_RELATIVE, ASSET_CONFIG_RELATIVE, "README.md",
         "deploy/include/FSM/State_RLBase.h", "deploy/robots/g1_29dof/src/State_RLBase.cpp",
         "deploy/include/unitree_articulation.h", "deploy/include/isaaclab/manager/observation_manager.h",
         "deploy/include/isaaclab/manager/manager_term_cfg.h",
         "deploy/include/isaaclab/envs/mdp/observations/observations.h",
         "deploy/include/isaaclab/envs/mdp/actions/joint_actions.h",
         "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py",
         "doc/licenses/isaaclab-license.txt")


def fetch_policy():
    repository = "unitreerobotics/unitree_rl_lab"
    request = Request(f"https://api.github.com/repos/{repository}/git/trees/{SOURCE_COMMIT}?recursive=1",
                      headers={"User-Agent": "RSS-G1-ONNX"})
    with urlopen(request, timeout=60) as response:
        tree = json.load(response)
    if tree.get("truncated"):
        raise RuntimeError("Incomplete official Git tree")
    entries = {item["path"]: item for item in tree["tree"] if item["type"] == "blob"}

    def fetch(relative):
        entry, destination = entries[relative], SOURCE_ROOT / relative
        url = f"https://raw.githubusercontent.com/{repository}/{SOURCE_COMMIT}/{relative}"
        if destination.exists():
            content = destination.read_bytes()
        else:
            with urlopen(url, timeout=120) as response:
                content = response.read(entry["size"]+1)
        if len(content) != entry["size"] or git_blob_digest(content) != entry["sha"]:
            raise ValueError(f"Official source checksum mismatch: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.download")
        if not destination.exists():
            try:
                with temporary.open("xb") as output:
                    output.write(content)
                os.link(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return {"path": relative, "url": url, "bytes": len(content), "git_blob_sha1": entry["sha"],
                "sha256": hashlib.sha256(content).hexdigest()}

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(fetch, FILES))
    verified_file(SOURCE_ROOT / POLICY_RELATIVE, POLICY_SHA256)
    Deployment()
    manifest = {"repository": repository, "commit": SOURCE_COMMIT, "files": files,
                "license_reference": "Upstream README: Apache-2.0; retained source notices and Isaac Lab BSD license"}
    path = SOURCE_ROOT / "g1-velocity-source.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Existing source manifest differs")
    else:
        write_json(path, manifest)
    return SOURCE_ROOT / POLICY_RELATIVE


if __name__ == "__main__":
    print(fetch_policy())
