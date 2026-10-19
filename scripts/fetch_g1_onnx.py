"""Download the pinned official G1 29-DOF velocity ONNX deployment and references."""

from concurrent.futures import ThreadPoolExecutor

from g1_source_cache import cache_file, repository_tree, save_manifest
from g1_onnx_policy import (ASSET_CONFIG_RELATIVE, CONFIG_RELATIVE, POLICY_RELATIVE,
                            SOURCE_COMMIT, SOURCE_ROOT, Deployment, POLICY_SHA256, verified_file)

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
    entries = repository_tree(repository, SOURCE_COMMIT)

    def fetch(relative):
        return cache_file(repository, SOURCE_COMMIT, SOURCE_ROOT, relative, entries[relative])

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(fetch, FILES))
    verified_file(SOURCE_ROOT / POLICY_RELATIVE, POLICY_SHA256)
    Deployment()
    manifest = {"repository": repository, "commit": SOURCE_COMMIT, "files": files,
                "license_reference": "Upstream README: Apache-2.0; retained source notices and Isaac Lab BSD license"}
    save_manifest(SOURCE_ROOT / "g1-velocity-source.json", manifest)
    return SOURCE_ROOT / POLICY_RELATIVE


if __name__ == "__main__":
    print(fetch_policy())
