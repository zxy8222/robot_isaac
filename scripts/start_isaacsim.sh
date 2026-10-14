#!/usr/bin/env bash
# Configure workspace-local runtime paths and launch Isaac Sim on a GPU node.
set -euo pipefail

workspace_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$workspace_dir/.venv-isaacsim-6.0.1/bin/python"
runtime_dir="${ISAACSIM_RUNTIME_DIR:-$workspace_dir/.runtime/isaacsim-6.0.1}"

if [[ ! -x "$python_bin" ]]; then
    printf 'Isaac Sim 6.0.1 environment is missing: %s\n' "$python_bin" >&2
    exit 1
fi
if [[ -z "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
    printf 'Run this script inside a Slurm GPU job, not on a login node.\n' >&2
    exit 1
fi

# The cluster supplies Vulkan/OpenGL through this module dependency chain.
if type module >/dev/null 2>&1; then
    module load GCCcore/13.2.0
    module load libglvnd/1.7.0
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'nvidia-smi is unavailable; check GPU allocation and driver modules.\n' >&2
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"$python_bin" -B -c 'import sys; from importlib.metadata import version; assert sys.version_info[:2] == (3, 12), sys.version; assert version("isaacsim") == "6.0.1.0", version("isaacsim")'

mkdir -p "$runtime_dir"/{kit,cache,config,data,logs,tmp,warp}
# The user has accepted the NVIDIA Omniverse EULA for this installation.
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"
export XDG_CACHE_HOME="$runtime_dir/cache"
export XDG_CONFIG_HOME="$runtime_dir/config"
export XDG_DATA_HOME="$runtime_dir/data"
export TMPDIR="$runtime_dir/tmp"
export WARP_CACHE_PATH="$runtime_dir/warp"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$workspace_dir/.runtime/deps/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ "${1:-}" == "--check" ]]; then
    shift
    exec "$python_bin" -u "$workspace_dir/scripts/check_isaacsim.py" \
        --portable-root "$runtime_dir/kit" "$@"
fi
if [[ "${1:-}" == "--python" ]]; then
    shift
    if [[ $# -eq 0 || ! -f "$1" ]]; then
        printf 'Usage: %s --python <existing-script.py> [arguments]\n' "$0" >&2
        exit 2
    fi
    standalone_script="$1"
    shift
    exec "$python_bin" -u "$standalone_script" \
        --portable-root "$runtime_dir/kit" "$@"
fi
exec "$workspace_dir/.venv-isaacsim-6.0.1/bin/isaacsim" \
    isaacsim.exp.full.streaming --no-window \
    --portable-root "$runtime_dir/kit" \
    --/renderer/multiGpu/enabled=false "$@"
