"""Fetch only the pinned official Unitree G1 policy; verify before publishing it."""

import os
import uuid
from pathlib import Path
from urllib.request import urlopen

from g1_outputs import file_digest, write_json
from g1_policy import DEFAULT_POLICY_PATH, POLICY_SHA256, POLICY_URL, SOURCE_COMMIT


def fetch_policy(destination: Path = DEFAULT_POLICY_PATH) -> Path:
    if destination.exists():
        if file_digest(destination) != POLICY_SHA256:
            raise ValueError(f"Existing policy checksum mismatch: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".motion-{uuid.uuid4().hex}.download")
    try:
        with urlopen(POLICY_URL, timeout=60) as response, temporary.open("xb") as output:
            output.write(response.read(2_000_000))
        if file_digest(temporary) != POLICY_SHA256:
            raise ValueError("Downloaded official policy checksum mismatch")
        os.link(temporary, destination)  # Atomic publication; never overwrite another file.
        write_json(destination.with_suffix(".json"), {
            "url": POLICY_URL, "commit": SOURCE_COMMIT, "sha256": POLICY_SHA256,
            "license": "BSD-3-Clause; see LICENSES/Unitree-RL-Gym-BSD-3-Clause.txt",
        })
    finally:
        temporary.unlink(missing_ok=True)
    return destination


if __name__ == "__main__":
    print(fetch_policy())
