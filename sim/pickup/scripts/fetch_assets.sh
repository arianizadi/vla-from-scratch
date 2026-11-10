#!/usr/bin/env bash
# Fetch the Franka Panda MJCF model (Apache-2.0) from MuJoCo Menagerie at a pinned commit.
set -euo pipefail
dest="$(dirname "$0")/../assets/third_party/mujoco_menagerie"
commit=98a70c569c4c1b6da467a20012f766ee1f8dc521
if [ ! -f "$dest/franka_emika_panda/panda.xml" ]; then
  rm -rf "$dest"
  git clone -q --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$dest"
  git -C "$dest" sparse-checkout set franka_emika_panda
  git -C "$dest" checkout -q "$commit"
fi
echo "panda model: $dest/franka_emika_panda/panda.xml"
