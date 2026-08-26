#!/usr/bin/env bash
# Train the symmetric blind Go1 rough-terrain policy.
# Optional overrides:
#   NUM_ENVS=8192 MAX_ITERATIONS=3000 ./scripts/reinforcement_learning/rsl_rl/train_go1_blind_symmetric.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1500}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/isaac-stack/envs/isaaclab-2.3.2
cd "${ISAACLAB_ROOT}"

TERM=xterm ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Unitree-Go1-Blind-Symmetric-v0 \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --headless \
  "$@"
