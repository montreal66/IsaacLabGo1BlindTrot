# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configuration for asymmetric Go1 blind locomotion."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import UnitreeGo1RoughPPORunnerCfg


@configclass
class UnitreeGo1BlindRoughPPORunnerCfg(UnitreeGo1RoughPPORunnerCfg):
    """Map separate actor and privileged-critic observation groups explicitly."""

    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    experiment_name = "unitree_go1_rough_blind"
