# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Symmetric blind-locomotion configuration for Unitree Go1 on rough terrain."""

from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg as LocomotionObservationsCfg

from .rough_env_cfg import UnitreeGo1RoughEnvCfg


@configclass
class UnitreeGo1BlindSymmetricRoughObservationsCfg(LocomotionObservationsCfg):
    """Give actor and critic the same proprioceptive observations."""

    policy: LocomotionObservationsCfg.PolicyCfg = LocomotionObservationsCfg.PolicyCfg()
    critic: LocomotionObservationsCfg.PolicyCfg = LocomotionObservationsCfg.PolicyCfg()


@configclass
class UnitreeGo1BlindSymmetricRoughEnvCfg(UnitreeGo1RoughEnvCfg):
    """Default Go1 rough training with no terrain-height observations for either network."""

    observations: UnitreeGo1BlindSymmetricRoughObservationsCfg = UnitreeGo1BlindSymmetricRoughObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # No observation group uses height data, so disable the scanner itself as well.
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_distribution = CurrTerm(func=mdp.terrain_level_distribution)


@configclass
class UnitreeGo1BlindSymmetricRoughEnvCfg_PLAY(UnitreeGo1BlindSymmetricRoughEnvCfg):
    """Inference configuration matching the default Go1 rough play setup."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
