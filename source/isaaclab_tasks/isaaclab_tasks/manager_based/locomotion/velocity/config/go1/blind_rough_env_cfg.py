# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asymmetric blind-locomotion configuration for Unitree Go1 on rough terrain."""

from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg as LocomotionObservationsCfg

from .rough_env_cfg import UnitreeGo1RoughEnvCfg


@configclass
class UnitreeGo1BlindRoughObservationsCfg(LocomotionObservationsCfg):
    """Give the actor proprioception only and retain terrain heights for the critic."""

    policy: LocomotionObservationsCfg.PolicyCfg = LocomotionObservationsCfg.PolicyCfg()
    critic: LocomotionObservationsCfg.PolicyCfg = LocomotionObservationsCfg.PolicyCfg()


@configclass
class UnitreeGo1BlindRoughEnvCfg(UnitreeGo1RoughEnvCfg):
    """The default Go1 rough task with terrain heights hidden from the actor."""

    observations: UnitreeGo1BlindRoughObservationsCfg = UnitreeGo1BlindRoughObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # The ray caster remains in the scene for privileged critic observations only.
        self.observations.policy.height_scan = None
        self.curriculum.terrain_distribution = CurrTerm(func=mdp.terrain_level_distribution)


@configclass
class UnitreeGo1BlindRoughEnvCfg_PLAY(UnitreeGo1BlindRoughEnvCfg):
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
