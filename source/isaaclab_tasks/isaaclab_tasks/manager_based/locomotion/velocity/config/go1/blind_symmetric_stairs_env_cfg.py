# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed pit-climbing evaluation for the symmetric blind Unitree Go1 rough policy."""

from isaaclab.utils import configclass

from .blind_symmetric_rough_env_cfg import UnitreeGo1BlindSymmetricRoughEnvCfg_PLAY


@configclass
class UnitreeGo1BlindSymmetricStairsEnvCfg_PLAY(UnitreeGo1BlindSymmetricRoughEnvCfg_PLAY):
    """Run the symmetric blind policy from the center of a deterministic inverted-pyramid stair pit."""

    def __post_init__(self):
        super().__post_init__()

        terrain_generator = self.scene.terrain.terrain_generator
        terrain_generator.num_rows = 1
        terrain_generator.num_cols = 1
        terrain_generator.curriculum = False
        terrain_generator.difficulty_range = (0.0, 0.0)
        for terrain_cfg in terrain_generator.sub_terrains.values():
            terrain_cfg.proportion = 0.0
        terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 1.0
        self.curriculum.terrain_levels = None

        # The tile has 5 cm stairs rising from the central pit toward its outer edge.
        self.events.reset_base.params = {
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
