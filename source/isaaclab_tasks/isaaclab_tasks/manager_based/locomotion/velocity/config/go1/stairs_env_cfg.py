# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic stair-climbing evaluation configuration for Unitree Go1."""

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo1RoughEnvCfg_PLAY


@configclass
class UnitreeGo1StairsEnvCfg_PLAY(UnitreeGo1RoughEnvCfg_PLAY):
    """Evaluate the rough-terrain policy while it climbs a fixed pyramid stair terrain."""

    def __post_init__(self):
        super().__post_init__()

        # Generate one deterministic, upward pyramid-stair tile.  Difficulty 0.3 gives
        # 10.4 cm steps from the default 5--23 cm range.
        terrain_generator = self.scene.terrain.terrain_generator
        terrain_generator.num_rows = 1
        terrain_generator.num_cols = 1
        terrain_generator.curriculum = False
        terrain_generator.difficulty_range = (0.3, 0.3)
        for terrain_cfg in terrain_generator.sub_terrains.values():
            terrain_cfg.proportion = 0.0
        terrain_generator.sub_terrains["pyramid_stairs"].proportion = 1.0
        self.curriculum.terrain_levels = None

        # Start on the low outer edge and point the robot toward the raised center.
        # For this fixed tile, the terrain origin is 7 * 0.104 m above the floor.
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-2.9, -2.9),
                "y": (0.0, 0.0),
                "z": (-0.728, -0.728),
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

        # Keep one forward command for the whole evaluation run.
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
