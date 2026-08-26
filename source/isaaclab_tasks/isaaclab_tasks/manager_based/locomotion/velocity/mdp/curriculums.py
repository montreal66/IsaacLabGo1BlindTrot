# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def terrain_level_distribution(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> dict[str, torch.Tensor]:
    """Report the current number of environments at every terrain-type and curriculum-level pair.

    This term deliberately does not update the curriculum.  It is intended to be used directly after
    :func:`terrain_levels_vel`, which updates the levels for environments that have just reset.  The returned
    dictionary is expanded into individual TensorBoard scalars by :class:`CurriculumManager`.
    """
    del env_ids

    terrain: TerrainImporter = env.scene.terrain
    terrain_generator = terrain.cfg.terrain_generator
    if terrain_generator is None or terrain.terrain_origins is None:
        return {}

    # ``terrain_types`` stores a curriculum-grid column, not a sub-terrain index.  Recreate the same column-to-type
    # assignment used by TerrainGenerator._generate_curriculum_terrains().
    terrain_names = list(terrain_generator.sub_terrains)
    proportions = torch.tensor(
        [sub_terrain.proportion for sub_terrain in terrain_generator.sub_terrains.values()],
        device=terrain.device,
        dtype=torch.float,
    )
    proportions = torch.cumsum(proportions / proportions.sum(), dim=0)
    column_positions = torch.arange(terrain_generator.num_cols, device=terrain.device, dtype=torch.float)
    column_positions = column_positions / terrain_generator.num_cols + 0.001
    column_to_type = torch.searchsorted(proportions, column_positions, right=True)

    stats: dict[str, torch.Tensor] = {}
    for terrain_index, terrain_name in enumerate(terrain_names):
        type_mask = column_to_type[terrain.terrain_types] == terrain_index
        for level in range(terrain.max_terrain_level):
            stats[f"{terrain_name}/level_{level}"] = torch.count_nonzero(
                type_mask & (terrain.terrain_levels == level)
            )
    return stats
