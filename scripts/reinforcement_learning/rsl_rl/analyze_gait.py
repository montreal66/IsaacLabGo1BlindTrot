# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless per-environment gait-period analysis for RSL-RL locomotion policies.

Examples:
    # Flat-terrain statistics for 32 independent environments.
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/analyze_gait.py \
        --task Isaac-Velocity-Flat-Unitree-Go1-Play-v0 --checkpoint /path/to/model.pt \
        --num_envs 32 --velocity 0.5 0.0 0.0

    # Measure one chosen rough-terrain tile in every environment.
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/analyze_gait.py \
        --task Isaac-Velocity-Rough-Unitree-Go1-Blind-Symmetric-Play-v0 --checkpoint /path/to/model.pt \
        --num_envs 32 --velocity 0.5 0.0 0.0 --terrain pyramid_stairs:5

    # Measure different terrain type/level pairs in separate environments.
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/analyze_gait.py \
        --task Isaac-Velocity-Rough-Unitree-Go1-Blind-Symmetric-Play-v0 --checkpoint /path/to/model.pt \
        --num_envs 4 --terrain pyramid_stairs:0 --terrain pyramid_stairs:5 \
        --terrain boxes:5 --terrain random_rough:5
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Measure per-foot gait periods without rendering or video recording.")
parser.add_argument("--task", required=True, help="Registered Isaac Lab task ID.")
parser.add_argument("--checkpoint", required=True, help="RSL-RL checkpoint path.")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point", help="Task agent-config registry key.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to evaluate independently.")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of policy steps to simulate after reset.")
parser.add_argument("--warmup_steps", type=int, default=100, help="Initial policy steps excluded from statistics.")
parser.add_argument(
    "--velocity",
    nargs=3,
    type=float,
    metavar=("VX", "VY", "WZ"),
    default=None,
    help="Fix the velocity command for all environments. Omit to use the task command distribution.",
)
parser.add_argument(
    "--spawn_pose",
    nargs=3,
    type=float,
    metavar=("X", "Y", "YAW"),
    default=None,
    help="Fix the reset pose relative to each terrain origin. Useful for controlled uphill tests.",
)
parser.add_argument(
    "--terrain",
    action="append",
    default=[],
    metavar="TYPE:LEVEL",
    help="Force a rough-terrain type and curriculum level. Repeat once per environment, or give one value to repeat.",
)
parser.add_argument("--plane", action="store_true", help="Replace a generator terrain with a plane while keeping the task/model configuration.")
parser.add_argument(
    "--terrain_rows",
    type=int,
    default=10,
    help="Curriculum rows used with --terrain. Default matches the Go1 rough training task.",
)
parser.add_argument(
    "--terrain_cols",
    type=int,
    default=20,
    help="Terrain columns used with --terrain. Default matches the Go1 rough training task.",
)
parser.add_argument("--foot_pattern", default=".*_foot", help="Regular expression selecting foot bodies in contact_forces.")
parser.add_argument("--min_period", type=float, default=0.15, help="Discard shorter stride periods (seconds).")
parser.add_argument("--max_period", type=float, default=1.50, help="Discard longer stride periods (seconds).")
parser.add_argument("--seed", type=int, default=42, help="Environment and terrain seed.")
parser.add_argument("--output_dir", type=Path, default=None, help="Directory for gait_periods.csv and summary.csv.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# This script is analysis-only: never create a viewport, render frames, or RecordVideo output.
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def _parse_terrain_specs(specs: list[str], num_envs: int) -> list[tuple[str, int]]:
    """Parse and expand ``TYPE:LEVEL`` terrain selectors to one selector per environment."""
    if not specs:
        return []
    parsed = []
    for spec in specs:
        try:
            terrain_type, level_text = spec.rsplit(":", maxsplit=1)
            level = int(level_text)
        except ValueError as err:
            raise ValueError(f"Invalid --terrain value '{spec}'; expected TYPE:LEVEL.") from err
        if not terrain_type:
            raise ValueError(f"Invalid --terrain value '{spec}'; terrain type cannot be empty.")
        parsed.append((terrain_type, level))
    if len(parsed) == 1:
        return parsed * num_envs
    if len(parsed) != num_envs:
        raise ValueError("Pass one --terrain selector to repeat it, or exactly --num_envs selectors.")
    return parsed


def _terrain_columns_by_name(terrain_generator) -> dict[str, list[int]]:
    """Map each terrain sub-type to the generator columns allocated to it."""
    sub_terrain_names = list(terrain_generator.sub_terrains)
    proportions = np.asarray([cfg.proportion for cfg in terrain_generator.sub_terrains.values()], dtype=np.float64)
    if np.any(proportions < 0.0) or proportions.sum() <= 0.0:
        raise ValueError("Terrain sub-terrain proportions must be non-negative and sum to a positive value.")
    proportions /= proportions.sum()
    cumulative = np.cumsum(proportions)
    columns = defaultdict(list)
    for column in range(terrain_generator.num_cols):
        sub_index = int(np.min(np.where(column / terrain_generator.num_cols + 0.001 < cumulative)[0]))
        columns[sub_terrain_names[sub_index]].append(column)
    return dict(columns)


def _configure_fixed_command(env_cfg, velocity: tuple[float, float, float] | None):
    """Turn the velocity distribution into one fixed command when requested."""
    if velocity is None:
        return
    vx, vy, wz = velocity
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.ranges.lin_vel_x = (vx, vx)
    command_cfg.ranges.lin_vel_y = (vy, vy)
    command_cfg.ranges.ang_vel_z = (wz, wz)
    command_cfg.ranges.heading = (0.0, 0.0)
    command_cfg.heading_command = False
    command_cfg.rel_heading_envs = 0.0
    command_cfg.rel_standing_envs = 0.0
    command_cfg.resampling_time_range = (1.0e6, 1.0e6)


def _configure_fixed_spawn(env_cfg, spawn_pose: tuple[float, float, float] | None):
    """Remove reset-position and yaw randomization when a controlled start is requested."""
    if spawn_pose is None:
        return
    x, y, yaw = spawn_pose
    pose_range = env_cfg.events.reset_base.params["pose_range"]
    pose_range["x"] = (x, x)
    pose_range["y"] = (y, y)
    pose_range["yaw"] = (yaw, yaw)


def _configure_selected_terrains(
    env_cfg, terrain_specs: list[tuple[str, int]], terrain_rows: int, terrain_cols: int
):
    """Generate curriculum-ordered terrain rows so their type and level can be selected deterministically."""
    if not terrain_specs:
        return
    terrain_generator = env_cfg.scene.terrain.terrain_generator
    if terrain_generator is None:
        raise ValueError("--terrain is only valid for generator-based terrain tasks, not a plane/USD terrain.")
    if terrain_rows <= 0 or terrain_cols <= 0:
        raise ValueError("--terrain_rows and --terrain_cols must both be positive.")
    # Play configs reduce this grid to 5x5. Restore the default rough curriculum grid for meaningful level IDs.
    terrain_generator.num_rows = terrain_rows
    terrain_generator.num_cols = terrain_cols
    terrain_generator.curriculum = True
    # We set origins ourselves after construction, so the task curriculum must not move them at episode reset.
    env_cfg.curriculum.terrain_levels = None


def _configure_plane(env_cfg, use_plane: bool, terrain_specs: list[tuple[str, int]]):
    """Replace the terrain generator with a plane for architecture-compatible flat evaluation."""
    if not use_plane:
        return
    if terrain_specs:
        raise ValueError("--plane and --terrain cannot be used together.")
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.curriculum.terrain_levels = None


def _assign_selected_terrains(base_env, terrain_specs: list[tuple[str, int]]):
    """Assign a requested terrain type/level pair to each environment before the first reset."""
    if not terrain_specs:
        return
    terrain = base_env.scene.terrain
    generator = terrain.cfg.terrain_generator
    if terrain.terrain_origins is None or not hasattr(terrain, "terrain_levels"):
        raise RuntimeError("The task does not expose terrain origins required by --terrain.")
    columns_by_name = _terrain_columns_by_name(generator)
    levels = []
    columns = []
    for terrain_type, level in terrain_specs:
        if terrain_type not in columns_by_name:
            choices = ", ".join(columns_by_name)
            raise ValueError(f"Unknown terrain type '{terrain_type}'. Available types: {choices}.")
        if not 0 <= level < generator.num_rows:
            raise ValueError(f"Terrain level {level} is outside [0, {generator.num_rows - 1}].")
        levels.append(level)
        # Multiple columns of a type differ only by the small generator perturbation; choose the first reproducibly.
        columns.append(columns_by_name[terrain_type][0])
    device = terrain.env_origins.device
    level_tensor = torch.tensor(levels, dtype=torch.long, device=device)
    column_tensor = torch.tensor(columns, dtype=torch.long, device=device)
    terrain.terrain_levels[:] = level_tensor
    terrain.terrain_types[:] = column_tensor
    terrain.env_origins[:] = terrain.terrain_origins[level_tensor, column_tensor]


def _terrain_metadata(base_env, type_names_by_column: dict[int, str]) -> tuple[list[str], list[int | None]]:
    """Return the terrain label currently assigned to each environment."""
    terrain = base_env.scene.terrain
    if not hasattr(terrain, "terrain_levels") or terrain.terrain_levels is None:
        return ["plane"] * base_env.num_envs, [None] * base_env.num_envs
    levels = terrain.terrain_levels.detach().cpu().tolist()
    columns = terrain.terrain_types.detach().cpu().tolist()
    return [type_names_by_column.get(column, f"column_{column}") for column in columns], levels


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Load the policy, simulate it headlessly, and summarize contact-to-contact periods."""
    if args_cli.num_steps <= args_cli.warmup_steps:
        raise ValueError("--num_steps must be greater than --warmup_steps.")
    if not 0.0 < args_cli.min_period < args_cli.max_period:
        raise ValueError("Require 0 < --min_period < --max_period.")

    terrain_specs = _parse_terrain_specs(args_cli.terrain, args_cli.num_envs)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    _configure_fixed_command(env_cfg, args_cli.velocity)
    _configure_fixed_spawn(env_cfg, args_cli.spawn_pose)
    _configure_plane(env_cfg, args_cli.plane, terrain_specs)
    _configure_selected_terrains(env_cfg, terrain_specs, args_cli.terrain_rows, args_cli.terrain_cols)

    checkpoint_path = Path(retrieve_file_path(args_cli.checkpoint)).resolve()
    output_dir = args_cli.output_dir or (
        checkpoint_path.parent / "gait_metrics" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped

    # The terrain origin has to be changed after TerrainImporter exists and before the first reset.
    _assign_selected_terrains(base_env, terrain_specs)
    observations, _ = env.reset()

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner type: {agent_cfg.class_name}.")
    runner.load(str(checkpoint_path))
    policy = runner.get_inference_policy(device=base_env.device)
    policy_nn = runner.alg.policy

    contact_sensor = base_env.scene.sensors.get("contact_forces")
    if contact_sensor is None:
        raise RuntimeError("Task has no 'contact_forces' sensor. Add a contact sensor before gait analysis.")
    foot_ids, foot_names = contact_sensor.find_bodies(args_cli.foot_pattern)
    if len(foot_ids) == 0:
        raise RuntimeError(f"No feet matched --foot_pattern '{args_cli.foot_pattern}'.")

    type_names_by_column = {}
    terrain_generator = base_env.scene.terrain.cfg.terrain_generator
    if terrain_generator is not None:
        for name, columns in _terrain_columns_by_name(terrain_generator).items():
            type_names_by_column.update({column: name for column in columns})

    dt = base_env.step_dt
    episodes = np.zeros(args_cli.num_envs, dtype=np.int64)
    last_contacts: dict[tuple[int, int], tuple[float, int, str, int | None]] = {}
    period_rows: list[dict] = []

    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(observations)
            observations, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        terrain_types, terrain_levels = _terrain_metadata(base_env, type_names_by_column)
        first_contacts = contact_sensor.compute_first_contact(dt)[:, foot_ids].detach().cpu().numpy()
        time_s = (step + 1) * dt
        for env_id, foot_index in np.argwhere(first_contacts):
            foot_index = int(foot_index)
            key = (int(env_id), foot_index)
            current = (time_s, int(episodes[env_id]), terrain_types[env_id], terrain_levels[env_id])
            previous = last_contacts.get(key)
            if previous is not None:
                previous_time, previous_episode, previous_type, previous_level = previous
                period = time_s - previous_time
                same_condition = (
                    previous_episode == current[1]
                    and previous_type == current[2]
                    and previous_level == current[3]
                )
                if (
                    step >= args_cli.warmup_steps
                    and same_condition
                    and args_cli.min_period <= period <= args_cli.max_period
                ):
                    period_rows.append(
                        {
                            "env_id": int(env_id),
                            "episode": current[1],
                            "terrain_type": current[2],
                            "terrain_level": current[3],
                            "foot": foot_names[foot_index],
                            "touchdown_time_s": round(time_s, 6),
                            "period_s": round(period, 6),
                            "frequency_hz": round(1.0 / period, 6),
                        }
                    )
            last_contacts[key] = current
        episodes += dones.detach().cpu().numpy().astype(np.int64)

    group_values: dict[tuple, list[float]] = defaultdict(list)
    for row in period_rows:
        group_values[(row["env_id"], row["episode"], row["terrain_type"], row["terrain_level"], row["foot"])].append(
            row["period_s"]
        )
    summary_rows = []
    for (env_id, episode, terrain_type, terrain_level, foot), periods in sorted(group_values.items()):
        periods_array = np.asarray(periods)
        summary_rows.append(
            {
                "env_id": env_id,
                "episode": episode,
                "terrain_type": terrain_type,
                "terrain_level": terrain_level,
                "foot": foot,
                "samples": len(periods),
                "median_period_s": round(float(np.median(periods_array)), 6),
                "median_frequency_hz": round(float(1.0 / np.median(periods_array)), 6),
                "mean_frequency_hz": round(float(np.mean(1.0 / periods_array)), 6),
                "frequency_p25_hz": round(float(1.0 / np.percentile(periods_array, 75)), 6),
                "frequency_p75_hz": round(float(1.0 / np.percentile(periods_array, 25)), 6),
            }
        )

    _write_csv(
        output_dir / "gait_periods.csv",
        ["env_id", "episode", "terrain_type", "terrain_level", "foot", "touchdown_time_s", "period_s", "frequency_hz"],
        period_rows,
    )
    _write_csv(
        output_dir / "summary.csv",
        [
            "env_id",
            "episode",
            "terrain_type",
            "terrain_level",
            "foot",
            "samples",
            "median_period_s",
            "median_frequency_hz",
            "mean_frequency_hz",
            "frequency_p25_hz",
            "frequency_p75_hz",
        ],
        summary_rows,
    )
    with (output_dir / "run_config.json").open("w") as file:
        json.dump(
            {
                "task": args_cli.task,
                "checkpoint": str(checkpoint_path),
                "num_envs": args_cli.num_envs,
                "num_steps": args_cli.num_steps,
                "warmup_steps": args_cli.warmup_steps,
                "velocity": args_cli.velocity,
                "spawn_pose": args_cli.spawn_pose,
                "terrain": args_cli.terrain,
                "plane": args_cli.plane,
                "terrain_rows": args_cli.terrain_rows,
                "terrain_cols": args_cli.terrain_cols,
                "seed": args_cli.seed,
                "step_dt_s": dt,
            },
            file,
            indent=2,
        )

    print(f"[INFO] Wrote {len(period_rows)} valid stride periods to: {output_dir}")
    print(f"[INFO] Per-environment summaries: {output_dir / 'summary.csv'}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
