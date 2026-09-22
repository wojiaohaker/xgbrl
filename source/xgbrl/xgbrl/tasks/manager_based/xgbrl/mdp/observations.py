# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for XGB, matching qiyuan_mc deployment obs format.

The deployment (qiyuan_mc) uses a 48-dim observation vector with specific
scaling factors. These functions produce observations that match the
deployment format exactly, so a policy trained in IsaacLab can be exported
as ONNX and used directly in qiyuan_mc.

Deployment obs layout (48-dim):
    [0:3]   2×base_vel       (from odom network at deployment, GT during training)
    [3:6]   gyro×0.25        (body angular velocity × 0.25)
    [6:9]   (roll, pitch, 0) (Euler angles)
    [9:12]  2×vel_cmd        (velocity command × 2)
    [12:24] jpos_delta       (joint_pos - default_pos, no scaling)
    [24:36] qd×0.05          (joint velocity × 0.05)
    [36:48] last_action      (previous action)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def base_lin_vel_2x(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Body-frame linear velocity multiplied by 2.

    Matches deployment obs[0:3] = 2×base_vel. During training this uses
    ground-truth velocity; at deployment the odom network provides the estimate.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, asset_cfg.joint_ids] * 2.0 \
        if asset_cfg.joint_ids is not None \
        else asset.data.root_lin_vel_b * 2.0


def base_ang_vel_025(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Body-frame angular velocity multiplied by 0.25.

    Matches deployment obs[3:6] = gyro×0.25.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b[:, asset_cfg.joint_ids] * 0.25 \
        if asset_cfg.joint_ids is not None \
        else asset.data.root_ang_vel_b * 0.25


def roll_pitch_zero(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Euler angles (roll, pitch, 0) from root quaternion.

    Matches deployment obs[6:9] = (roll, pitch, 0).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    qw = asset.data.root_quat_w[..., 0]
    qx = asset.data.root_quat_w[..., 1]
    qy = asset.data.root_quat_w[..., 2]
    qz = asset.data.root_quat_w[..., 3]
    # roll = atan2(2(wx+yz), 1-2(x²+y²))
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    # pitch = asin(2(wy-zx))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    zeros = torch.zeros_like(roll)
    return torch.stack([roll, pitch, zeros], dim=-1)


def velocity_commands_2x(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity"
) -> torch.Tensor:
    """Velocity command multiplied by 2.

    Matches deployment obs[9:12] = 2×vel_cmd.
    """
    cmd = env.command_manager.get_command(command_name)
    return cmd * 2.0


def joint_vel_005(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Joint velocity multiplied by 0.05.

    Matches deployment obs[24:36] = qd×0.05.
    Note: joint_vel_rel = joint_vel - default_joint_vel, and default_joint_vel=0,
    so this is equivalent to raw joint_vel × 0.05.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids] * 0.05
