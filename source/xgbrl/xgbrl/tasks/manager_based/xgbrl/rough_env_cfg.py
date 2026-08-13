# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_tasks.utils import preset

##
# Pre-defined configs
##
from isaaclab_assets.robots.xgb import XGB_CFG  # isort: skip


@configclass
class XgbRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = XGB_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["base_legs"].armature = preset(default=0.0, newton_mjwarp=0.02)
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # velocity command ranges (Matrix: cmpc_x_vel=3.0, cmpc_y_vel=1.0, cmpc_yaw_vel=3.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-3.0, 3.0)

        # rewards
        # xgb foot is part of KNEE_LINK (no separate FOOT_LINK body)
        # Disable feet_air_time for now since contact sensor only detects base_link
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7
        
        # 新增：惩罚侧向移动和转向（强制直线行走）
        from isaaclab.envs import mdp
        from isaaclab.managers import RewardTermCfg as RewTerm
        
        # 惩罚 Y 方向线速度（侧向移动）- 提高权重
        self.rewards.lin_vel_y_l2 = RewTerm(
            func=lambda env: mdp.base_lin_vel(env)[..., 1] ** 2,
            weight=-2.0,  # 从 -1.0 提高到 -2.0
        )
        
        # 惩罚 Z 方向角速度（转向）- 提高权重
        self.rewards.ang_vel_z_l2 = RewTerm(
            func=lambda env: mdp.base_ang_vel(env)[..., 2] ** 2,
            weight=-1.0,  # 从 -0.5 提高到 -1.0
        )

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"

        # fix body name for mass randomization (xgb body is 'base_link', not 'base')
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.base_com.default.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"


@configclass
class XgbRoughEnvCfg_PLAY(XgbRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
