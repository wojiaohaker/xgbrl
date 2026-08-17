# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from xgbrl.assets.xgb import XGB_CFG  # isort: skip


@configclass
class XgbRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Override observation order to match Matrix robot_mc format:
        #   projected_gravity(3) → base_ang_vel(3) → base_lin_vel(3) →
        #   velocity_commands(3) → joint_pos_rel(12) → joint_vel_rel(12) → last_action(12) = 48
        @configclass
        class MatrixPolicyCfg(ObsGroup):
            projected_gravity = ObsTerm(
                func=mdp.projected_gravity,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise=Unoise(n_min=-0.05, n_max=0.05),
            )
            base_ang_vel = ObsTerm(
                func=mdp.base_ang_vel,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise=Unoise(n_min=-0.2, n_max=0.2),
            )
            base_lin_vel = ObsTerm(
                func=mdp.base_lin_vel,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise=Unoise(n_min=-0.1, n_max=0.1),
            )
            velocity_commands = ObsTerm(
                func=mdp.generated_commands,
                params={"command_name": "base_velocity"},
            )
            joint_pos = ObsTerm(
                func=mdp.joint_pos_rel,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise=Unoise(n_min=-0.01, n_max=0.01),
            )
            joint_vel = ObsTerm(
                func=mdp.joint_vel_rel,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise=Unoise(n_min=-1.5, n_max=1.5),
            )
            actions = ObsTerm(func=mdp.last_action)
            height_scan = None  # disabled, set to None by flat_env_cfg

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        @configclass
        class MatrixObsCfg:
            policy: MatrixPolicyCfg = MatrixPolicyCfg()

        self.observations = MatrixObsCfg()

        self.scene.robot = XGB_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["base_legs"].armature = 0.0
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # velocity command ranges (官方默认 ±1.0，匹配 XGB 物理能力)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # rewards — 使用官方默认权重，不做额外修改
        # xgb foot is part of KNEE_LINK (no separate FOOT_LINK body)
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = 0.0  # 官方默认禁用

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
