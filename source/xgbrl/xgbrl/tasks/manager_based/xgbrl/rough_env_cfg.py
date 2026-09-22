# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from xgbrl.tasks.manager_based.xgbrl.mdp.observations import (
    base_lin_vel_2x,
    base_ang_vel_025,
    roll_pitch_zero,
    velocity_commands_2x,
    joint_vel_005,
)
from xgbrl.tasks.manager_based.xgbrl.mdp.rewards import joint_pos_target_l2 as xgb_joint_pos_target_l2

##
# Pre-defined configs
##
from xgbrl.assets.xgb import XGB_CFG  # isort: skip


@configclass
class XgbRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Override observation to match qiyuan_mc deployment obs format (48-dim):
        #   [0:3]   2×base_vel       (GT during training, odom at deployment)
        #   [3:6]   gyro×0.25        (body ang vel × 0.25)
        #   [6:9]   (roll, pitch, 0) (Euler angles)
        #   [9:12]  2×vel_cmd        (velocity command × 2)
        #   [12:24] jpos_delta       (joint_pos - default_pos)
        #   [24:36] qd×0.05          (joint vel × 0.05)
        #   [36:48] last_action
        @configclass
        class MatrixPolicyCfg(ObsGroup):
            base_lin_vel = ObsTerm(
                func=base_lin_vel_2x,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise=Unoise(n_min=-0.1, n_max=0.1),
            )
            base_ang_vel = ObsTerm(
                func=base_ang_vel_025,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise=Unoise(n_min=-0.05, n_max=0.05),
            )
            roll_pitch = ObsTerm(
                func=roll_pitch_zero,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise=Unoise(n_min=-0.05, n_max=0.05),
            )
            velocity_commands = ObsTerm(
                func=velocity_commands_2x,
                params={"command_name": "base_velocity"},
            )
            joint_pos = ObsTerm(
                func=mdp.joint_pos_rel,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise=Unoise(n_min=-0.01, n_max=0.01),
            )
            joint_vel = ObsTerm(
                func=joint_vel_005,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise=Unoise(n_min=-0.075, n_max=0.075),
            )
            actions = ObsTerm(func=mdp.last_action)
            height_scan = None  # disabled; set to None by flat_env_cfg too

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

        # reduce action scale (matches qiyuan_mc deployment: qdes = q_default + 0.25 * action)
        self.actions.joint_pos.scale = 0.25

        # velocity command ranges
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # rewards — re-enabled with XGB body names
        # feet_air_time: xgb foot is part of KNEE_LINK (no separate FOOT body)
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time,
            weight=0.125,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*KNEE_LINK"),
                "command_name": "base_velocity",
                "threshold": 0.5,
            },
        )
        # penalize base_link contact (prevents falling on face/back)
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
        )
        # encourage upright orientation
        self.rewards.flat_orientation_l2.weight = -1.0

        # stand pose regularization (penalize deviation from default joint positions)
        self.rewards.joint_pos_target_abad = RewTerm(
            func=xgb_joint_pos_target_l2,
            weight=-0.1,
            params={"target": 0.0, "asset_cfg": SceneEntityCfg("robot", joint_names=".*_ABAD_JOINT")},
        )
        self.rewards.joint_pos_target_hip = RewTerm(
            func=xgb_joint_pos_target_l2,
            weight=-0.1,
            params={"target": 0.8, "asset_cfg": SceneEntityCfg("robot", joint_names=".*_HIP_JOINT")},
        )
        self.rewards.joint_pos_target_knee = RewTerm(
            func=xgb_joint_pos_target_l2,
            weight=-0.1,
            params={"target": -1.5, "asset_cfg": SceneEntityCfg("robot", joint_names=".*_KNEE_JOINT")},
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
