# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the xgb quadruped robot.

The xgb is a 12-DOF quadruped robot with 3 joints per leg (ABAD, HIP, KNEE).
MJCF model located at: ~/Softwares/Matrix/src/robot_mujoco/zsibot_robots/xgb/
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration - Actuators
##

XGB_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KNEE_JOINT"],
    effort_limit=28.0,
    saturation_effort=28.0,
    velocity_limit=30.0,
    stiffness=20.0,   # Matrix: FSM_RL_ABAD/HIP/KNEE_Kp = 20.0
    damping=0.7,      # Matrix: FSM_RL_Kd = 0.7
    friction=0.0,
)
"""Configuration for xgb leg actuators using DC motor model.

Torque limit: 28 Nm (from MJCF actuatorfrcrange).
"""

##
# Configuration - Articulation
##

XGB_CFG = ArticulationCfg(
    spawn=sim_utils.MjcfFileCfg(
        asset_path="/home/qiyuan/Softwares/Matrix/src/robot_mujoco/zsibot_robots/xgb/xgb.xml",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.32),  # Matrix: body_height = 0.32m
        joint_pos={
            ".*_ABAD_JOINT": 0.0,           # Matrix: abad_stand_pos = 0
            ".*_HIP_JOINT": 0.8,            # Matrix: hip_stand_pos = 0.8
            ".*_KNEE_JOINT": -1.5,          # Matrix: knee_stand_pos = -1.5
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={"base_legs": XGB_ACTUATOR_CFG},
)
"""Configuration of xgb quadruped robot using DC motor model."""
