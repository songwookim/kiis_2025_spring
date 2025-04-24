# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_CFG`: The UR10 arm without a gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from custom_utils.assets import WORK_DIR
##
# Configuration
##

UR10_DCLAW_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{WORK_DIR}/custom_assets/Robots/ur5e_dclaw_instanceable_ver5.usd",
        # usd_path=f"{WORK_DIR}/custom_assets/Robots/ur5e_dclaw_instanceable_ver4__.usd",
        # usd_path=f"{WORK_DIR}/custom_assets/Robots/urdclaw__.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.57,
            "elbow_joint": 0,
            "wrist_1_joint": -1.57,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": 1.57,
            "joint_f1_0": 0.0,
            "joint_f2_0": 0.0,
            "joint_f3_0": 0.0,
            "joint_f1_1": -0.0,
            "joint_f2_1": -0.0,
            "joint_f3_1": -0.0,
            "joint_f1_2": 0.0,
            "joint_f2_2": 0.0,
            "joint_f3_2": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["^(shoulder_pan_joint|shoulder_lift_joint|elbow_joint|wrist_1_joint|wrist_2_joint|wrist_3_joint)$"],
            velocity_limit=0.0,
            effort_limit=0.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint_f.*"],
            velocity_limit=100.0,
            effort_limit=0.1,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)

UR10_ROBOTIQ_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{WORK_DIR}/custom_assets/Robots/ur10e_robotiq2f-140.usd",
        # usd_path=f"{WORK_DIR}/custom_assets/Robots/ur5e_dclaw_instanceable_ver4__.usd",
        # usd_path=f"{WORK_DIR}/custom_assets/Robots/urdclaw__.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            # max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.57,
            "elbow_joint": 0,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
            'finger_joint' : 0.0,
            'right_outer_knuckle_joint' : 0.0,
            'left_outer_finger_joint' : 0.0,
            'right_outer_finger_joint' : 0.0,
            'left_inner_finger_joint' : 0.0,
            'right_inner_finger_joint' : 0.0,
            'left_inner_finger_pad_joint' : 0.0,
            'right_inner_finger_pad_joint' : 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=0.0,
            effort_limit=0.0,
            stiffness=800.0,
            damping=40.0,
        ),
        # "gripper": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_joint"],
        #     velocity_limit=100.0,
        #     effort_limit=0.1,
        #     stiffness=800.0,
        #     damping=40.0,
        # ),
    },
)

DCLAW_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{WORK_DIR}/custom_assets/Robots/dclaw_ver2.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_f1_0": 0.0,
            "joint_f2_0": 0.0,
            "joint_f3_0": 0.0,
            # "joint_f1_1": -0.0,
            # "joint_f2_1": -0.0,
            # "joint_f3_1": -0.0,
            # "joint_f1_2": 0.0,
            # "joint_f2_2": 0.0,
            # "joint_f3_2": 0.0,
        },
    ),
    actuators={
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=1000.0,
            effort_limit=100.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)


"""Configuration of UR-10 arm using implicit actuator models."""