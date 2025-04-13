# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from isaaclab.assets import DeformableObjectCfg
import isaaclab.sim as sim_utils
from . import joint_pos_def_env_cfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sensors import FrameTransformerCfg
##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.markers.config import DEFORMABLE_TARGET_MARKER_CFG
from isaaclab.sim.schemas.schemas_cfg import DeformableBodyPropertiesCfg
@configclass
class FrankaCubeLiftEnvCfg(joint_pos_def_env_cfg.FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        
        # self.scene.object = DeformableObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     init_state=DeformableObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]), # type: ignore            
        #     debug_vis=True,
        #     spawn=sim_utils.MeshCuboidCfg(
        #         size=(0.055, 0.055, 0.055),
        #         deformable_props=DeformableBodyPropertiesCfg(
        #             rest_offset=0.0,
        #             contact_offset=0.001,
        #             vertex_velocity_damping = 0.1,
        #             solver_position_iteration_count= 50,
        #             collision_simplification=True,
        #             collision_simplification_target_triangle_count=512,  # 삼각형 수 감소 (기본값: 0)
        #             simulation_hexahedral_resolution=8                   # 시뮬레이션 해상도 낮춤
        #             ),
        #         physics_material=sim_utils.DeformableBodyMaterialCfg(
        #             poissons_ratio=0.4, 
        #             youngs_modulus=1e5, 
        #         ),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
        #         mass_props=sim_utils.MassPropertiesCfg(
        #             mass=0.1,
        #         ),
        #     )
        # )
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        
        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]), # type: ignore            
            debug_vis=True,
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
                scale=(0.005, 0.005, 0.005),
            ),
        )
        
        self.scene.replicate_physics = False


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
