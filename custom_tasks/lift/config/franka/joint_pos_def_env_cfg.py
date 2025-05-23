# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg, DeformableObjectCfg
from isaaclab.sim.schemas.schemas_cfg import DeformableBodyPropertiesCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
# from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from work_dir.custom_tasks.lift.lift_env_cfg import LiftEnvCfg  # isort: skip
from isaaclab.sim.spawners import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
import isaaclab.sim as sim_utils

@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        # # Set Cube as object
        # self.scene.object = DeformableObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     init_state=DeformableObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]), # type: ignore            
        #     spawn=sim_utils.MeshCuboidCfg(
        #         size=(0.04, 0.04, 0.04),
        #         deformable_props=DeformableBodyPropertiesCfg(rest_offset=0.0,contact_offset=0.001),
        #         physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
        #         # physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=5000, elasticity_damping=0.005),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
        #         mass_props=sim_utils.MassPropertiesCfg(
        #             mass=0.1,
        #         ),
        #     )
        #     # spawn=UsdFileCfg( # Teddy Bear
        #     #     usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
        #     #     scale=(0.005, 0.005, 0.005),
        #     # ),
        # )
        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]), # type: ignore            
            debug_vis=False,
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.055, 0.055, 0.055),
                deformable_props=DeformableBodyPropertiesCfg(
                    rest_offset=0.0,
                    contact_offset=0.001,
                    vertex_velocity_damping = 0.1,
                    solver_position_iteration_count= 50,
                    collision_simplification=True,
                    collision_simplification_target_triangle_count=512,  # 삼각형 수 감소 (기본값: 0)
                    simulation_hexahedral_resolution=8,                   # 시뮬레이션 해상도 낮춤
                    ),
                physics_material=sim_utils.DeformableBodyMaterialCfg(
                    poissons_ratio=0.4, 
                    # youngs_modulus=1e5, 
                    youngs_modulus=1.5e6, 
                    dynamic_friction=30.
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
                mass_props=sim_utils.MassPropertiesCfg(
                    mass=0.5,
                ),
            )
        )
        self.scene.replicate_physics = False
         
        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy() # type: ignore
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034], #  type: ignore
                    ),
                ),
            ],
        )
        
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
