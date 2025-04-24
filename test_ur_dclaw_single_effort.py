# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates different single-arm manipulators.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/demos/arms.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import random

import tqdm

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different single-arm manipulators.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import sys
from pathlib import Path
root_path = Path(__file__).resolve().parents[2]  # train.py → custom_rl → work_dir → IsaacLab
sys.path.append(str(root_path))


import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets.articulation import ArticulationCfg, Articulation
from isaacsim.core.prims import Articulation as Art_core
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationData
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
# isort: off
# from omni.isaac.lab_assets import (
#     UR10_CFG,
# )
from custom_utils.assets import WORK_DIR
from custom_assets.universal_robots import UR10_DCLAW_CFG
import os
from isaaclab.assets.deformable_object.deformable_object import DeformableObject
from isaaclab.assets.deformable_object.deformable_object_cfg import DeformableObjectCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import DeformableBodyMaterialCfg
from pxr import UsdPhysics
import isaacsim.core.utils.stage as stage_utils
# isort: on

obj_flag = True

def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_rows = np.floor(np.sqrt(num_origins))
    num_cols = np.ceil(num_origins / num_rows)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    # return the origins
    return env_origins.tolist()


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Each group will have a mount and a robot on top of it
    origins = torch.tensor([0,0,0], device="cuda").reshape(1,3)
    
    # Origin with UR10_DCLAW
    prim_utils.create_prim("/World/Origin", "Xform", translation=origins[0])
    
    # markers
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
                
            ),
        }
    )
    my_visualizer = VisualizationMarkers(marker_cfg)
    
    # -- Table
    cfg = sim_utils.UsdFileCfg(
        usd_path=f"{WORK_DIR}/custom_assets/Objects/rounded_table.usd", scale=(1, 1, 1)
    )
    cfg.func("/World/Origin/Table", cfg, translation=(0.0, 0.0, .0))
    
    # -- Robot
    # ur10_dclaw_cfg : ArticulationCfg = UR10_CFG.replace(prim_path="/World/Origin/Robot")
    ur10_dclaw_cfg : ArticulationCfg = UR10_DCLAW_CFG.replace(prim_path="/World/Origin/Robot")
    ur10_dclaw_cfg.init_state.pos = (-0.3, 0.0, 0.7)
    ur10_dclaw_cfg.spawn.rigid_props.disable_gravity = True
    ur10_dclaw = Articulation(cfg=ur10_dclaw_cfg)
    
    # -- Object
    cfg_cone = sim_utils.MeshCylinderCfg(
        radius=0.045,
        height=0.2525,  
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material =sim_utils.DeformableBodyMaterialCfg(),        
    )
    # for idx, origin in tqdm.tqdm(enumerate(origins), total=len(origins)):
        # randomly select an object to spawn
    obj_cfg : sim_utils.MeshCylinderCfg= cfg_cone
    obj_cfg.physics_material : DeformableBodyMaterialCfg # type: ignore
    # randomize the young modulus (somewhere between a Silicone 30 and Silicone 70)
    obj_cfg.physics_material.youngs_modulus = 3.3e6 #random.uniform(0.7e6, 3.3e6)
    # randomize the poisson's ratio
    obj_cfg.physics_material.poissons_ratio = 0.25
    # randomize the color
    obj_cfg.visual_material.diffuse_color = (random.random(), random.random(), random.random())
    obj_cfg.visual_material.dynamic_friction = 10
    obj_cfg.visual_material.elasticity_damping = 0.01
    
    # spawn the object
    if obj_flag :
        obj_cfg.func(f"/World/Origin/Object1", obj_cfg, translation=(0.17869, 0.143, .83))
            
        def_obj_cfg = DeformableObjectCfg.InitialStateCfg()
        def_obj_cfg.pos = (0., 0.,  0.)
        cfg = DeformableObjectCfg(
            prim_path="/World/Origin/Object1",
            spawn=None,
            init_state=def_obj_cfg,
        )
        deformable_object = DeformableObject(cfg=cfg)
    # return the scene information
    scene_entities = {
        "ur10_dclaw": ur10_dclaw,
        "marker": my_visualizer
        # "deformable_object": deformable_object
    }
    if obj_flag :
        scene_entities = {
            "ur10_dclaw": ur10_dclaw,
            "deformable_object": deformable_object,
            "marker": my_visualizer
        }           
    return scene_entities, origins
import matplotlib.pyplot as plt


"""
00 = 'shoulder_pan_joint'
01 = 'shoulder_lift_joint'
02 = 'elbow_joint'
03 = 'wrist_1_joint'
04 = 'wrist_2_joint'
05 = 'wrist_3_joint'
06 = 'joint_f1_0'
07 = 'joint_f2_0'
08 = 'joint_f3_0'
09 = 'joint_f1_1'
10 = 'joint_f2_1'
11 = 'joint_f3_1'
12 = 'joint_f1_2'
13 = 'joint_f2_2'
14 = 'joint_f3_2'
"""
def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    robot : Articulation= entities["ur10_dclaw"]
    # robot.actuators["arm"].stiffness = torch.zeros_like(robot.actuators["arm"].stiffness ).cuda() # for effort control
    # robot.actuators["arm"].damping = torch.zeros_like(robot.actuators["arm"].damping ).cuda()
    # robot.write_joint_stiffness_to_sim(robot.actuators["arm"].stiffness)
    # robot.write_joint_damping_to_sim(robot.actuators["arm"].damping)
    # robot.actuators["gripper"].stiffness = torch.zeros_like(robot.actuators["gripper"].stiffness ).cuda() # for effort control
    # robot.actuators["gripper"].damping = torch.zeros_like(robot.actuators["gripper"].damping ).cuda()
    # robot.write_joint_stiffness_to_sim(robot.actuators["gripper"].stiffness)
    # robot.write_joint_damping_to_sim(robot.actuators["gripper"].damping)
    
    marker : VisualizationMarkers = entities["marker"] # type: ignore # ignore
        
    robot_core = Art_core(robot.root_physx_view.prim_paths[0])
    robot_core.initialize()
    if obj_flag :
        object : DeformableObject = entities["deformable_object"] # type: ignore
    # Simulate physics
    torque = 0.01
    torque_g = 0.01
    arm_ids = robot.find_joints([".*joint"])[0]
    gripper_ids = robot.find_joints(["joint.*"])[0]
    
    n_arm = len(arm_ids)
    # robot.set_joint_position_target(robot.data.default_joint_pos[:,:n], joint_ids=arm_ids)
    n_gripper = len(gripper_ids)
    
    history_ft = []
    i = 5
    while simulation_app.is_running():
        # reset
        
        if count % 150 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset the scene entities
            
            root_state = robot.data.default_root_state.clone()
            
            root_state[:, :3] += origins[0]
        
            robot.write_root_link_pose_to_sim(root_state[:, :7])
            robot.write_root_com_velocity_to_sim(root_state[:, 7:])
            
            # set joint positions
            default_joint_pos, default_joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            robot.set_joint_position_target(default_joint_pos)
            robot.set_joint_velocity_target(default_joint_vel)
            robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            robot.set_joint_effort_target(torch.zeros_like(robot.data.joint_pos[:,:], device=sim.device))
            
            
            if i > 8:
                i = 5
            i += 1    
            joint_ids = [i]
            target = torch.full((1, 1), 1.)  # shape (N_envs, 1)

            robot.set_joint_effort_target(target=target, joint_ids=joint_ids)
            # if i > 13:
            #     i = 5
            # i += 1
            # joint_ids = [i]
            print(f"[INFO]: Resetting robots state... {i}th joint  {robot.joint_names[i]}")
            
            robot.write_data_to_sim()
            robot.reset()
            robot.update(sim_dt)
            
            

            
                
            if obj_flag:
                object.write_nodal_state_to_sim(object.data.default_nodal_state_w)
                object.reset()
                obstacle_prim = stage_utils.get_current_stage().GetPrimAtPath("/World/Origin/Object1")
                mass_body = UsdPhysics.MassAPI.Apply(obstacle_prim)
                mass_body.CreateMassAttr().Set(0.1)
            
            # clear internal buffers
            robot.reset()
            

            # robot.write_joint_stiffness_to_sim(0.0, joint_ids=arm_ids)
            # robot.write_joint_damping_to_sim(0.0, joint_ids=arm_ids)
            # robot.write_joint_effort_limit_to_sim(0.0, joint_ids=arm_ids)
            # robot.write_joint_velocity_limit_to_sim(0.0, joint_ids=arm_ids)


        # robot.set_joint_effort_target(torch.ones([1,n]).cuda(), joint_ids=gripper_ids)
        # # # robot.write_data_to_sim()
        # robot.set_joint_position_target(robot.data.default_joint_pos)
        # robot.set_joint_position_target(torch.zeros(1).cuda(), joint_ids=[1]) 
        # if count < 250 :
        #     # robot.set_joint_effort_target(torch.ones(1)*torque, joint_ids=[1])
        #     # object.write_nodal_pos_to_sim(object.data.default_nodal_state_w[0,:,:3])
        #     # robot.set_joint_effort_target(torch.ones(1, device=sim.device)*torque_g, joint_ids=[14])
        #     # robot.set_joint_effort_target(torch.ones(1, device=sim.device)*torque, joint_ids=[2])
        #     # robot.set_joint_position_target(torch.tensor([1.2211,1.2211,1.2211],device=sim.device), joint_ids=[12,13,14])
        #     robot.write_data_to_sim()
            
        # elif count < 500 :
        #     # robot.set_joint_effort_target(torch.ones(2).cuda()*torque, joint_ids=[0,1])
        #     # robot.set_joint_position_target(torch.tensor([1.4,1.4,1.4],device=sim.device), joint_ids=[12,13,14])
        #     # robot.set_joint_position_target(torch.tensor([-0.3,-0.3,-0.3],device=sim.device), joint_ids=[9,10,11])
        #     robot.write_data_to_sim()
        # elif count < 750 :
        #     # robot.set_joint_position_target(torch.tensor([-2.791],device=sim.device), joint_ids=[1])
        #     robot.write_data_to_sim()
        #     # robot.set_joint_position_target(torch.zeros(1).cuda(), joint_ids=[0]) 
        # # error = robot.data.joint_pos[:,n_arm:] 

        
        # print(f"effort(inv dyn = computed torque) : {robot_core.get_measured_joint_efforts().round().tolist()}")
        # # print(f"force/torque : {robot_core.get_measured_joint_forces()[15].tolist()}")
        # history_ft.append(robot_core.get_measured_joint_efforts()[0][14].tolist())
        # a = robot.data.body_pos_w[0,13:][:,0]
        # b = robot.data.body_pos_w[0,13:][:,1]
        # c = robot.data.body_pos_w[0,13:][:,2]
        # x = torch.sum(a)
        # y = torch.sum(b)
        # z = torch.sum(c)
        # marker.visualize(robot.data.body_pos_w[0, 16:], robot.data.body_quat_w[0, 15:])
        # robot.set_joint_position_target(torch.zeros(1).cuda(), joint_ids=[0])

        # target = torch.full((1, 1), 1.57)  # shape (N_envs, 1)

        # robot.set_joint_position_target(target=target, joint_ids=joint_ids)
        robot.write_data_to_sim()
        # perform step
        sim.step()
        
        robot.update(sim_dt)
        # update sim-time
        sim_time += sim_dt
        count += 1

        # update buffers
        # for robot in entities.values():
        
    # 데이터 분리
    forces = [data[:3] for data in history_ft]  # Force: F_x, F_y, F_z
    torques = [data[3:] for data in history_ft] # Torque: τ_x, τ_y, τ_z

    # Force와 Torque를 각각 Transpose하여 개별 데이터 시리즈로 분리
    forces = list(zip(*forces))  # F_x, F_y, F_z 각각의 시리즈
    torques = list(zip(*torques))  # τ_x, τ_y, τ_z 각각의 시리즈

    # Plotting
    plt.figure(figsize=(12, 8))

    # Force Plot
    plt.subplot(2, 1, 1)
    for i, force in enumerate(forces):
        plt.plot(force, label=['F_x', 'F_y', 'F_z'][i])
    plt.title('Forces Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Force Values')
    plt.legend()
    plt.grid(True)

    # Torque Plot
    plt.subplot(2, 1, 2)
    for i, torque in enumerate(torques):
        plt.plot(torque, label=['τ_x', 'τ_y', 'τ_z'][i])
    plt.title('Torques Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Torque Values')
    plt.legend()
    plt.grid(True)

    # Show the plots
    plt.tight_layout()
    plt.show()
        
        


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()