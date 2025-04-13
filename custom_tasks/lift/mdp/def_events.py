# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Reset the scene to the default state specified in the scene configuration."""
    # rigid bodies
    for rigid_object in env.scene.rigid_objects.values():
        # obtain default and deal with the offset for env origins
        default_root_state = rigid_object.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        rigid_object.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        rigid_object.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
    # articulations
    for articulation_asset in env.scene.articulations.values():
        # obtain default and deal with the offset for env origins
        default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        articulation_asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        articulation_asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        # obtain default joint positions
        default_joint_pos = articulation_asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel[env_ids].clone()
        # set into the physics simulation
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
    # deformable objects
    for deformable_object in env.scene.deformable_objects.values():
        # obtain default and set into the physics simulation
        nodal_state = deformable_object.data.default_nodal_state_w[env_ids].clone()
        deformable_object.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)
        
def reset_nodal_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: DeformableObject = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.nodal_state_w.mean(dim=1).clone()
    root_vel = asset.data.nodal_vel_w.mean(dim=1).clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    # orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    # orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_vel + rand_samples

    # set into the physics simulation
    asset.write_nodal_pos_to_sim(positions, env_ids=env_ids) # type: ignore[assignment]
    asset.write_nodal_velocity_to_sim(velocities, env_ids=env_ids) # type: ignore[assignment]