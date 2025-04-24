# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
import isaaclab.utils.string as string_utils
from isaaclab.managers.action_manager import ActionTerm
from typing import TYPE_CHECKING
from isaaclab.controllers.joint_impedance import JointImpedanceController
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    # from . import actions_cfg
    from .actions_cfg import JointImpedanceActionCfg

class JointImpedanceAction(ActionTerm):
    """Joint Impedance Action term with preprocessing and torque command application."""

    cfg: JointImpedanceActionCfg
    _asset: Articulation
    _controller: JointImpedanceController

    def __init__(self, cfg: JointImpedanceActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Resolve joints
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_dof = len(self._joint_ids)
        if self._num_dof == self._asset.num_joints:
            self._joint_ids = slice(None)

        # Get DOF limits for controller use
        dof_limits = self._asset.data.joint_pos_limits[:, self._joint_ids, :]
        self._mass_matrix = torch.zeros(self.num_envs, self._num_dof, self._num_dof, device=self.device)
        self._gravity = torch.zeros(self.num_envs, self._num_dof, device=self.device)

        # Create the impedance controller
        self._controller = JointImpedanceController(
            cfg=self.cfg.controller,
            num_robots=self.num_envs,
            dof_pos_limits=dof_limits,
            device=self.device,
        )

        # Action tensors
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # Scale tensor
        self._scale = torch.ones_like(self._raw_actions) * torch.tensor(self.cfg.scale, device=self.device)

        # Clip tensor setup (optional)
        if self.cfg.clip is not None:
            self._clip = torch.full((self.num_envs, self.action_dim, 2), float("inf"), device=self.device)
            self._clip[..., 0] *= -1.0
            idxs, _, clip_vals = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
            self._clip[:, idxs] = torch.tensor(clip_vals, device=self.device)
        else:
            self._clip = None

    @property
    def action_dim(self) -> int:
        return self._controller.num_actions

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Pre-process and scale the raw actions."""
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions * self._scale

        if self._clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[..., 0],
                max=self._clip[..., 1],
            )

        self._controller.set_command(self._processed_actions)

    def apply_actions(self):
        """Compute torques and apply them to articulation."""
        dof_pos = self._asset.data.joint_pos[:, self._joint_ids]
        dof_vel = self._asset.data.joint_vel[:, self._joint_ids]

        self._mass_matrix[:] = self._asset.root_physx_view.get_generalized_mass_matrices()[:, self._joint_ids, :][
            :, :, self._joint_ids
        ]
        self._gravity[:] = self._asset.root_physx_view.get_gravity_compensation_forces()[:, self._joint_ids]

        torques = self._controller.compute(dof_pos, dof_vel, self._mass_matrix, self._gravity)
        self._asset.set_joint_effort_target(torques, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0