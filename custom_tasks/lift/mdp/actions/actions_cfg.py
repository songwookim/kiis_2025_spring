# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.controllers.joint_impedance import JointImpedanceControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from .joint_impedance import JointImpedanceAction
##
# Joint actions.
##


from collections.abc import Sequence

# from isaaclab.utils import Range


@configclass
class JointImpedanceActionCfg(ActionTermCfg):
    """Configuration for Joint Impedance Action term."""
    class_type: type[ActionTerm] = JointImpedanceAction
    
    asset_name: str = MISSING
    """The name of the asset in the scene on which the action term is applied."""

    joint_names: Sequence[str] = MISSING
    """The names of the joints to apply the action term."""

    controller: JointImpedanceControllerCfg = MISSING
    """The configuration for the impedance controller."""

    scale: float | Sequence[float] = 1.0
    """The scale factor applied to the input action."""

    clip: dict[str, tuple] | None = None
    """The minimum and maximum action values to clip to.

    Note:
        The keys of the dictionary are regular expressions matched against the joint names.
        The values are tuples or Range objects with (min, max) values.
    """

    use_target_offset: bool = False
    """Whether to apply the action relative to the current joint position."""

    command_type: str = "p_abs"
    """Whether to interpret action command as absolute (p_abs) or relative (p_rel) joint position commands."""

    dof_pos_offset: Sequence[float] | None = None
    """Offset added to position command before computing torque output.

    This offset is useful for tasks where biasing the joint position command
    (e.g. due to elastic elements or persistent interaction) improves behavior.
    """
