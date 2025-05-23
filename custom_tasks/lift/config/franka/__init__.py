# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents

##
# Register Gym environments.
##

## KIIS 2025 Spring
gym.register(
    id="Isaac-Lift-Def-Cube-Franka-IK-OSC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.osc_env_cfg:FrankaCubeLiftEnvCfg",
        # "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
        # "robomimic_bc_cfg_entry_point": f"{__name__}.robomimic/bc.json",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Def-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaCubeLiftEnvCfg",
        # "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
        "robomimic_bc_cfg_entry_point": f"{__name__}.robomimic/bc.json",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Def-Cube-Franka-Ik-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:FrankaCubeLiftEnvCfg",
        # "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
        "robomimic_bc_cfg_entry_point": f"{__name__}.robomimic/bc.json",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Def-Cube-Franka-Ik-Rel-Imp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_imp_env_cfg:FrankaCubeLiftEnvCfg",
        # "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
        "robomimic_bc_cfg_entry_point": f"{__name__}.robomimic/bc.json",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_imp:LiftCubePPORunnerCfg",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)
