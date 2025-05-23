        
        # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to work with the deformable object and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_deformable_object.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on interacting with a deformable object.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.sim import SimulationContext
from isaaclab.app import AppLauncher
import isaaclab.utils.math as math_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR   # 필요한 경우 사용


def design_scene():
    # Ground
    gp_cfg = sim_utils.GroundPlaneCfg()
    gp_cfg.func("/World/defaultGroundPlane", gp_cfg)

    # Light
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # 여러 Origin Xform 생성
    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # Deformable Object (Cube Mesh)
    cube_cfg = DeformableObjectCfg(
        prim_path="/World/Origin.*/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.2),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        debug_vis=True,
    )
    cube_object = DeformableObject(cfg=cube_cfg)

    return {"cube_object": cube_object}, torch.tensor(origins)


# ----------------------------------------------------------------
# 3) "frame" 마커 정의 함수
# ----------------------------------------------------------------
def define_frame_marker():
    """
    'frame' USD를 하나만 사용하는 VisualizationMarkersCfg 예시
    (필요하다면 markers 딕셔너리에 다른 아이템들도 추가 가능)
    """
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Markers",  # 원하는 경로 지정
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.25, 0.25, 0.25),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)

def average_quaternions_batched(quats: torch.Tensor) -> torch.Tensor:
    """
    quats: (num_envs, num_quats, 4), 쿼터니언은 wxyz 순서
    반환: (num_envs, 4) 각 env 별 평균 쿼터니언 (wxyz 순서)
    """
    N, M, _ = quats.shape
    out = torch.empty(N, 4, dtype=quats.dtype, device=quats.device)

    for i in range(N):
        # i번째 env의 모든 쿼터니언 (shape: (M,4))
        q = quats[i].clone()  # clone하여 원본 수정 방지

        # 부호 정렬: 첫번째 쿼터니언과의 dot product가 음수인 경우 부호 뒤집기
        ref = q[0].clone()
        dots = (q * ref).sum(dim=-1)
        mask = dots < 0.0
        q[mask] = -q[mask]

        # 모든 쿼터니언의 합을 구한 후 정규화
        q_sum = q.sum(dim=0)
        norm_val = torch.linalg.norm(q_sum)
        if norm_val < 1e-6:
            out[i] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=quats.device, dtype=quats.dtype)
        else:
            out[i] = q_sum / norm_val

    return out


import omni
from pxr import Usd, UsdGeom, Gf


# ----------------------------------------------------------------
# 4) 메인 시뮬레이션 루프
# ----------------------------------------------------------------
def run_simulator(sim: SimulationContext, entities: dict, origins: torch.Tensor, marker):
    """
    - deformable 객체 시뮬레이션
    - 매 step마다 cube_object의 평균 위치/회전(pos, quat) 추출
    - frame 마커를 pos, quat 에 맞춰 시각화
    """
    cube_object : DeformableObject = entities["cube_object"]
    sim_dt = sim.get_physics_dt()
    count = 0

    # nodal kinematic target(초기화 용)
    nodal_kinematic_target = cube_object.data.nodal_kinematic_target.clone()


    stage = omni.usd.get_context().get_stage()
    # result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cone")
    # Get the prim
    prim = stage.GetPrimAtPath("/World/Origin0/Cube")
    # Get the size
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    bbox_cache.Clear()
    while simulation_app.is_running():
        # 일정 주기마다 리셋
        if count % 250 == 0:
            count = 0
            # reset deformable
            nodal_state = cube_object.data.default_nodal_state_w.clone()

            # random으로 위에 놓기 (4개의 origin 각각 약간 랜덤)
            pos_w = torch.rand(cube_object.num_instances, 3, device=sim.device) * 0.1 + origins
            quat_w = math_utils.random_orientation(cube_object.num_instances, device=sim.device)
            nodal_state[..., :3] = cube_object.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)

            # write nodal state
            cube_object.write_nodal_state_to_sim(nodal_state)

            # kinematic target 갱신 (모두 free로, 위치만 맞춤)
            nodal_kinematic_target[..., :3] = nodal_state[..., :3]
            nodal_kinematic_target[..., 3] = 1.0
            cube_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

            cube_object.reset()

            print("[INFO]: Resetting object state...")

        # 특정 vertex를 kinematic constraint 예시
        nodal_kinematic_target[[0, 3], 0, 2] += 0.001
        nodal_kinematic_target[[0, 3], 0, 3] = 0.0
        cube_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

        # deformable 객체 상태 write
        cube_object.write_data_to_sim()

        # 물리 스텝
        sim.step()
        count += 1

        # deformable 객체 내부 버퍼 업데이트
        cube_object.update(sim_dt)


        prim_bbox = bbox_cache.ComputeWorldBound(prim)
        prim_range = prim_bbox.ComputeAlignedRange()
        prim_size = prim_range.GetSize()
        print(prim_bbox.GetVolume())
    
        # ---------------------------
        # (중요) 여기서 pos, quat 값을 받아 마커 시각화
        # ---------------------------
        # 여러 인스턴스가 있을 수 있으므로 shape: (N,3), (N,4)
        
        pos = cube_object.data.root_pos_w        # 위치 cube_object.data.nodal_pos_w.mean(dim=1)      
        # quat = cube_object.data.sim_element_quat_w.mean(dim=1)  # 평균 쿼터니언
        quat = cube_object.data.sim_element_quat_w
        quat = quat / quat.norm(dim=-1, keepdim=True)
        quat = average_quaternions_batched(quat)
        # 모든 인스턴스에 대해 "frame" 마커(인덱스 0)로 표기
        marker_indices = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        marker.visualize(pos, quat, marker_indices=marker_indices)

        if count % 50 == 0:
            print(f"[INFO] root pos: {cube_object.data.root_pos_w[:, :3]}")

# ----------------------------------------------------------------
# 5) main() 실행부
# ----------------------------------------------------------------
def main():
    # 시뮬레이션 컨텍스트 생성
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # 카메라 시점 잡기 (원하는 위치로 수정)
    sim.set_camera_view(eye=[3.0, 0.0, 1.0], target=[0.0, 0.0, 0.5])

    # 장면 생성
    scene_entities, scene_origins = design_scene()

    # VisualizationMarkers 준비 ("frame"만 정의)
    my_marker = define_frame_marker()

    # 시뮬레이션 준비
    sim.reset()

    print("[INFO]: Setup complete. Running simulation...")

    # 시뮬레이션 루프
    run_simulator(sim, scene_entities, scene_origins.to(sim.device), my_marker)

    # 종료
    simulation_app.close()

if __name__ == "__main__":
    main()
    