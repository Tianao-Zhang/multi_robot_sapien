"""Task-agnostic RoboTwin 2.0 / SAPIEN 3.0 + PhysX + mplib multi-robot world API.

Provides a clean, open-source multi-robot simulation layer shared by MA-Task Harness
(Generator G^(n) writes scenes/experts against this API) and MA-Bench:
  - Loads arbitrary N >= 2 RoboTwin embodiments (`panda`, `ur5`, `aloha`) from
    `$ROBOTWIN_ROOT/assets/embodiments/` with real PhysX PD joint drives & grippers
  - Instantiates real `mplib.Planner` IK & motion planning per arm
  - Supports dynamic rigid bodies, static fixtures, spring-return revolute fixtures,
    and named tabletop regions
  - Executes real 250 Hz PhysX physics (10 substeps per 25 Hz control step), reads
    real `scene.get_contacts()`, renders real GPU RGB camera views (base64 JPEG
    keyframes for the dialogue trace viewer), and exports standard `EpisodeTrace`
    stamped with `simulator_engine = "sapien_physx"`.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import io
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ma_task_harness.runtime.state_types import (
    ArmState,
    BodyState,
    ContactEvent,
    ControlFrame,
    EpisodeTrace,
    ExecutionMode,
)
from ma_task_harness.spec.schema import TaskSpec


EMBODIMENT_ASSET_SPECS: Dict[str, Dict[str, Any]] = {
    "panda": {
        "dir": "franka-panda",
        "urdf": "panda.urdf",
        "srdf": "panda.srdf",
        "arms": [
            {
                "name": "main",
                "move_group": "panda_hand",
                "arm_joints": [f"panda_joint{i}" for i in range(1, 8)],
                "finger_joints": ["panda_finger_joint1", "panda_finger_joint2"],
                "finger_links": ("panda_leftfinger", "panda_rightfinger"),
                "gripper_open_q": [0.04, 0.04],
                "gripper_closed_q": [0.0, 0.0],
                "home_q": [0.0, 0.196, 0.0, -2.20, 0.0, 2.40, 0.785],
                "topdown_quat": [0.0, 1.0, 0.0, 0.0],
                "tcp_z_offset": 0.103,
            }
        ],
    },
    "ur5": {
        "dir": "ur5-wsg",
        "urdf": "ur5_wsg_gripper.urdf",
        "srdf": "ur5.srdf",
        "arms": [
            {
                "name": "main",
                "move_group": "ee_link",
                "arm_joints": [
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ],
                "finger_joints": ["base_joint_gripper_left", "base_joint_gripper_right"],
                "finger_links": ("finger_left", "finger_right", "gripper_left", "gripper_right"),
                "gripper_open_q": [-0.055, 0.055],
                "gripper_closed_q": [-0.0027, 0.0027],
                "home_q": [0.0, -1.35, 1.55, -1.75, -1.57, 0.0],
                "topdown_quat": [0.7071, 0.0, 0.7071, 0.0],
                "tcp_z_offset": 0.135,
            }
        ],
    },
    "aloha": {
        "dir": "aloha-agilex",
        "urdf": "urdf/arx5_description_isaac.urdf",
        "srdf": "srdf/arx5_description_isaac.srdf",
        "arms": [
            {
                "name": "left",
                "move_group": "fl_link6",
                "arm_joints": [f"fl_joint{i}" for i in range(1, 7)],
                "finger_joints": ["fl_joint7", "fl_joint8"],
                "finger_links": ("fl_link7", "fl_link8"),
                "gripper_open_q": [0.045, 0.045],
                "gripper_closed_q": [0.0, 0.0],
                "home_q": [0.0, 0.35, 0.70, 0.0, 0.50, 0.0],
                "topdown_quat": [0.7071, 0.0, 0.7071, 0.0],
                "tcp_z_offset": 0.120,
            },
            {
                "name": "right",
                "move_group": "fr_link6",
                "arm_joints": [f"fr_joint{i}" for i in range(1, 7)],
                "finger_joints": ["fr_joint7", "fr_joint8"],
                "finger_links": ("fr_link7", "fr_link8"),
                "gripper_open_q": [0.045, 0.045],
                "gripper_closed_q": [0.0, 0.0],
                "home_q": [0.0, 0.35, 0.70, 0.0, 0.50, 0.0],
                "topdown_quat": [0.7071, 0.0, 0.7071, 0.0],
                "tcp_z_offset": 0.120,
            },
        ],
    },
}


def look_at_pose(position: Sequence[float], target: Sequence[float]):
    """Construct a SAPIEN camera Pose looking from `position` toward `target` (+X forward, +Y left, +Z up)."""
    import sapien

    pos = np.asarray(position, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    fwd = tgt - pos
    fwd /= max(1e-9, float(np.linalg.norm(fwd)))
    up_ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(fwd, up_ref))) > 0.99:
        up_ref = np.array([0.0, 1.0, 0.0])
    left = np.cross(up_ref, fwd)
    left /= max(1e-9, float(np.linalg.norm(left)))
    up = np.cross(fwd, left)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = np.column_stack([fwd, left, up])
    mat[:3, 3] = pos
    return sapien.Pose(mat)


def yaw_quat_wxyz(yaw_deg: float) -> Tuple[float, float, float, float]:
    rad = math.radians(yaw_deg) * 0.5
    return (math.cos(rad), 0.0, 0.0, math.sin(rad))


def _encode_rgb_jpeg_b64(rgb_uint8: np.ndarray, quality: int = 80) -> str:
    """Encode an HxWx3 uint8 RGB array into a compact base64 JPEG data URL."""
    try:
        import cv2

        bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        pass
    try:
        from PIL import Image

        bio = io.BytesIO()
        Image.fromarray(rgb_uint8).save(bio, format="JPEG", quality=quality)
        return "data:image/jpeg;base64," + base64.b64encode(bio.getvalue()).decode("ascii")
    except Exception:
        return ""


@dataclass
class _ArmHandle:
    arm_key: str        # e.g. "A.main", "B.left"
    agent_id: str
    arm_name: str
    robot_type: str
    articulation: Any
    arm_indices: List[int]
    finger_indices: List[int]
    finger_links: Tuple[str, ...]
    move_group: str
    gripper_open_q: List[float]
    gripper_closed_q: List[float]
    planner: Any
    target_q: np.ndarray
    gripper_open: float
    topdown_quat: Tuple[float, float, float, float]
    tcp_z_offset: float
    last_ik_valid: bool = True


class SapienMultiRobotWorld:
    """Real SAPIEN 3.0 + PhysX (250 Hz) + mplib multi-robot environment builder."""

    def __init__(
        self,
        spec: TaskSpec,
        mode: ExecutionMode,
        robotwin_root: Optional[Path] = None,
        render_width: int = 480,
        render_height: int = 360,
        capture_every_n_steps: int = 12,
        enable_render: bool = True,
    ) -> None:
        import sapien

        self.spec = spec
        self.mode = mode
        self.robotwin_root = Path(
            robotwin_root or os.environ.get("ROBOTWIN_ROOT", "./third_party/RoboTwin")
        ).resolve()
        if not (self.robotwin_root / "assets" / "embodiments").is_dir():
            raise FileNotFoundError(
                f"RoboTwin embodiments not found under {self.robotwin_root / 'assets' / 'embodiments'}. "
                "Set ROBOTWIN_ROOT to a RoboTwin 2.0 checkout."
            )

        self.control_hz = float(spec.control_hz)
        self.physics_hz = float(spec.physics_hz)
        self.control_dt = 1.0 / self.control_hz
        self.substeps = max(1, int(round(self.physics_hz / self.control_hz)))
        self.physics_dt = self.control_dt / self.substeps
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.capture_every_n_steps = int(capture_every_n_steps)
        self.enable_render = bool(enable_render) and os.environ.get("MAWAM_HARNESS_DISABLE_RENDER") != "1"

        sapien.render.set_camera_shader_dir("default")
        render_device = os.environ.get("MAWAM_RENDER_DEVICE")
        self.scene = sapien.Scene([
            sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(render_device),
        ]) if render_device else sapien.Scene()
        self.scene.set_timestep(self.physics_dt)
        self.scene.default_physical_material = self.scene.create_physical_material(1.0, 0.9, 0.0)
        self.scene.add_ground(0.0)
        self.scene.set_ambient_light([0.55, 0.55, 0.55])
        self.scene.add_directional_light([0.35, 0.45, -1.0], [1.6, 1.6, 1.6], shadow=True)
        self.scene.add_point_light([1.0, -0.8, 2.0], [0.9, 0.9, 0.9], shadow=False)
        self.scene.add_point_light([-1.0, 0.8, 2.0], [0.9, 0.9, 0.9], shadow=False)

        self.robots: Dict[str, Any] = {}
        self.arms: Dict[str, _ArmHandle] = {}
        self.bodies: Dict[str, Dict[str, Any]] = {}  # name -> {actor, is_fixture, joint, ...}
        self.regions: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {}
        self.cameras: Dict[str, Any] = {}
        self._entity_tags: Dict[int, str] = {}
        self._finger_tags: Dict[int, Tuple[str, str]] = {}  # entity_id -> (arm_key, link_name)
        self._whitelisted_grasps: set[Tuple[str, str]] = set()  # (arm_key, body_name)

        self.step_count = 0
        self.frames: List[ControlFrame] = []
        self.initial_bodies: Dict[str, BodyState] = {}
        self.rendered_frames: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Scene building helpers (real PhysX actors, joints, robots, cameras)
    # ------------------------------------------------------------------

    def add_table(
        self,
        center: Sequence[float] = (0.0, 0.0, 0.74),
        size: Sequence[float] = (1.60, 1.10, 0.04),
        color: Sequence[float] = (0.86, 0.88, 0.90),
    ):
        import sapien

        cx, cy, top_z = float(center[0]), float(center[1]), float(center[2])
        sx, sy, thick = float(size[0]), float(size[1]), float(size[2])
        b = self.scene.create_actor_builder()
        b.add_box_collision(half_size=[sx / 2, sy / 2, thick / 2])
        b.add_box_visual(half_size=[sx / 2, sy / 2, thick / 2], material=sapien.render.RenderMaterial(base_color=[*color, 1.0]))
        actor = b.build_static(name="table")
        actor.set_pose(sapien.Pose([cx, cy, top_z - thick / 2]))
        self._entity_tags[actor.per_scene_id] = "table"
        return actor

    def add_region(
        self,
        name: str,
        bounds_min: Sequence[float],
        bounds_max: Sequence[float],
        visual_color: Optional[Sequence[float]] = None,
    ) -> None:
        import sapien

        lo = (float(bounds_min[0]), float(bounds_min[1]), float(bounds_min[2]))
        hi = (float(bounds_max[0]), float(bounds_max[1]), float(bounds_max[2]))
        self.regions[name] = (lo, hi)
        if visual_color is not None:
            b = self.scene.create_actor_builder()
            hx = max(0.01, (hi[0] - lo[0]) / 2.0)
            hy = max(0.01, (hi[1] - lo[1]) / 2.0)
            b.add_box_visual(
                half_size=[hx, hy, 0.002],
                material=sapien.render.RenderMaterial(base_color=[*visual_color, 1.0]),
            )
            pad = b.build_static(name=f"pad_{name}")
            pad.set_pose(sapien.Pose([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] + 0.002]))

    def add_box(
        self,
        name: str,
        pos: Sequence[float],
        half_size: Sequence[float],
        color: Sequence[float] = (0.85, 0.25, 0.20),
        mass: float = 0.12,
        is_static: bool = False,
        is_fixture: bool = False,
        friction: float = 1.4,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        import sapien

        mat = self.scene.create_physical_material(friction, friction * 0.9, 0.0)
        b = self.scene.create_actor_builder()
        b.add_box_collision(half_size=list(half_size), material=mat, density=max(50.0, mass / max(1e-6, 8 * half_size[0] * half_size[1] * half_size[2])))
        b.add_box_visual(
            half_size=list(half_size),
            material=sapien.render.RenderMaterial(base_color=[*color, 1.0]),
        )
        actor = b.build_static(name=name) if is_static else b.build(name=name)
        actor.set_pose(sapien.Pose(list(pos)))
        self._entity_tags[actor.per_scene_id] = name
        self.bodies[name] = {
            "kind": "rigid",
            "actor": actor,
            "is_fixture": bool(is_fixture or is_static),
            "attributes": dict(attributes or {}),
        }
        return actor

    def add_spring_hinge_fixture(
        self,
        name: str,
        hinge_pos: Sequence[float],
        leaf_offset: Sequence[float],
        leaf_half_size: Sequence[float],
        handle_offset: Optional[Sequence[float]] = None,
        handle_half_size: Sequence[float] = (0.015, 0.05, 0.015),
        limits_deg: Tuple[float, float] = (0.0, 95.0),
        stiffness: float = 0.45,
        damping: float = 0.55,
        axis: str = "z",  # "z" (door/gate) or "y" (lever)
        color: Sequence[float] = (0.25, 0.55, 0.78),
        handle_color: Sequence[float] = (0.95, 0.72, 0.18),
    ):
        """Build a real PhysX articulated revolute fixture with spring-return drive to 0 deg."""
        import sapien
        import transforms3d as t3d

        builder = self.scene.create_articulation_builder()
        root = builder.create_link_builder()
        root.set_name(f"{name}_root")
        leaf = builder.create_link_builder(root)
        leaf.set_name(name)
        leaf.set_joint_name(f"{name}_joint")
        # SAPIEN revolute joint rotates around local X; orient joint frame so local X matches requested axis
        if axis == "z":
            j_quat = t3d.euler.euler2quat(0.0, -math.pi / 2.0, 0.0)
        elif axis == "y":
            j_quat = t3d.euler.euler2quat(0.0, 0.0, math.pi / 2.0)
        else:
            j_quat = (1.0, 0.0, 0.0, 0.0)
        leaf.set_joint_properties(
            "revolute",
            [[math.radians(limits_deg[0]), math.radians(limits_deg[1])]],
            sapien.Pose(list(hinge_pos), j_quat),
            sapien.Pose([0.0, 0.0, 0.0], j_quat),
            friction=0.01,
            damping=0.05,
        )
        leaf.add_box_collision(pose=sapien.Pose(list(leaf_offset)), half_size=list(leaf_half_size), density=250.0)
        leaf.add_box_visual(
            pose=sapien.Pose(list(leaf_offset)),
            half_size=list(leaf_half_size),
            material=sapien.render.RenderMaterial(base_color=[*color, 1.0]),
        )
        if handle_offset is not None:
            leaf.add_box_collision(pose=sapien.Pose(list(handle_offset)), half_size=list(handle_half_size), density=300.0)
            leaf.add_box_visual(
                pose=sapien.Pose(list(handle_offset)),
                half_size=list(handle_half_size),
                material=sapien.render.RenderMaterial(base_color=[*handle_color, 1.0]),
            )
        art = builder.build(fix_root_link=True)
        art.set_qpos([0.0])
        joint = art.get_active_joints()[0]
        joint.set_drive_properties(stiffness=float(stiffness), damping=float(damping), force_limit=5.0)
        joint.set_drive_target(0.0)
        leaf_link = art.find_link_by_name(name)
        self._entity_tags[leaf_link.entity.per_scene_id] = name
        self.bodies[name] = {
            "kind": "hinge",
            "articulation": art,
            "link": leaf_link,
            "joint": joint,
            "is_fixture": True,
            "attributes": {"stiffness": stiffness, "damping": damping},
        }
        return art

    def add_robot(
        self,
        agent_id: str,
        robot_type: str,
        base_pos: Sequence[float],
        yaw_deg: float = 0.0,
    ) -> List[str]:
        """Load a real RoboTwin URDF robot + PhysX drives + mplib.Planner for every arm."""
        import mplib
        import sapien

        key = robot_type.strip().lower()
        if key not in EMBODIMENT_ASSET_SPECS:
            raise KeyError(f"Unsupported SAPIEN robot_type {robot_type!r}; supported: {sorted(EMBODIMENT_ASSET_SPECS)}")
        meta = EMBODIMENT_ASSET_SPECS[key]
        rdir = self.robotwin_root / "assets" / "embodiments" / meta["dir"]
        urdf = rdir / meta["urdf"]
        srdf = rdir / meta["srdf"]

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        art = loader.load(str(urdf))
        art.set_name(f"robot_{agent_id}_{key}")
        root_pose = sapien.Pose(list(base_pos), yaw_quat_wxyz(yaw_deg))
        art.set_root_pose(root_pose)

        active_joints = art.get_active_joints()
        jname_to_idx = {j.name: i for i, j in enumerate(active_joints)}
        qpos = np.zeros(art.dof, dtype=np.float64)

        # Configure stiff PD drives on every active joint
        for j in active_joints:
            j.set_drive_properties(stiffness=1200.0, damping=180.0, force_limit=200.0)

        for link in art.get_links():
            self._entity_tags[link.entity.per_scene_id] = f"{agent_id}.{link.name}"

        created_keys: List[str] = []
        for arm_spec in meta["arms"]:
            arm_key = f"{agent_id}.{arm_spec['name']}"
            arm_idxs = [jname_to_idx[jn] for jn in arm_spec["arm_joints"]]
            fin_idxs = [jname_to_idx[jn] for jn in arm_spec["finger_joints"]]
            home_q = np.asarray(arm_spec["home_q"], dtype=np.float64)
            qpos[arm_idxs] = home_q
            qpos[fin_idxs] = np.asarray(arm_spec["gripper_open_q"], dtype=np.float64)
            for idx in fin_idxs:
                active_joints[idx].set_drive_properties(stiffness=800.0, damping=120.0, force_limit=80.0)

            for fl_name in arm_spec["finger_links"]:
                lk = art.find_link_by_name(fl_name)
                if lk is not None:
                    self._finger_tags[lk.entity.per_scene_id] = (arm_key, fl_name)
                    self._entity_tags[lk.entity.per_scene_id] = f"{arm_key}.{fl_name}"

            # Initialize real mplib planner in robot-local frame (solve_ik maps world -> local once)
            planner = mplib.Planner(
                urdf=str(urdf),
                srdf=str(srdf),
                user_link_names=[l.name for l in art.get_links()],
                user_joint_names=[j.name for j in active_joints],
                move_group=arm_spec["move_group"],
            )
            # Disable static adjacent-link URDF overlaps already present at the home pose
            # (e.g., fixed wrist-to-gripper mounts or base wheels) while keeping all arm collisions active.
            acm = planner.planning_world.get_allowed_collision_matrix()
            for col in planner.check_for_self_collision(qpos):
                acm.set_entry(col.link_name1, col.link_name2, True)

            self.arms[arm_key] = _ArmHandle(
                arm_key=arm_key,
                agent_id=agent_id,
                arm_name=arm_spec["name"],
                robot_type=key,
                articulation=art,
                arm_indices=arm_idxs,
                finger_indices=fin_idxs,
                finger_links=tuple(arm_spec["finger_links"]),
                move_group=arm_spec["move_group"],
                gripper_open_q=list(arm_spec["gripper_open_q"]),
                gripper_closed_q=list(arm_spec["gripper_closed_q"]),
                planner=planner,
                target_q=home_q.copy(),
                gripper_open=1.0,
                topdown_quat=tuple(float(x) for x in arm_spec["topdown_quat"]),
                tcp_z_offset=float(arm_spec["tcp_z_offset"]),
            )
            created_keys.append(arm_key)

        art.set_qpos(qpos)
        art.set_qvel(np.zeros(art.dof))
        for idx, j in enumerate(active_joints):
            j.set_drive_target(float(qpos[idx]))
        self.robots[agent_id] = art
        return created_keys

    def add_camera(
        self,
        name: str,
        position: Sequence[float],
        target: Sequence[float],
        fov_deg: float = 54.0,
    ):
        cam = self.scene.add_camera(
            name, self.render_width, self.render_height, math.radians(fov_deg), 0.05, 25.0
        )
        cam.set_entity_pose(look_at_pose(position, target))
        self.cameras[name] = cam
        return cam

    def allow_grasp(self, arm_key: str, body_name: str, allowed: bool = True) -> None:
        """Whitelist intentional finger–object grasp contacts for physical collision auditing."""
        pair = (arm_key, body_name)
        if allowed:
            self._whitelisted_grasps.add(pair)
        else:
            self._whitelisted_grasps.discard(pair)

    # ------------------------------------------------------------------
    # Real mplib IK & smooth joint-space control over 250 Hz PhysX
    # ------------------------------------------------------------------

    def solve_ik(
        self,
        arm_key: str,
        target_pos: Sequence[float],
        target_quat_wxyz: Sequence[float] = (0.0, 1.0, 0.0, 0.0),
    ) -> Tuple[Optional[np.ndarray], bool]:
        """Solve real `mplib` IK for `arm_key` at world pose `(target_pos, target_quat_wxyz)`."""
        import mplib
        import sapien

        h = self.arms[arm_key]
        cur_full = h.articulation.get_qpos().copy()
        # Note: `planner.set_base_pose(root_pose)` is NOT called; instead we transform
        # the world target into the robot's local base frame explicitly once, matching
        # RoboTwin / mplib conventions cleanly across arbitrary base poses & yaws.
        root_inv = np.linalg.inv(h.articulation.get_root_pose().to_transformation_matrix())
        world_mat = sapien.Pose(list(target_pos), list(target_quat_wxyz)).to_transformation_matrix()
        local_pose = sapien.Pose(root_inv @ world_mat)
        goal = mplib.Pose(local_pose.p, local_pose.q)

        mask = np.ones(len(cur_full), dtype=bool)
        mask[h.arm_indices] = False
        status, sols = h.planner.IK(goal, cur_full, mask=mask, n_init_qpos=30, return_closest=False)
        if sols is None or len(sols) == 0:
            h.last_ik_valid = False
            return None, False
        sols = np.asarray(sols, dtype=np.float64)
        if sols.ndim == 1:
            sols = sols[None, :]
        qlim = h.articulation.get_qlimits()[h.arm_indices]
        cur_arm = cur_full[h.arm_indices]
        revolute = np.array([h.planner.joint_types[i].startswith("JointModelR") for i in h.arm_indices], dtype=bool)
        best_q = None
        best_dist = float("inf")
        for full_cand in sols:
            q = full_cand[h.arm_indices].copy()
            if not np.isfinite(q).all():
                continue
            # mplib puts revolute solutions near the lower limit. Choose the
            # equivalent angle nearest to the actual state before interpolating.
            angle = q[revolute]
            turn = 2.0 * np.pi
            limits = qlim[revolute]
            low = np.ceil((limits[:, 0] - angle - 1e-4) / turn)
            high = np.floor((limits[:, 1] - angle + 1e-4) / turn)
            q[revolute] += turn * np.clip(np.rint((cur_arm[revolute] - angle) / turn), low, high)
            if np.any(q < qlim[:, 0] - 1e-4) or np.any(q > qlim[:, 1] + 1e-4):
                continue
            d = float(np.linalg.norm(q - cur_arm))
            if d < best_dist:
                best_dist = d
                best_q = q.copy()
        if best_q is None:
            h.last_ik_valid = False
            return None, False
        h.last_ik_valid = True
        return best_q, True

    def set_gripper(self, arm_key: str, open_ratio: float) -> None:
        h = self.arms[arm_key]
        r = max(0.0, min(1.0, float(open_ratio)))
        h.gripper_open = r
        joints = h.articulation.get_active_joints()
        for fi, qo, qc in zip(h.finger_indices, h.gripper_open_q, h.gripper_closed_q):
            joints[fi].set_drive_target(qc + r * (qo - qc))

    def get_arm_pose(self, arm_key: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return current world `(pos_xyz, quat_wxyz)` of `arm_key`'s move_group link."""
        h = self.arms[arm_key]
        p = h.articulation.find_link_by_name(h.move_group).get_entity_pose()
        return np.asarray(p.p, dtype=np.float64), np.asarray(p.q, dtype=np.float64)

    def get_body_pose(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return current world `(pos_xyz, quat_wxyz)` of rigid or hinged body `name`."""
        rec = self.bodies[name]
        p = rec["link"].get_entity_pose() if rec["kind"] == "hinge" else rec["actor"].get_pose()
        return np.asarray(p.p, dtype=np.float64), np.asarray(p.q, dtype=np.float64)

    def get_hinge_angle_deg(self, name: str) -> float:
        """Return current revolute joint angle in degrees for a hinged fixture."""
        return math.degrees(float(self.bodies[name]["articulation"].get_qpos()[0]))

    def move_tcp_to(
        self,
        arm_key: str,
        tcp_pos: Sequence[float],
        quat: Optional[Sequence[float]] = None,
        steps: int = 25,
        gripper: Optional[float] = None,
    ) -> bool:
        """Move `arm_key`'s finger-tip TCP to `tcp_pos` using top-down orientation & URDF bias."""
        h = self.arms[arm_key]
        q = list(h.topdown_quat) if quat is None else list(quat)
        mg_pos = [float(tcp_pos[0]), float(tcp_pos[1]), float(tcp_pos[2]) + h.tcp_z_offset]
        return self.move_arms_to(
            {arm_key: (mg_pos, q)},
            steps=steps,
            grippers={arm_key: gripper} if gripper is not None else None,
        )

    def move_arms_to(
        self,
        targets: Dict[str, Tuple[Sequence[float], Optional[Sequence[float]]]],
        steps: int = 25,
        grippers: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Solve IK for all requested arms and smoothly execute via real 250 Hz PhysX steps.

        If `quat` in `(pos, quat)` is `None`, preserves the arm's current orientation.
        """
        q_starts: Dict[str, np.ndarray] = {}
        q_goals: Dict[str, np.ndarray] = {}
        for arm_key, (pos, quat) in targets.items():
            h = self.arms[arm_key]
            if quat is None:
                _, quat = self.get_arm_pose(arm_key)
            q_sol, ok = self.solve_ik(arm_key, pos, quat)
            if not ok or q_sol is None:
                return False
            q_starts[arm_key] = h.articulation.get_qpos()[h.arm_indices].copy()
            q_goals[arm_key] = q_sol
        if grippers:
            for ak, g in grippers.items():
                self.set_gripper(ak, g)
        n = max(1, int(steps))
        for k in range(1, n + 1):
            u = k / n
            alpha = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
            for arm_key, qg in q_goals.items():
                h = self.arms[arm_key]
                qt = q_starts[arm_key] + alpha * (qg - q_starts[arm_key])
                h.target_q = qt
                joints = h.articulation.get_active_joints()
                for idx, val in zip(h.arm_indices, qt):
                    joints[idx].set_drive_target(float(val))
            self.step_control()
        return True

    def hold_steps(self, steps: int = 10) -> None:
        for _ in range(max(1, int(steps))):
            self.step_control()

    def settle(self, steps: int = 15) -> None:
        """Let initial scene settle under PhysX before recording frame 0."""
        for _ in range(steps):
            for _ in range(self.substeps):
                self.scene.step()
        self.step_count = 0
        self.frames.clear()
        self.rendered_frames.clear()
        self.initial_bodies = self._snapshot_bodies(contacts=[])

    # ------------------------------------------------------------------
    # Stepping, real PhysX contact auditing, rendering, EpisodeTrace export
    # ------------------------------------------------------------------

    def _read_contacts(self) -> List[ContactEvent]:
        events: List[ContactEvent] = []
        seen_pairs: Dict[Tuple[str, str], float] = {}
        finger_hits: Dict[Tuple[str, str], set[str]] = {}  # (arm_key, body_name) -> {finger_link_names}

        for contact in self.scene.get_contacts():
            imp = sum(float(np.linalg.norm(pt.impulse)) for pt in contact.points)
            if imp <= 1e-6:
                continue
            force_n = imp / self.physics_dt
            b0, b1 = contact.bodies[0].entity, contact.bodies[1].entity
            id0, id1 = b0.per_scene_id, b1.per_scene_id
            n0 = self._entity_tags.get(id0, b0.name)
            n1 = self._entity_tags.get(id1, b1.name)
            if n0 == n1:
                continue
            # Ignore internal self-contacts within the same robot base/wheels unless arm-arm
            if "." in n0 and "." in n1 and n0.split(".")[0] == n1.split(".")[0]:
                continue
            if id0 in self._finger_tags and n1 in self.bodies:
                ak, fl = self._finger_tags[id0]
                finger_hits.setdefault((ak, n1), set()).add(fl)
            if id1 in self._finger_tags and n0 in self.bodies:
                ak, fl = self._finger_tags[id1]
                finger_hits.setdefault((ak, n0), set()).add(fl)
            key = (n0, n1) if n0 <= n1 else (n1, n0)
            seen_pairs[key] = max(seen_pairs.get(key, 0.0), force_n)

        for (e0, e1), force_n in seen_pairs.items():
            # Classify contacts by the contacting arm, including two arms of
            # the same articulation holding one object simultaneously.
            wl = False
            opposed = False
            for ak, bn in self._whitelisted_grasps:
                if (e0 == bn and e1.startswith(f"{ak}.")) or (e1 == bn and e0.startswith(f"{ak}.")):
                    wl = True
                    opposed = len(finger_hits.get((ak, bn), set())) >= 2
                    break
            events.append(
                ContactEvent(
                    entity_a=e0,
                    entity_b=e1,
                    normal_force_n=round(force_n, 3),
                    two_finger_opposed=opposed,
                    whitelisted_grasp=wl,
                )
            )
        return events

    def _snapshot_bodies(self, contacts: List[ContactEvent]) -> Dict[str, BodyState]:
        import sapien
        from scipy.spatial.transform import Rotation

        out: Dict[str, BodyState] = {}
        for name, rec in self.bodies.items():
            if rec["kind"] == "hinge":
                art = rec["articulation"]
                lk = rec["link"]
                pose = lk.get_entity_pose()
                deg = math.degrees(float(art.get_qpos()[0]))
                out[name] = BodyState(
                    name=name,
                    pos=(float(pose.p[0]), float(pose.p[1]), float(pose.p[2])),
                    joint_deg=deg,
                    is_fixture=True,
                    supported_by=("hinge",),
                    attributes=dict(rec.get("attributes", {})),
                )
            else:
                actor = rec["actor"]
                pose = actor.get_pose()
                p = (float(pose.p[0]), float(pose.p[1]), float(pose.p[2]))
                q_wxyz = pose.q
                rot = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
                z_axis = rot.as_matrix()[:, 2]
                tilt_deg = float(math.degrees(math.acos(max(-1.0, min(1.0, float(z_axis[2]))))))
                dyn = actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
                lin_v = tuple(float(x) for x in dyn.get_linear_velocity()) if dyn is not None else (0.0, 0.0, 0.0)
                ang_v = tuple(float(x) for x in dyn.get_angular_velocity()) if dyn is not None else (0.0, 0.0, 0.0)
                sups = tuple(
                    sorted(
                        {
                            c.entity_b if c.entity_a == name else c.entity_a
                            for c in contacts
                            if (c.entity_a == name or c.entity_b == name)
                            and not (c.entity_b if c.entity_a == name else c.entity_a).startswith(
                                tuple(f"{aid}." for aid in self.robots)
                            )
                        }
                    )
                )
                in_regs = tuple(
                    rname
                    for rname, (lo, hi) in self.regions.items()
                    if lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1] and lo[2] <= p[2] <= hi[2]
                )
                out[name] = BodyState(
                    name=name,
                    pos=p,
                    lin_vel=lin_v,
                    ang_vel=ang_v,
                    tilt_deg=tilt_deg,
                    supported_by=sups,
                    in_regions=in_regs,
                    is_fixture=bool(rec["is_fixture"]),
                    attributes=dict(rec.get("attributes", {})),
                )
        return out

    def _capture_render(self, t_s: float) -> None:
        if not self.enable_render or not self.cameras:
            return
        self.scene.update_render()
        views: Dict[str, str] = {}
        for cname, cam in self.cameras.items():
            cam.take_picture()
            rgba = cam.get_picture("Color")
            rgb = (np.clip(rgba[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)
            url = _encode_rgb_jpeg_b64(rgb, quality=78)
            if url:
                views[cname] = url
        if views:
            self.rendered_frames.append({"step": self.step_count, "t": round(t_s, 3), "views": views})

    def step_control(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        physics_observer: Optional[Callable[[List[ContactEvent]], None]] = None,
    ) -> ControlFrame:
        """Advance 1 control step (10 PhysX substeps @ 250 Hz), record real contacts & optional RGB."""
        all_contacts: List[ContactEvent] = []
        for _ in range(self.substeps):
            self.scene.step()
            if physics_observer is not None:
                all_contacts = self._read_contacts()
                physics_observer(all_contacts)
        if physics_observer is None:
            all_contacts = self._read_contacts()

        t_s = self.step_count * self.control_dt
        arm_states: Dict[str, ArmState] = {}
        for ak, h in self.arms.items():
            lk = h.articulation.find_link_by_name(h.move_group)
            pose = lk.get_pose()
            qfull = h.articulation.get_qpos()
            qlim = h.articulation.get_qlimits()[h.arm_indices]
            arm_states[ak] = ArmState(
                agent_id=h.agent_id,
                arm_name=h.arm_name,
                ee_pos=(float(pose.p[0]), float(pose.p[1]), float(pose.p[2])),
                ee_quat_wxyz=(float(pose.q[0]), float(pose.q[1]), float(pose.q[2]), float(pose.q[3])),
                gripper_open=float(h.gripper_open),
                qpos=tuple(float(qfull[i]) for i in h.arm_indices),
                joint_lower=tuple(float(x) for x in qlim[:, 0]),
                joint_upper=tuple(float(x) for x in qlim[:, 1]),
                ik_valid=bool(h.last_ik_valid),
            )

        body_states = self._snapshot_bodies(all_contacts)
        if not self.initial_bodies:
            self.initial_bodies = dict(body_states)

        cam_shapes = {cname: (3, 4) for cname in self.cameras}
        frame = ControlFrame(
            step=self.step_count,
            time_s=round(t_s, 6),
            arms=arm_states,
            bodies=body_states,
            contacts=all_contacts,
            camera_extrinsics_shapes=cam_shapes,
            metadata=dict(metadata or {}),
        )
        self.frames.append(frame)
        if self.enable_render and (self.step_count == 0 or self.step_count % self.capture_every_n_steps == 0):
            self._capture_render(t_s)
        self.step_count += 1
        return frame

    def build_trace(
        self,
        planner_feasible: bool = True,
        planner_failure_reason: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> EpisodeTrace:
        """Finalize and return an `EpisodeTrace` stamped with `simulator_engine='sapien_physx'`."""
        if self.enable_render and self.frames:
            # Ensure final frame is always rendered for the viewer
            if not self.rendered_frames or self.rendered_frames[-1]["step"] != self.frames[-1].step:
                self._capture_render(self.frames[-1].time_s)
        meta = {
            "simulator_engine": "sapien_physx",
            "physics_hz": self.physics_hz,
            "control_hz": self.control_hz,
            "rendered_frames": self.rendered_frames,
            **(extra_metadata or {}),
        }
        return EpisodeTrace(
            task=self.spec.task,
            seed=self.mode.seed,
            dt_s=self.control_dt,
            frames=self.frames,
            initial_bodies=self.initial_bodies,
            execution_mode=self.mode,
            planner_feasible=bool(planner_feasible),
            planner_failure_reason=str(planner_failure_reason),
            metadata=meta,
        )
