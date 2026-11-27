"""Adapter between SAPIEN state and the released MA-Bench task evaluators."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
from scipy.spatial.transform import Rotation

from ma_bench.envs.base_env import MultiRobotEnv, MultiRobotObservation
from ma_bench.evaluators import StepEvidence, make_evaluator
from ma_task_harness.gates.physical import PhysicalGate
from ma_task_harness.runtime.sapien_world import SapienMultiRobotWorld, look_at_pose
from ma_task_harness.runtime.state_types import ExecutionMode
from ma_task_harness.spec.schema import examples_dir, load_task_spec

from .scene_tools import Motion, as_rotation, wxyz


def build_layout(world, task):
    if task in ("joint_bowl_placing", "hold_and_scan"):
        from .relay_scenes import build_bowl, build_scan

        return (build_bowl if task == "joint_bowl_placing" else build_scan)(world)
    if task in ("lift_tray_keep_blocks", "lift_tray_pour_food"):
        from .tray_scenes import build_tray

        return build_tray(world, pour=task == "lift_tray_pour_food")
    if task == "open_door_place_pot":
        from .door_scene import build_door

        return build_door(world)
    if task == "pour_and_catch":
        from .catch_scene import build_catch

        return build_catch(world)
    raise ValueError(f"No SAPIEN layout for {task}")


class _ScoredWorld(SapienMultiRobotWorld):
    on_control = None

    def step_control(self, metadata=None):
        frame = super().step_control(metadata)
        if self.on_control is not None:
            self.on_control(frame)
        return frame


class LiveSceneEnv(MultiRobotEnv):
    endpose_frame = "shared PhysX world, quaternion wxyz"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.task_info.slug == "joint_bowl_placing":
            self.endpose_frame = "per-robot tabletop frame, quaternion wxyz"
        self.spec = load_task_spec(examples_dir() / self.task_info.taskspec_filename)
        self.max_steps = round(self.task_info.nominal_duration_s * self.task_info.control_hz)
        self.world = None
        self.layout = None
        self.on_control = None
        self.camera_mounts = {}
        self.motion = None
        self._last_bowl_command = None

    def initialize(self, seed: int) -> None:
        import mplib

        self.close()
        self.step_count = 0
        self.time_s = 0.
        self.evaluator = make_evaluator(self.task_info.slug, seed=seed)
        mplib.set_global_seed(1 + seed % ((1 << 32) - 1))
        height, width = self.image_hw
        self.world = _ScoredWorld(self.spec, ExecutionMode(mode="team", seed=seed), robotwin_root=self.robotwin_root,
                                  render_width=width, render_height=height, enable_render=False)
        self.layout = build_layout(self.world, self.task_info.slug)
        if self.task_info.slug == "joint_bowl_placing":
            low, high = self.world.regions["goal_zone"]
            self.evaluator = make_evaluator(self.task_info.slug, seed=seed, goal_bounds_xy=(low[:2], high[:2]))
        self._add_cameras()
        self.world.settle()
        self.world.on_control = self._score
        self.motion = Motion(self.world)
        self._reference_pose_wxyz = self.canonical_state()

    def reset(self, seed: int) -> MultiRobotObservation:
        self.initialize(seed)
        return self._observation()

    def _add_cameras(self) -> None:
        world = self.world
        position, target = self.layout.overview
        camera = world.scene.add_camera("overview", 960, 540, np.deg2rad(48), .035, 25.)
        camera.set_entity_pose(look_at_pose(position, target))
        world.cameras["overview"] = camera
        for robot in self.task_info.robots:
            agent = robot.role_key
            art = world.robots[agent]
            root = art.get_root_pose()
            front = as_rotation(root.q).apply([1, 0, 0])
            ahead = root.p + front * (.36 if robot.arms == 2 else .10)
            ahead[2] = 1.75 if robot.arms == 2 else root.p[2] + .85
            focus = root.p + front * .60
            focus[2] = .80
            world.add_camera(f"{agent}/head", ahead, focus, fov_deg=64)
            for view, arm in self._arms(robot):
                key = f"{agent}/{view}_wrist"
                if robot.arms == 2:
                    link = art.find_link_by_name(f"{view}_camera")
                    offset = look_at_pose((.018, -.022, .006), (.139, -.032, -.033))
                else:
                    link = art.find_link_by_name(world.arms[arm].move_group)
                    if robot.embodiment == "panda":
                        offset = look_at_pose((.03, -.035, .025), (0., 0., .14))
                    else:
                        offset = look_at_pose((.040, -.035, .035), (.17, 0., 0.))
                world.add_camera(key, ahead, focus, fov_deg=62)
                self.camera_mounts[key] = (link, offset)

    @staticmethod
    def _arms(robot):
        agent = robot.role_key
        if robot.arms == 2:
            return (("left", f"{agent}.left"), ("right", f"{agent}.right"))
        return (("left", f"{agent}.main"),)

    def canonical_state(self):
        state = np.zeros((self.num_agents, 16), dtype=np.float64)
        for i, robot in enumerate(self.task_info.robots):
            transform = self.agent_to_world(robot.role_key)
            rotation = Rotation.from_matrix(transform[:3, :3]).inv()
            for j, (_, key) in enumerate(self._arms(robot)):
                pos, quat = self.world.get_arm_pose(key)
                arm = self.world.arms[key]
                qpos = arm.articulation.get_qpos()[arm.finger_indices]
                low = np.asarray(arm.gripper_closed_q)
                high = np.asarray(arm.gripper_open_q)
                opening = np.clip((qpos - low) / (high - low), 0., 1.).mean()
                state[i, 8 * j:8 * (j + 1)] = [
                    *rotation.apply(pos - transform[:3, 3]), *wxyz(rotation * as_rotation(quat)), opening,
                ]
        return state

    def agent_to_world(self, agent: str) -> np.ndarray:
        transform = np.eye(4)
        if self.task_info.slug == "joint_bowl_placing":
            root = self.world.robots[agent].get_root_pose()
            transform[:3, :3] = (as_rotation(root.q) * Rotation.from_euler("Z", -90, degrees=True)).as_matrix()
            transform[:3, 3] = (root.p[0], 0., root.p[2])
        return transform

    def commanded_state(self) -> np.ndarray:
        """Forward kinematics of the joint drive targets applied at this tick."""
        state = np.zeros((self.num_agents, 16), dtype=np.float64)
        for i, robot in enumerate(self.task_info.robots):
            arms = [self.world.arms[key] for _, key in self._arms(robot)]
            articulation = self.world.robots[robot.role_key]
            root = articulation.get_root_pose()
            rotation = as_rotation(root.q)
            transform = self.agent_to_world(robot.role_key)
            world_to_agent = Rotation.from_matrix(transform[:3, :3]).inv()
            qpos = articulation.get_qpos().astype(np.float64)
            for arm in arms:
                qpos[arm.arm_indices] = arm.target_q
                closed, opened = map(np.asarray, (arm.gripper_closed_q, arm.gripper_open_q))
                qpos[arm.finger_indices] = closed + arm.gripper_open * (opened - closed)
            planner = arms[0].planner
            kinematics = planner.pinocchio_model
            kinematics.compute_forward_kinematics(qpos)
            for j, arm in enumerate(arms):
                pose = kinematics.get_link_pose(planner.link_name_2_idx[arm.move_group])
                position = root.p + rotation.apply(pose.p)
                state[i, 8*j:8*(j+1)] = [
                    *world_to_agent.apply(position - transform[:3, 3]),
                    *wxyz(world_to_agent * rotation * as_rotation(pose.q)), arm.gripper_open,
                ]
        return state

    def render(self, names):
        names = tuple(names)
        for name in names:
            if name in self.camera_mounts:
                link, offset = self.camera_mounts[name]
                self.world.cameras[name].set_entity_pose(link.get_entity_pose() * offset)
        self.world.scene.update_render()
        for name in names:
            self.world.cameras[name].take_picture()
        images = {}
        for name in names:
            camera = self.world.cameras[name]
            images[name] = (255 * np.clip(camera.get_picture("Color")[..., :3], 0, 1)).astype(np.uint8)
        return images

    def _observation(self):
        views = ("head", "left_wrist", "right_wrist")
        images = self.render(f"{robot.role_key}/{view}" for robot in self.task_info.robots for view in views
                             if f"{robot.role_key}/{view}" in self.world.cameras)
        zeros = np.zeros((*self.image_hw, 3), dtype=np.uint8)
        grouped = {key: np.stack([images.get(f"{r.role_key}/{view}", zeros) for r in self.task_info.robots])
                   for key, view in zip(self.CAMERA_KEYS, views)}
        return MultiRobotObservation(self.step_count, self.time_s, grouped, self.canonical_state())

    def advance_control(self, absolute_ee_targets_wxyz) -> bool:
        if self.world is None:
            raise RuntimeError("Call reset(seed) before step_control")
        actions = np.asarray(absolute_ee_targets_wxyz, dtype=np.float64)
        if actions.shape != (self.num_agents, 16) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite [{self.num_agents}, 16] end-effector targets")
        bowl = self.task_info.slug == "joint_bowl_placing"
        targets, grippers = {}, {}
        for i, robot in enumerate(self.task_info.robots):
            transform = self.agent_to_world(robot.role_key)
            rotation = Rotation.from_matrix(transform[:3, :3])
            for j, (_, key) in enumerate(self._arms(robot)):
                xyz, quat, opening = actions[i, 8*j:8*j+3], actions[i, 8*j+3:8*j+7], actions[i, 8*j+7]
                norm = np.linalg.norm(quat)
                if norm < 1e-8:
                    raise ValueError(f"Invalid zero quaternion for {key}")
                pose = slice(8*j, 8*j+7)
                same_pose = (
                    bowl and self._last_bowl_command is not None
                    and np.array_equal(actions[i, pose], self._last_bowl_command[i, pose])
                )
                if not same_pose:
                    targets[key] = (rotation.apply(xyz) + transform[:3, 3], wxyz(rotation * as_rotation(quat / norm)))
                if not bowl or np.clip(opening, 0., 1.) != self.world.arms[key].gripper_open:
                    grippers[key] = opening
        if not self.world.move_arms_to(targets, steps=1, grippers=grippers, warm_start_ik=bowl):
            raise RuntimeError("SAPIEN/mplib could not solve the requested control step")
        if bowl:
            self._last_bowl_command = actions.copy()
        return self.step_count >= self.max_steps

    def step_control(self, absolute_ee_targets_wxyz):
        done = self.advance_control(absolute_ee_targets_wxyz)
        return self._observation(), done

    def _score(self, frame) -> None:
        self.step_count = self.world.step_count
        self.time_s = self.step_count * self.control_dt
        objects = {name: asdict(body) for name, body in frame.bodies.items()}
        for container in self.layout.containers:
            center, quat = self.world.get_body_pose(container.name)
            orientation = as_rotation(quat)
            for name in container.member_names:
                if container.contains(objects[name]["pos"], center, orientation):
                    objects[name]["container"] = container.name
        if self.layout.annotate:
            self.layout.annotate(objects)
        contacts = tuple((c.entity_a, c.entity_b) for c in frame.contacts)
        self.evaluator.update(StepEvidence(
            time_s=self.time_s, dt_s=self.control_dt, objects=objects,
            grippers={key: arm.gripper_open for key, arm in frame.arms.items()}, contacts=contacts,
            hand_positions={key: arm.ee_pos for key, arm in frame.arms.items()},
            robot_collision=any({a.split(".")[0], b.split(".")[0]} == {"A", "B"} for a, b in contacts),
            metadata=frame.metadata,
        ))
        if self.on_control is not None:
            self.on_control(frame)

    def run_expert(self) -> None:
        self.motion.hold(5)
        self.layout.expert(self.motion)
        self.motion.hold(30)

    def acceptance(self, error=None) -> dict:
        trace = self.world.build_trace()
        physical = PhysicalGate(self.spec).evaluate(trace).to_dict()
        if error:
            physical["planner_feasible"] = False
            physical["passed"] = False
            physical["reasons"].insert(0, error)
        metrics = self.evaluator.report(execution_valid=physical["passed"] and bool(trace.frames))
        dynamic = [key for key, b in trace.initial_bodies.items() if not b.is_fixture]
        samples = {key: {
            "initial_pos": trace.initial_bodies[key].pos,
            "final_pos": trace.frames[-1].bodies[key].pos,
            "max_z_m": max(f.bodies[key].pos[2] for f in trace.frames),
            "final_regions": trace.frames[-1].bodies[key].in_regions,
        } for key in dynamic if trace.frames}
        joints = {key: {"final_angle_deg": trace.frames[-1].bodies[key].joint_deg,
                        "max_angle_deg": max(frame.bodies[key].joint_deg for frame in trace.frames)}
                  for key, record in self.world.bodies.items() if record["kind"] == "hinge" and trace.frames}
        result = {"simulator_engine": "sapien_physx", "task_spec_sha256": self.spec.sha256(),
                  "metrics": metrics.to_dict(), "physical": physical, "accepted": metrics.joint_success,
                  "object_motion": samples, "fixture_joint_motion": joints, "phases": self.motion.phases}
        if self.task_info.slug == "joint_bowl_placing" and "bowl" in trace.initial_bodies:
            arm_order = ("A.left", "A.right", "B.right", "B.left")
            lifted_z = trace.initial_bodies["bowl"].pos[2] + .025
            lifts = {}
            for frame in trace.frames:
                if frame.bodies["bowl"].pos[2] < lifted_z:
                    continue
                for contact in frame.contacts:
                    a, b = contact.entity_a, contact.entity_b
                    other = b if a == "bowl" else a if b == "bowl" else ""
                    for arm in arm_order:
                        if other.startswith(arm + "."):
                            lifts.setdefault(arm, frame.step)
            ordered = len(lifts) == len(arm_order) and all(lifts[a] < lifts[b] for a, b in zip(arm_order, arm_order[1:]))
            result.update(four_arm_lift_ticks=lifts, four_arm_relay=ordered, accepted=metrics.joint_success and ordered)
        return result

    def close(self):
        self.world = self.layout = self.motion = None
        self.camera_mounts.clear()
        self._reference_pose_wxyz = None
        self._last_bowl_command = None
