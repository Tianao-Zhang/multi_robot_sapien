"""Live SAPIEN implementation of the public Find common color TaskSpec."""

from __future__ import annotations

import numpy as np

from ma_bench.envs.base_env import MultiRobotEnv, MultiRobotObservation
from ma_bench.evaluators import StepEvidence, make_evaluator
from ma_task_harness.gates.physical import PhysicalGate
from ma_task_harness.runtime.sapien_world import SapienMultiRobotWorld
from ma_task_harness.runtime.state_types import ExecutionMode
from ma_task_harness.spec.schema import examples_dir, load_task_spec

from .color_expert import execute_expert
from .color_scene import AGENTS, VIEWS, build_scene, update_wrist_cameras


class _ObservedWorld(SapienMultiRobotWorld):
    def step_control(self, metadata=None):
        frame = super().step_control(metadata, physics_observer=self.observe_physics)
        if self.on_control is not None:
            self.on_control(frame)
        return frame


class FindCommonColorEnv(MultiRobotEnv):
    endpose_frame = "agent tabletop frame, quaternion wxyz"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec = load_task_spec(examples_dir() / "find_common_color.yaml")
        self.world = None
        self.layout = None
        self.on_control = None
        self.max_steps = round(self.task_info.nominal_duration_s * self.task_info.control_hz)

    def reset(self, seed: int) -> MultiRobotObservation:
        import mplib

        self.close()
        self.step_count = 0
        self.time_s = 0.0
        self.evaluator = make_evaluator(self.task_info.slug, seed=seed)
        mplib.set_global_seed(1 + seed % ((1 << 32) - 1))
        height, width = self.image_hw
        self.world = _ObservedWorld(
            self.spec, ExecutionMode(mode="team", seed=seed), robotwin_root=self.robotwin_root,
            render_width=width, render_height=height, enable_render=False,
        )
        self.world.observe_physics = self._observe_physics
        self.world.on_control = self._after_control
        self.layout = build_scene(self.world)
        self.world.settle()
        self._observe_physics(self.world._read_contacts())
        obs = self._observation()
        self._reference_pose_wxyz = obs.canonical_state.copy()
        return obs

    def agent_to_world(self, agent: str) -> np.ndarray:
        """Translate the task-local tabletop frame into the shared PhysX world."""
        transform = np.eye(4)
        sign = -1 if agent == "A" else 1
        transform[0, 3] = sign * float(self.spec.scene_params["separation_m"]) / 2
        return transform

    def _observe_physics(self, contacts) -> None:
        world = self.world
        objects = {}
        for agent, colors in self.layout.block_colors.items():
            region = f"pad_{agent}"
            low, high = world.regions[region]
            for name in colors:
                pos, _ = world.get_body_pose(name)
                objects[name] = {
                    "pos": pos,
                    "side": agent,
                    "in_regions": (region,) if np.all((low <= pos) & (pos <= high)) else (),
                    "pad_center_xy": self.layout.pad_centers[agent][:2],
                }
        self.evaluator.update(StepEvidence(
            time_s=self.evaluator.steps * world.physics_dt,
            dt_s=world.physics_dt,
            objects=objects,
            grippers={key: arm.gripper_open for key, arm in world.arms.items()},
            contacts=tuple((c.entity_a, c.entity_b) for c in contacts),
        ))

    def _after_control(self, frame) -> None:
        self.step_count = self.world.step_count
        self.time_s = self.step_count * self.control_dt
        if self.on_control is not None:
            self.on_control(frame)

    def canonical_state(self) -> np.ndarray:
        state = np.zeros((self.num_agents, 16), dtype=np.float64)
        for index, agent in enumerate(AGENTS):
            origin = self.agent_to_world(agent)[:3, 3]
            qpos = self.world.robots[agent].get_qpos()
            for offset, side in ((0, "left"), (8, "right")):
                key = f"{agent}.{side}"
                arm = self.world.arms[key]
                pos, quat = self.world.get_arm_pose(key)
                closed, opened = map(np.asarray, (arm.gripper_closed_q, arm.gripper_open_q))
                opening = np.clip((qpos[arm.finger_indices] - closed) / (opened - closed), 0, 1).mean()
                state[index, offset:offset + 8] = [*(pos - origin), *quat, opening]
        return state

    def render(self, names) -> dict[str, np.ndarray]:
        update_wrist_cameras(self.world, self.layout)
        self.world.scene.update_render()
        frames = {}
        for name in names:
            camera = self.world.cameras[name]
            camera.take_picture()
            rgb = camera.get_picture("Color")[..., :3]
            frames[name] = (255 * np.clip(rgb, 0, 1)).astype(np.uint8)
        return frames

    def _observation(self) -> MultiRobotObservation:
        views = self.render(f"{agent}/{view}" for agent in AGENTS for view in VIEWS)
        images = {
            key: np.stack([views[f"{agent}/{view}"] for agent in AGENTS])
            for key, view in zip(self.CAMERA_KEYS, VIEWS)
        }
        return MultiRobotObservation(self.step_count, self.time_s, images, self.canonical_state())

    def step_control(self, absolute_ee_targets_wxyz: np.ndarray) -> tuple[MultiRobotObservation, bool]:
        if self.world is None:
            raise RuntimeError("Call reset(seed) before step_control")
        action = np.asarray(absolute_ee_targets_wxyz, dtype=np.float64)
        if action.shape != (self.num_agents, 16) or not np.isfinite(action).all():
            raise ValueError(f"Expected finite [{self.num_agents}, 16] end-effector targets")
        targets, grippers = {}, {}
        for index, agent in enumerate(AGENTS):
            origin = self.agent_to_world(agent)[:3, 3]
            for offset, side in ((0, "left"), (8, "right")):
                key = f"{agent}.{side}"
                pos = action[index, offset:offset + 3] + origin
                quat = action[index, offset + 3:offset + 7]
                norm = np.linalg.norm(quat)
                if norm < 1e-8:
                    raise ValueError(f"Invalid zero quaternion for {key}")
                targets[key] = (pos, quat / norm)
                grippers[key] = action[index, offset + 7]
        if not self.world.move_arms_to(targets, steps=1, grippers=grippers):
            raise RuntimeError("SAPIEN/mplib could not solve the requested control step")
        return self._observation(), self.step_count >= self.max_steps

    def run_expert(self) -> None:
        execute_expert(self.world, self.layout)
        remaining = self.max_steps - self.step_count
        if remaining > 0:
            self.world.hold_steps(remaining)
        elif remaining < 0:
            raise RuntimeError(f"Expert exceeded the {self.max_steps}-tick task horizon")

    def acceptance(self) -> dict:
        trace = self.world.build_trace()
        physical = PhysicalGate(self.spec).evaluate(trace)
        tail_size = round(self.spec.thresholds["terminal_rest_s"] * self.spec.control_hz)
        tail = trace.frames[-tail_size:]
        supported = {}
        opposed_lift = {}
        for agent, target in zip(AGENTS, (self.evaluator.target_a, self.evaluator.target_b)):
            supported[agent] = len(tail) == tail_size and all(
                f"pad_{agent}" in frame.bodies[target].in_regions
                and "table" in frame.bodies[target].supported_by
                and np.linalg.norm(np.asarray(frame.bodies[target].pos[:2]) - self.layout.pad_centers[agent][:2])
                <= self.evaluator.place_xy_tol_m
                and not any(target in (contact.entity_a, contact.entity_b) and contact.whitelisted_grasp
                            for contact in frame.contacts)
                for frame in tail
            )
            initial_z = trace.initial_bodies[target].pos[2]
            opposed_lift[agent] = any(
                frame.bodies[target].pos[2] - initial_z >= self.evaluator.lift_min_m
                and any(c.whitelisted_grasp and c.two_finger_opposed and target in (c.entity_a, c.entity_b)
                        for c in frame.contacts)
                for frame in trace.frames
            )
        valid = physical.passed and all(supported.values()) and all(opposed_lift.values())
        metrics = self.evaluator.report(execution_valid=valid)
        return {
            "simulator_engine": trace.metadata["simulator_engine"],
            "metrics": metrics.to_dict(),
            "physical": physical.to_dict(),
            "target_supported_on_pad_last_second": supported,
            "opposed_contact_during_lift": opposed_lift,
            "accepted": metrics.joint_success,
        }

    def close(self) -> None:
        self.world = None
        self.layout = None
        self._reference_pose_wxyz = None
