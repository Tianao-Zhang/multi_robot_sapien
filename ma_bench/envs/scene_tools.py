"""Geometry and motion helpers shared by the local live MA-Bench scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import sapien
from scipy.spatial.transform import Rotation

from ma_bench.envs.color_scene import HOME


def wxyz(rotation: Rotation) -> np.ndarray:
    return rotation.as_quat()[[3, 0, 1, 2]]


def as_rotation(quat) -> Rotation:
    return Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]])


def down(yaw_deg: float) -> np.ndarray:
    return wxyz(Rotation.from_euler("Z", yaw_deg, degrees=True) * Rotation.from_euler("Y", 90, degrees=True))


@dataclass(frozen=True)
class Piece:
    shape: str
    center: tuple[float, float, float]
    dimensions: dict[str, float | tuple]
    color: tuple[float, float, float]
    friction: float
    density: float
    rotation: tuple = (1.0, 0.0, 0.0, 0.0)
    collision: bool = True

    def build(self, world, builder) -> None:
        pose = sapien.Pose(self.center, self.rotation)
        if self.collision:
            material = world.scene.create_physical_material(self.friction, .85 * self.friction, .01)
            getattr(builder, f"add_{self.shape}_collision")(
                pose=pose, material=material, density=self.density, **self.dimensions,
            )
        material = sapien.render.RenderMaterial(base_color=[*self.color, 1.0], roughness=.68)
        getattr(builder, f"add_{self.shape}_visual")(pose=pose, material=material, **self.dimensions)


def box(center, half_size, color, *, friction=1.0, density=200.0, rotation=None, collision=True) -> Piece:
    return Piece("box", center, {"half_size": half_size}, color, friction, density,
                 rotation if rotation is not None else (1, 0, 0, 0), collision)


def sphere(center, radius, color, *, friction=1.0, density=200.0) -> Piece:
    return Piece("sphere", center, {"radius": radius}, color, friction, density)


def capsule(center, radius, half_length, color, *, friction=1.0, density=200.0, rotation=None) -> Piece:
    return Piece("capsule", center, {"radius": radius, "half_length": half_length}, color, friction, density,
                 rotation if rotation is not None else (1, 0, 0, 0))


def cylinder(center, radius, half_length, color, *, friction=1.0, density=200.0) -> Piece:
    return Piece("cylinder", center, {"radius": radius, "half_length": half_length}, color, friction, density,
                 wxyz(Rotation.from_euler("Y", -90, degrees=True)))


def compound(world, name, position, pieces, *, fixed=False, quat=(1, 0, 0, 0)):
    builder = world.scene.create_actor_builder()
    for piece in pieces:
        piece.build(world, builder)
    builder.initial_pose = sapien.Pose(position, quat)
    actor = builder.build_static(name) if fixed else builder.build(name)
    world._entity_tags[actor.per_scene_id] = name
    world.bodies[name] = {"kind": "rigid", "actor": actor, "is_fixture": fixed, "attributes": {}}
    return actor


def hinge(world, name, pivot, pieces, *, axis, limits, stiffness, damping):
    builder = world.scene.create_articulation_builder()
    root = builder.create_link_builder()
    root.set_name(f"{name}_root")
    leaf = builder.create_link_builder(root)
    leaf.set_name(name)
    leaf.set_joint_name(f"{name}_joint")
    quat = wxyz(Rotation.align_vectors([axis], [(1., 0., 0.)])[0])
    leaf.set_joint_properties(
        "revolute", [np.deg2rad(limits)], sapien.Pose(pivot, quat), sapien.Pose([0, 0, 0], quat),
        friction=.008, damping=.02,
    )
    for piece in pieces:
        piece.build(world, leaf)
    art = builder.build(fix_root_link=True)
    leaf_link = art.find_link_by_name(name)
    joint = art.get_active_joints()[0]
    joint.set_drive_properties(stiffness=stiffness, damping=damping, force_limit=8.0)
    joint.set_drive_target(0.)
    world._entity_tags[leaf_link.entity.per_scene_id] = name
    world.bodies[name] = {"kind": "hinge", "articulation": art, "link": leaf_link,
                          "is_fixture": True, "attributes": {}, "joint": joint}
    return art


def bin_parts(half_x, half_y, height, color, *, floor=.008, wall=.008):
    """Rectangular open container with an interior floor at z=floor."""
    side_z = floor + height / 2
    return [
        box((0, 0, 0), (half_x, half_y, floor), color, friction=.55),
        *(box((sign * (half_x - wall), 0, side_z), (wall, half_y, height / 2), color)
          for sign in (-1, 1)),
        *(box((0, sign * (half_y - wall), side_z), (half_x - 2 * wall, wall, height / 2), color)
          for sign in (-1, 1)),
    ]


def round_bin(radius, height, color, *, floor=.008, wall=.005, sides=16, friction=.45):
    """Shallow round cup with a solid base and an open circular rim."""
    pieces = [cylinder((0, 0, 0), radius, floor, color, friction=friction)]
    side_z = floor + height / 2
    for angle in np.linspace(0, 2 * np.pi, sides, endpoint=False):
        r = radius - wall
        center = (r * np.cos(angle), r * np.sin(angle), side_z)
        pieces.append(box(center, (wall, radius * np.tan(np.pi / sides), height / 2), color,
                          rotation=wxyz(Rotation.from_euler("Z", angle)), friction=friction))
    return pieces


def set_home(world, agent, values) -> None:
    articulation = world.robots[agent]
    qpos = articulation.get_qpos().copy()
    joints = articulation.get_active_joints()
    for side, target in values.items():
        handle = world.arms[f"{agent}.{side}"]
        handle.target_q = np.asarray(target, dtype=float)
        qpos[handle.arm_indices] = target
        for index, command in zip(handle.arm_indices, target):
            joints[index].set_drive_target(float(command))
    articulation.set_qpos(qpos)


def add_aloha(world, agent, base_xyz, yaw_deg) -> None:
    world.add_robot(agent, "aloha", base_xyz, yaw_deg=yaw_deg)
    set_home(world, agent, HOME)


def worktable(world, center=(0., 0., .74), size=(1.22, .64, .045), color=(.61, .64, .64)):
    world.add_table(center=center, size=size, color=color)
    x, y, z = center
    width, length, thickness = size
    leg_h = (z - thickness) / 2
    for index, (dx, dy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        world.add_box(f"table_leg_{index}", (x + dx * (width / 2 - .06), y + dy * (length / 2 - .055), leg_h),
                      (.022, .022, leg_h), color=(.29, .33, .35), is_static=True)


@dataclass(frozen=True)
class Container:
    name: str
    member_names: tuple[str, ...]
    low: tuple[float, float, float]
    high: tuple[float, float, float]
    radius: float | None = None

    def contains(self, point, center, rotation: Rotation) -> bool:
        local = rotation.inv().apply(np.asarray(point) - center)
        in_box = np.all((self.low <= local) & (local <= self.high))
        return bool(in_box and (self.radius is None or np.linalg.norm(local[:2]) <= self.radius))


@dataclass
class Layout:
    expert: Callable[[Motion], None]
    overview: tuple[tuple[float, float, float], tuple[float, float, float]]
    containers: tuple[Container, ...] = ()
    annotate: Callable | None = None


class MotionError(RuntimeError):
    pass


class Motion:
    def __init__(self, world):
        self.world = world
        self.phases = []

    def move(self, label, targets, steps=30):
        tick = self.world.step_count
        self.phases.append({"phase": label, "tick": tick})
        if not self.world.move_arms_to(targets, steps=steps):
            raise MotionError(f"IK failed at tick {tick} during {label}")

    def hold(self, steps):
        self.world.hold_steps(steps)

    def close(self, body, arm_keys, steps=24, *, openings=(0.,)):
        self.phases.append({"phase": f"grasp {body}", "tick": self.world.step_count})
        for key in arm_keys:
            self.world.allow_grasp(key, body)
        for opening in openings:
            for key in arm_keys:
                self.world.set_gripper(key, opening)
            self.hold(steps // len(openings))

    def open(self, body, arm_keys, steps=24):
        self.phases.append({"phase": f"release {body}", "tick": self.world.step_count})
        for key in arm_keys:
            self.world.set_gripper(key, 1.)
        self.hold(steps)
        for key in arm_keys:
            self.world.allow_grasp(key, body, False)
