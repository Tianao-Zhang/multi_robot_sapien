"""Two independent ALOHA workspaces with one shared cube color."""
from dataclasses import dataclass

import numpy as np
import sapien

from ma_task_harness.runtime.sapien_world import look_at_pose


AGENTS = ("A", "B")
VIEWS = ("head", "left_wrist", "right_wrist")
HALF_SIZE = 0.019

COLORS = {
    "red": (0.85, 0.08, 0.08),
    "green": (0.08, 0.70, 0.18),
    "blue": (0.08, 0.25, 0.90),
    "yellow": (0.95, 0.80, 0.08),
    "purple": (0.60, 0.12, 0.80),
}
HOME = {
    "left": [-.1723, 1.2463, 1.1516, -1.4762, .0008, -.1574],
    "right": [.1694, 1.2489, 1.1574, -1.4793, .0008, .1744],
}


@dataclass
class ColorScene:
    pad_centers: dict[str, tuple[float, float, float]]
    block_colors: dict[str, dict[str, str]]
    wrist_links: dict[str, sapien.physx.PhysxArticulationLinkComponent]


def update_wrist_cameras(world, layout):
    """Call after physics updates, before rendering wrist observations."""
    # Look past the physical camera casing toward the gripper fingers.
    offset = look_at_pose((0.018, -0.022, 0.006), (0.139, -0.032, -0.033))
    for name, link in layout.wrist_links.items():
        world.cameras[name].set_entity_pose(link.get_entity_pose() * offset)


def build_scene(world) -> ColorScene:
    rng = np.random.default_rng(world.mode.seed)
    palette = rng.permutation(world.spec.scene_params["palette"])
    shared = str(palette[0])
    inventories = {
        "A": [shared, str(palette[1]), str(palette[2])],
        "B": [shared, str(palette[3]), str(palette[4])],
    }
    layout = ColorScene({}, {}, {})

    half_gap = world.spec.scene_params["separation_m"] / 2
    for aid, base_x in zip(AGENTS, (-half_gap, half_gap)):
        world.add_robot(
            aid, "aloha", base_pos=(base_x, -0.65, 0.0), yaw_deg=90
        )
        robot = world.robots[aid]
        qpos = np.array(robot.get_qpos(), copy=True)
        joints = robot.get_active_joints()

        # Initialize once, before any simulation steps; match drive targets.
        for side, values in HOME.items():
            arm = world.arms[f"{aid}.{side}"]
            indices = np.asarray(arm.arm_indices, dtype=int)
            qpos[indices] = values
            arm.target_q = np.array(values, dtype=float)
        robot.set_qpos(qpos)
        for side, values in HOME.items():
            for index, value in zip(
                world.arms[f"{aid}.{side}"].arm_indices, values
            ):
                joints[int(index)].set_drive_target(float(value))

        world.add_table(
            center=(base_x, 0.05, 0.75),
            size=(0.85, 0.65, 0.04),
            color=(0.48, 0.43, 0.36),
        )
        for index, (dx, y) in enumerate(((-.35, -.21), (.35, -.21), (-.35, .31), (.35, .31))):
            world.add_box(
                f"table_{aid}_leg_{index}", (base_x + dx, y, .355), (.022, .022, .355),
                color=(.24, .26, .29), is_static=True,
            )

        pad = (base_x, -0.16, 0.77)
        layout.pad_centers[aid] = pad
        # A scoring/visual region, not another collision surface.
        world.add_region(
            name=f"pad_{aid}",
            bounds_min=(base_x - 0.06, -0.22, 0.75),
            bounds_max=(base_x + 0.06, -0.10, 0.80),
            visual_color=(0.13, 0.65, 0.60),
        )

        layout.block_colors[aid] = {}
        order = rng.permutation(3)
        for slot, inventory_index in enumerate(order):
            color = inventories[aid][int(inventory_index)]
            suffix = (
                "shared" if inventory_index == 0
                else f"distractor_{int(inventory_index)}"
            )
            name = f"block_{aid.lower()}_{suffix}"
            jitter = rng.uniform(-0.002, 0.002, size=2)
            position = (
                base_x + (-0.14, 0.0, 0.14)[slot] + float(jitter[0]),
                -0.04 + float(jitter[1]),
                0.75 + HALF_SIZE,
            )
            world.add_box(
                name=name,
                pos=position,
                half_size=(HALF_SIZE,) * 3,
                color=COLORS[color],
                mass=0.045,
                friction=1.0,
                attributes={"color": color},
            )
            layout.block_colors[aid][name] = color

        world.add_camera(
            name=f"{aid}/head",
            position=(base_x, -0.32, 1.75),
            target=(base_x, -0.10, 0.76),
            fov_deg=37,
        )
        for side in ("left", "right"):
            name = f"{aid}/{side}_wrist"
            link = robot.find_link_by_name(f"{side}_camera")
            if link is None:
                raise RuntimeError(f"Missing wrist camera link: {name}")
            world.add_camera(
                name=name,
                position=(base_x, -0.56, 1.38),
                target=(base_x, -0.07, 0.77),
                fov_deg=62,
            )
            layout.wrist_links[name] = link

    overview = world.scene.add_camera("overview", 800, 360, np.deg2rad(44), .05, 25.0)
    overview.set_entity_pose(look_at_pose((0.0, 2.2, 2.15), (0.0, -.12, .65)))
    world.cameras["overview"] = overview

    update_wrist_cameras(world, layout)
    return layout
