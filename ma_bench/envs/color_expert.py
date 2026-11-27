"""Color-inventory exchange followed by synchronized physical pick-and-place."""
import numpy as np


class ExpertFailure(RuntimeError):
    pass


GRASP_QUAT = (0.5, -0.5, 0.5, 0.5)
GRASP_HEIGHT = 0.147


def execute_expert(world, layout):
    # Exchange color inventories, not scorer-specific object identities.
    inventories = {
        aid: tuple(layout.block_colors[aid].values())
        for aid in ("A", "B")
    }
    common = set(inventories["A"]).intersection(inventories["B"])
    if len(common) != 1:
        raise ExpertFailure("inventory exchange: expected one shared color")
    shared_color = next(iter(common))

    jobs = {}
    for aid in ("A", "B"):
        matches = [
            name for name, color in layout.block_colors[aid].items()
            if color == shared_color
        ]
        if len(matches) != 1:
            raise ExpertFailure(f"inventory selection: ambiguous color for {aid}")
        body = matches[0]
        position, _ = world.get_body_pose(body)
        source = np.asarray(position, dtype=float).copy()
        pad = np.asarray(layout.pad_centers[aid], dtype=float)
        # Pad X is the robot's center X.
        side = "left" if source[0] < pad[0] else "right"
        jobs[f"{aid}.{side}"] = {
            "body": body,
            "source": source,
            "destination": pad.copy(),
        }

    def move(phase, location, height, steps):
        targets = {}
        for arm, job in jobs.items():
            xyz = job[location].copy()
            xyz[2] += height
            targets[arm] = (tuple(float(v) for v in xyz), GRASP_QUAT)
        if not world.move_arms_to(targets, steps=steps):
            raise ExpertFailure(f"{phase}: IK failed")

    # Both robots execute each phase together. All cube motion below is due
    # to contact, gravity, and gripper actuation; no actor poses are written.
    for arm in jobs:
        world.set_gripper(arm, 1.0)
    world.hold_steps(4)

    move("source approach", "source", 0.22, 34)
    move("grasp descent", "source", GRASP_HEIGHT, 24)

    for arm, job in jobs.items():
        world.allow_grasp(arm, job["body"], allowed=True)
        world.set_gripper(arm, 0.0)
    world.hold_steps(24)

    move("lift", "source", 0.22, 28)
    move("pad transit", "destination", 0.22, 34)
    move("placement descent", "destination", GRASP_HEIGHT, 24)

    for arm in jobs:
        world.set_gripper(arm, 1.0)
    world.hold_steps(24)
    for arm, job in jobs.items():
        world.allow_grasp(arm, job["body"], allowed=False)

    move("empty retreat", "destination", 0.22, 28)

    # Nominal total: 254 ticks, including 30 final idle ticks at 25 Hz.
    # The caller independently evaluates physical grasp and placement success.
    world.hold_steps(30)
