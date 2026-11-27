"""Panda holds a passive spring door while ALOHA inserts a handled pot."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .scene_tools import Layout, MotionError, add_aloha, box, compound, cylinder, down, hinge, round_bin, set_home, worktable, wxyz


def build_door(world) -> Layout:
    table = .74
    shelf = .842
    worktable(world, center=(.04, .095, table), size=(1.52, .70, .045))
    world.add_robot("A", "panda", (.59, -.518, 1.122), yaw_deg=140)
    compound(world, "panda_pedestal", (.59, -.518, .561),
             [cylinder((0., 0., 0.), .055, .561, (.25, .30, .34))], fixed=True)
    panda_down = Rotation.from_euler("Z", 105, degrees=True) * Rotation.from_euler("X", 180, degrees=True)
    ready, valid = world.solve_ik("A.main", (.270, -.346, 1.485), wxyz(panda_down))
    if not valid:
        raise MotionError("Panda ready pose is outside its reachable workspace")
    set_home(world, "A", {"main": ready})
    panda_arm = world.arms["A.main"]
    panda_joints = panda_arm.articulation.get_active_joints()
    for index in panda_arm.arm_indices:
        panda_joints[index].set_drive_properties(stiffness=2000., damping=240., force_limit=220.)
    add_aloha(world, "B", (0., -.535, 0.), 90)
    # Park both ALOHA arms outward to keep the spring-door sweep clear.
    parked = {side: world.arms[f"B.{side}"].target_q.copy() for side in ("left", "right")}
    parked["left"][0] = .90
    parked["right"] = [.2761, -.06706, 1.48639, .06269, -.09535, -1.09597]
    set_home(world, "B", parked)

    frame = (.53, .59, .63)
    dark = (.32, .40, .47)
    compound(world, "cabinet_frame", (0., 0., 0.), [
        box((-.246, .178, 1.038), (.017, .115, .256), frame),
        box((.246, .178, 1.038), (.017, .115, .256), frame),
        box((0., .285, 1.038), (.229, .012, .256), dark),
        box((0., .181, 1.302), (.258, .118, .016), frame),
    ], fixed=True)
    world.add_box("cabinet_shelf", (0., .150, (shelf + table) / 2), (.242, .143, (shelf - table) / 2),
                  color=dark, is_static=True)
    world.add_region("cabinet_interior", (-.17, .064, shelf), (.17, .235, .963))

    pivot = np.asarray((.232, -.013, 1.012))
    handle = np.asarray((-.213, -.147, .287))
    cabinet_color = (.19, .46, .56)
    gold = (.93, .62, .16)
    hinge(world, "door", pivot, [
        # The closed door intersects the pot's insertion path, while the
        # lower edge clears the pot at its starting height on the table.
        box((-.233, 0., .075), (.233, .009, .147), cabinet_color, density=115.),
        box(handle + [0., .071, -.113], (.011, .079, .009), gold),
        box(handle + [0., 0., -.063], (.009, .009, .058), gold),
        box(handle, (.019, .019, .026), gold, rotation=wxyz(Rotation.from_euler("Z", 105, degrees=True)),
            friction=2.8),
    ], axis=(0., 0., 1.), limits=(0., 168.), stiffness=.35, damping=.60)

    pot_height = .008
    pot_pos = np.asarray((0., -.139, table + pot_height))
    steel = (.40, .48, .53)
    pot_pieces = round_bin(.076, .065, steel, floor=pot_height)
    for sign in (-1, 1):
        pot_pieces.extend((
            box((sign * .102, 0, .047), (.030, .015, .012), steel),
            box((sign * .130, 0, .064), (.009, .012, .018), steel),
            box((sign * .130, 0, .086), (.019, .017, .019), gold, friction=2.8),
        ))
    pot_pieces.append(cylinder((0., 0., .069), .069, .004, (.52, .58, .60)))
    pot_pieces.append(cylinder((0., 0., .084), .019, .011, (.19, .22, .23)))
    compound(world, "pot", pot_pos, pot_pieces)

    def expert(motion):
        panda = "A.main"
        grip_z = .138

        def door_target(angle, clearance=0.):
            turn = Rotation.from_euler("Z", angle, degrees=True)
            pos = pivot + turn.apply(handle) + [0., 0., grip_z + clearance]
            return {panda: (pos, wxyz(turn * panda_down))}

        motion.move("Panda approach the door handle", door_target(0., .075), 40)
        motion.move("Panda reach the door handle", door_target(0.), 32)
        motion.close("door", (panda,), 28)
        for angle in (*range(8, 97, 8), *range(100, 149, 4)):
            motion.move(f"open spring door to {angle} deg", door_target(angle), 22)
        motion.hold(25)
        if world.get_hinge_angle_deg("door") <= world.spec.thresholds["th_open_deg"]:
            raise MotionError(f"Door stopped at {world.get_hinge_angle_deg('door'):.1f} deg")

        arms = ("B.left", "B.right")
        home = {key: world.get_arm_pose(key) for key in arms}
        source, _ = world.get_body_pose("pot")
        destination = np.array([0., .084, shelf + pot_height])

        def pot_target(center, clearance=0.):
            return {key: (center + [sign * .130, 0., .086 + .147 + clearance], down(90))
                    for key, sign in zip(arms, (-1, 1))}

        motion.move("ALOHA approach both pot handles", pot_target(source, .085), 34)
        motion.move("ALOHA seat both grasps", pot_target(source), 29)
        motion.close("pot", arms, 30)
        motion.move("lift the pot with both arms", pot_target(source, .123), 40)
        motion.move("carry the pot through the open door", pot_target(destination, .010), 70)
        motion.move("lower the pot onto the shelf", pot_target(destination, 0.), 40)
        motion.hold(22)
        motion.open("pot", arms, 28)
        motion.move("withdraw both open grippers", pot_target(destination - [0., .232, 0.], .026), 48)
        motion.move("ALOHA retreat", home, 42)
        motion.hold(35)

    return Layout(expert, ((-1.22, -1.38, 1.84), (.03, .015, .90)))
