"""Tilt a spring-mounted hopper while a UR5 catches three physical balls."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .scene_tools import Container, Layout, bin_parts, box, compound, hinge, set_home, sphere, worktable, wxyz


def build_catch(world) -> Layout:
    table_z = .74
    worktable(world, center=(.15, .01, table_z), size=(1.48, 1.11, .046))
    for agent, embodiment, root, yaw, home in (
        ("A", "panda", (-.25, -.45, .90), 0., [.323, -.247, .285, -2.088, .072, 1.85, 1.366]),
        ("B", "ur5", (.70, .45, .89), 180., [-2.732, -1.311, -2.465, -.936, 1.571, -2.732]),
    ):
        world.add_robot(agent, embodiment, root, yaw_deg=yaw)
        set_home(world, agent, {"main": home})
        pedestal_h = (root[2] - table_z) / 2
        world.add_box(f"stand_{agent}", (root[0], root[1], table_z + pedestal_h),
                      (.105, .105, pedestal_h), color=(.31, .37, .42), is_static=True)
    ur5 = world.arms["B.main"]
    for index in ur5.finger_indices:
        joint = ur5.articulation.get_active_joints()[index]
        joint.set_friction(.05)
        joint.set_drive_properties(stiffness=1050., damping=90., force_limit=85.)

    pivot = np.array([0., .012, 1.14])
    blue = (.18, .48, .59)
    gold = (.94, .63, .18)
    for sign in (-1, 1):
        h = (pivot[2] - table_z) / 2
        world.add_box(f"hopper_stand_{sign}", (0., pivot[1] + sign * .143, table_z + h),
                      (.021, .016, h), color=(.40, .48, .52), is_static=True)
    grip = np.array([.140, -.187, .078])
    tip = np.array([.318, 0., .028])
    floor_tilt = wxyz(Rotation.from_euler("Y", -13, degrees=True))
    hopper = [
        box((.018, 0., 0.), (.181, .105, .007), blue, friction=.35, density=130.),
        box((.257, 0., .013), (.061, .105, .007), blue, friction=.32, density=130., rotation=floor_tilt),
        box((-.160, 0., .037), (.007, .105, .036), blue, density=130.),
        box((-.116, 0., -.039), (.056, .062, .025), (.32, .43, .50), density=445.),
        box((grip[0], grip[1] + .038, .006), (.013, .045, .008), gold),
        box((grip[0], grip[1], .041), (.007, .008, .036), gold),
        box(grip, (.018, .017, .018), gold, friction=2.4),
    ]
    for sign in (-1, 1):
        hopper.append(box((.077, sign * .104, .039), (.239, .007, .036), blue, density=115.))
    hinge(world, "tipping_hopper", pivot, hopper, axis=(0., 1., 0.),
          limits=(0., 48.), stiffness=.95, damping=.55)

    ball_names = ("ball_1", "ball_2", "ball_3")
    rng = np.random.default_rng(world.mode.seed)
    for name, y, color in zip(ball_names, (-.052, 0., .052), ((.94, .68, .17), (.86, .21, .20), (.30, .67, .65))):
        compound(world, name, pivot + [-.09 + rng.uniform(-.025, .025), y, .024],
                 [sphere((0, 0, 0), .016, color, friction=.45, density=270.)])

    width, depth, height, floor = .142, .092, .083, .007
    orange = (.81, .40, .17)
    cup_grip = np.array([0., .173, .085])
    cup_parts = bin_parts(width, depth, height, orange, floor=floor, wall=.006)
    cup_parts += [box((0., .129, .041), (.014, .048, .010), orange),
                  box((0., .173, .064), (.009, .009, .021), orange),
                  box(cup_grip, (.017, .019, .023), gold, friction=2.3)]
    start = np.array([.38, .257, table_z + floor])
    compound(world, "receiving_cup", start, cup_parts)
    target = np.array([.43, .265, table_z + floor])
    world.add_region("cup_target_zone", target - [.037, .038, .006], target + [.037, .038, .019],
                     visual_color=(.25, .60, .44))

    def expert(motion):
        panda, receiver = "A.main", "B.main"
        p_down = Rotation.from_euler("X", 180., degrees=True)
        ur_down = wxyz(Rotation.from_euler("Y", 90., degrees=True))
        panda_grip = grip + [0., 0., .111]

        def p_target(angle, extra=0.):
            turn = Rotation.from_euler("Y", angle, degrees=True)
            return {panda: (pivot + turn.apply(panda_grip) + [0., 0., extra], wxyz(turn * p_down))}

        def cup_target(center, extra=0.):
            return {receiver: (center + cup_grip + [0., 0., .137 + extra], ur_down)}

        def catch_center(angle):
            outlet = pivot + Rotation.from_euler("Y", angle, degrees=True).apply(tip)
            return np.array([outlet[0] + .045, pivot[1], max(.802, outlet[2] - .137)])

        actual_cup = world.get_body_pose("receiving_cup")[0]
        home_p = world.get_arm_pose(panda)
        home_r = world.get_arm_pose(receiver)
        motion.move("UR5 approach cup handle", cup_target(actual_cup, .096), 35)
        motion.move("UR5 reach cup handle", cup_target(actual_cup), 30)
        motion.close("receiving_cup", (receiver,), 56, openings=(.72, .53, .39, .29, .17, .08, 0.))
        motion.move("lift receiving cup", cup_target(actual_cup + [0., 0., .16]), 42)
        motion.move("move cup under hopper outlet", cup_target(catch_center(0.)), 52)
        motion.move("Panda approach hopper handle", p_target(0., .074), 32)
        motion.move("Panda reach hopper handle", p_target(0.), 32)
        motion.close("tipping_hopper", (panda,), 30)

        for angle in (4, 8, 12, 16, 20):
            motion.move(f"tilt to {angle} deg and track outlet", {**p_target(angle), **cup_target(catch_center(angle))}, 18)
        motion.hold(64)
        for angle in (16, 12, 8, 4, 0):
            motion.move(f"return hopper through {angle} deg", p_target(angle), 16)
        motion.open("tipping_hopper", (panda,))
        position, orientation = world.get_arm_pose(panda)
        for clearance in (.015, .030, .048, .067, .090, .110):
            motion.move("lift open fingers clear of the handle",
                        {panda: (position + [0., 0., clearance], orientation)}, 8)
        motion.move("Panda retreat", {panda: home_p}, 34)

        motion.move("withdraw the cup along the tabletop", cup_target(target + [0., 0., .16]), 62)
        motion.move("carry caught balls to target", cup_target(target + [0., 0., .085]), 38)
        motion.move("set cup on target", cup_target(target + [0., 0., .002]), 35)
        motion.open("receiving_cup", (receiver,))
        motion.move("UR5 withdraw open fingers", cup_target(target + [0., 0., .109]), 34)
        motion.move("UR5 retreat", {receiver: home_r}, 39)
        motion.hold(35)

    volume = Container("receiving_cup", ball_names, (-width + .006, -depth + .006, floor),
                       (width - .006, depth - .006, floor + height))
    return Layout(expert, ((1.35, -1.57, 2.05), (.18, -.01, 1.03)), containers=(volume,))
