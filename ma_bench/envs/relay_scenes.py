"""The bowl relay and barcode presentation scenes."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .scene_tools import Layout, add_aloha, as_rotation, box, compound, down, round_bin, worktable, wxyz


def _transfer(motion, arm, body, grip_offset, destination, yaw, home):
    source = motion.world.get_body_pose(body)[0]
    offset = np.asarray(grip_offset)
    destination = np.asarray(destination)
    rotation = down(yaw)

    def target(point, extra_height=0.):
        return {arm: (point + offset + [0, 0, extra_height], rotation)}

    motion.move(f"{arm} approach {body}", target(source, .085), 32)
    motion.move(f"{arm} reach handle", target(source), 25)
    motion.close(body, (arm,))
    motion.move(f"{arm} lift {body}", target(source, .100), 30)
    motion.move(f"{arm} transfer {body}", target(destination, .100), 42)
    motion.move(f"{arm} set down {body}", target(destination, .002), 30)
    motion.open(body, (arm,))
    motion.move(f"{arm} clear the handle", target(destination, .100), 26)
    motion.move(f"{arm} retreat", {arm: home}, 36)


def build_bowl(world) -> Layout:
    table_z = .74
    worktable(world, center=(0, -.045, table_z), size=(1.30, .64, .044))
    add_aloha(world, "A", (-.59, -.665, 0), 90)
    add_aloha(world, "B", (.59, -.665, 0), 90)
    rng = np.random.default_rng(world.mode.seed)
    position = (-.42 + rng.uniform(-.008, .008), .003, table_z + .008)
    blue = (.10, .31, .53)
    grip = (.92, .62, .20)
    parts = round_bin(.070, .037, blue)
    parts += [box((0., -.076, .031), (.016, .020, .012), blue),
              box((0., -.103, .047), (.019, .017, .021), grip, friction=2.6)]
    compound(world, "bowl", position, parts)
    for key, center, half, color in (
        ("start_zone", -.42, .115, (.75, .42, .13)),
        ("exchange_zone", 0., .125, (.85, .70, .14)),
        ("goal_zone", .45, .12, (.19, .58, .68)),
    ):
        world.add_region(key, (center - half, -.108, table_z + .0005),
                         (center + half, .108, .81), visual_color=color)

    def expert(motion):
        a, b = "A.right", "B.left"
        a_home, b_home = world.get_arm_pose(a), world.get_arm_pose(b)
        offset = (0, -.103, .047 + .143)
        _transfer(motion, a, "bowl", offset, (0., .003, table_z + .008), 90, a_home)
        motion.hold(25)
        _transfer(motion, b, "bowl", offset, (.45, .003, table_z + .008), 90, b_home)
        motion.hold(35)

    return Layout(expert, ((.09, 1.81, 1.97), (0., -.22, .78)))


def build_scan(world) -> Layout:
    table_z = .74
    worktable(world, size=(.94, .59, .044))
    add_aloha(world, "A", (0, -.602, 0), 90)
    add_aloha(world, "B", (0, .625, 0), -90)
    x = -.226 + np.random.default_rng(world.mode.seed).uniform(-.009, .009)
    scan_x = x + .100
    tea = [box((0, 0, 0), (.023, .025, .055), (.74, .69, .27), friction=2.0),
           box((0, 0, .075), (.019, .018, .021), (.74, .69, .27), friction=2.4),
           box((0, .0253, -.013), (.015, .0004, .009), (.91, .91, .84), collision=False)]
    for index in range(8):
        tea.append(box((-.012 + index * .0034, .0259, -.013), (.0008, .0004, .0075), (.06, .075, .065),
                       collision=False))
    compound(world, "tea_box", (x, -.155, table_z + .055), tea)
    scan_axis = np.asarray([-.100, -.052, .057])
    scan_axis /= np.linalg.norm(scan_axis)
    laser = Rotation.align_vectors([scan_axis], [[0., -1., 0.]])[0]
    scanner = [box((0., .017, 0.), (.019, .023, .018), (.18, .25, .31), friction=2.2),
               box((0., -.015, .023), (.020, .020, .013), (.20, .39, .45), rotation=wxyz(laser)),
               box(np.array([0., -.015, .023]) + .0207 * scan_axis, (.013, .001, .007),
                   (.93, .27, .19), rotation=wxyz(laser), collision=False)]
    compound(world, "scanner", (scan_x, .148, table_z + .018), scanner)
    initial = {}
    for name, cx, at, color in (("tea_box", x, -.155, (.73, .58, .25)),
                                ("scanner", scan_x, .148, (.30, .57, .63))):
        world.add_region(f"init_{'box' if name == 'tea_box' else 'scanner'}_zone",
                         (cx - .053, at - .057, table_z + .0005), (cx + .053, at + .057, table_z + .089),
                         visual_color=color)

    def annotate(objects):
        p, q = world.get_body_pose("scanner")
        direction = np.asarray(objects["tea_box"]["pos"]) - p
        forward = as_rotation(q).apply(scan_axis)
        cosine = float(forward @ direction) / max(1e-8, float(np.linalg.norm(direction)))
        objects["scanner"]["angle_to_target_deg"] = float(np.rad2deg(np.arccos(np.clip(cosine, -1., 1.))))

    def expert(motion):
        devices = {"tea_box": ("A.left", 90, (0, 0, .221)),
                   "scanner": ("B.right", -90, (0, .019, .144))}
        for name in devices:
            initial[name] = world.get_body_pose(name)[0]
        home = {arm: world.get_arm_pose(arm) for arm, _, _ in devices.values()}

        def pose(name, point):
            arm, yaw, offset = devices[name]
            return {arm: (np.asarray(point) + offset, down(yaw))}

        for name, (arm, _, _) in devices.items():
            start = initial[name]
            motion.move(f"{arm} approach {name}", pose(name, start + [0, 0, .085]), 28)
            motion.move(f"{arm} reach {name}", pose(name, start), 24)
            motion.close(name, (arm,))
            motion.move(f"{arm} lift {name}", pose(name, [start[0], start[1], .914 if name == "tea_box" else .857]), 32)
        motion.move("present the tea box", pose("tea_box", [x, -.050, .914]), 36)
        motion.move("align the scanner", pose("scanner", [scan_x, .002, .857]), 36)
        motion.hold(32)
        motion.move("scanner withdraws first", pose("scanner", [scan_x, .200, .857]), 32)
        names = ("scanner", "tea_box")
        for name in names:
            start = initial[name]
            arm, _, _ = devices[name]
            motion.move(f"return {name}", pose(name, [start[0], start[1], .914 if name == "tea_box" else .857]), 30)
            motion.move(f"set down {name}", pose(name, start + [0, 0, .002]), 30)
            motion.open(name, (arm,))
            motion.move(f"{arm} clear the object", pose(name, start + [0, 0, .080]), 24)
            motion.move(f"{arm} retreat", {arm: home[arm]}, 30)
        motion.hold(32)

    return Layout(expert, ((-1.64, -.31, 1.72), (-.12, 0, .84)), annotate=annotate)
