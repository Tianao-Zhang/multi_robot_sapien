"""Four-point, friction-only tray manipulation with two opposing ALOHAs.

All payloads are independent dynamic bodies. The expert only commands arm
poses and gripper opening/closing; it never moves or attaches a body directly.

Tuning:
    - HANDLE_FRICTION and HANDLE_HALF_SIZE control grasp security.
    - POUR_DECK_FRICTION, POUR_ANGLE_DEG, and POUR_DWELL_STEPS control discharge.
    - POUR_LIFT and the bag dimensions jointly determine pouring clearance.
    - RIGID_WAYPOINT_STEPS controls how finely rigid tool paths are sampled.

Both experts take approximately 400 control steps, including real settling.
"""

from math import ceil

from scipy.spatial.transform import Rotation

from .scene_tools import (
    Container,
    Layout,
    add_aloha,
    bin_parts,
    box,
    capsule,
    compound,
    down,
    sphere,
    worktable,
)


TABLE_TOP = 0.740
TABLE_SIZE = (1.65, 0.69, 0.045)

TRAY_HALF_X = 0.290
TRAY_HALF_Y = 0.240
DECK_HALF_Z = 0.006
TRAY_START = (-0.160, 0.0, TABLE_TOP + DECK_HALF_Z)

KEEP_SHIFT_X = 0.100
KEEP_LIFT = 0.115

POUR_SHIFT_X = 0.180
POUR_LIFT = 0.125
POUR_ANGLE_DEG = 12.0
POUR_DWELL_STEPS = 84

HANDLE_Z = 0.075
HANDLE_HALF_SIZE = (0.019, 0.018, 0.019)
HANDLE_FRICTION = 2.5
KEEP_DECK_FRICTION = 0.75
POUR_DECK_FRICTION = 0.045
TOOL_ABOVE_HANDLE = 0.147

BAG_CENTER = (0.510, 0.0, 0.605)
BAG_HALF_X = 0.250
BAG_HALF_Y = 0.180
BAG_HEIGHT = 0.085
BAG_FLOOR = 0.008
BAG_WALL = 0.008

RIGID_WAYPOINT_STEPS = 8

ARMS = ("A.left", "A.right", "B.right", "B.left")
HANDLE_CENTERS = {
    "A.left": (-0.240, -0.200, HANDLE_Z),
    "A.right": (0.240, -0.200, HANDLE_Z),
    "B.right": (-0.240, 0.200, HANDLE_Z),
    "B.left": (0.240, 0.200, HANDLE_Z),
}
ARM_YAWS = {
    "A.left": 90.0,
    "A.right": 90.0,
    "B.right": -90.0,
    "B.left": -90.0,
}

TRAY_COLOR = (0.23, 0.38, 0.46)
RAIL_COLOR = (0.30, 0.46, 0.53)
HANDLE_COLOR = (0.93, 0.65, 0.16)
BAG_COLOR = (0.72, 0.60, 0.39)


def _add(a, b):
    return tuple(float(a[i] + b[i]) for i in range(3))


def _rotation(wxyz):
    """Convert the helper/world quaternion convention to scipy's convention."""
    w, x, y, z = wxyz
    return Rotation.from_quat((x, y, z, w))


def _quaternion(rotation):
    x, y, z, w = rotation.as_quat()
    return (float(w), float(x), float(y), float(z))


def _smooth(t):
    """Quintic easing: zero velocity and acceleration at segment endpoints."""
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


class _RigidToolPath:
    """Preserve four measured tool poses relative to a measured tray frame.

    This is only a target generator, not a simulator constraint. Finger
    friction must actually support the tray throughout every commanded move.
    """

    def __init__(self, motion):
        self.motion = motion
        position, quaternion = motion.world.get_body_pose("tray")
        self.position = tuple(float(v) for v in position)
        self.rotation = _rotation(quaternion)

        inverse = self.rotation.inv()
        self.local_positions = {}
        self.local_rotations = {}
        for arm in ARMS:
            tool_position, tool_quaternion = motion.world.get_arm_pose(arm)
            displacement = tuple(
                float(tool_position[i] - self.position[i]) for i in range(3)
            )
            self.local_positions[arm] = inverse.apply(displacement)
            self.local_rotations[arm] = inverse * _rotation(tool_quaternion)

    def move(self, label, position, rotation, steps):
        """Interpolate both translation and rotation through rigid waypoints."""
        start_position = self.position
        start_rotation = self.rotation
        rotation_vector = (start_rotation.inv() * rotation).as_rotvec()

        count = max(1, int(ceil(steps / float(RIGID_WAYPOINT_STEPS))))
        base_steps, remainder = divmod(steps, count)

        for index in range(1, count + 1):
            t = _smooth(index / float(count))
            center = tuple(
                start_position[j] + t * (position[j] - start_position[j])
                for j in range(3)
            )
            orientation = start_rotation * Rotation.from_rotvec(
                rotation_vector * t
            )

            targets = {}
            for arm in ARMS:
                tool_position = _add(
                    center, orientation.apply(self.local_positions[arm])
                )
                tool_rotation = orientation * self.local_rotations[arm]
                targets[arm] = (tool_position, _quaternion(tool_rotation))

            waypoint_steps = base_steps + (1 if index <= remainder else 0)
            self.motion.move(
                "{} {}/{}".format(label, index, count),
                targets,
                steps=waypoint_steps,
            )

        self.position = tuple(float(v) for v in position)
        self.rotation = rotation


def _make_tray(world, pour):
    deck_friction = POUR_DECK_FRICTION if pour else KEEP_DECK_FRICTION
    pieces = [
        box(
            (0.0, 0.0, 0.0),
            (TRAY_HALF_X, TRAY_HALF_Y, DECK_HALF_Z),
            TRAY_COLOR,
            friction=deck_friction,
            density=160.0,
        )
    ]

    # Low side/back rails contain incidental motion, but the +X discharge
    # edge is deliberately flush. No crossbar spans the cargo corridor.
    rail_height = 0.024
    rail_z = DECK_HALF_Z + rail_height / 2.0
    for y in (-TRAY_HALF_Y + 0.006, TRAY_HALF_Y - 0.006):
        pieces.append(
            box(
                (0.0, y, rail_z),
                (TRAY_HALF_X, 0.006, rail_height / 2.0),
                RAIL_COLOR,
                friction=deck_friction,
                density=160.0,
            )
        )
    pieces.append(
        box(
            (-TRAY_HALF_X + 0.006, 0.0, rail_z),
            (0.006, TRAY_HALF_Y, rail_height / 2.0),
            RAIL_COLOR,
            friction=deck_friction,
            density=160.0,
        )
    )

    for x, y, z in HANDLE_CENTERS.values():
        # Narrow pedestals end inside the raised pinch pads. Their position
        # at y=+/-0.20 leaves the middle of the deck unobstructed.
        pieces.append(
            box(
                (x, y, 0.034),
                (0.006, 0.011, 0.028),
                HANDLE_COLOR,
                friction=HANDLE_FRICTION,
                density=300.0,
            )
        )
        pieces.append(
            box(
                (x, y, z),
                HANDLE_HALF_SIZE,
                HANDLE_COLOR,
                friction=HANDLE_FRICTION,
                density=300.0,
            )
        )

    compound(world, "tray", TRAY_START, pieces, fixed=False)


def _make_blocks(world):
    half = 0.019
    deck_top = TRAY_START[2] + DECK_HALF_Z
    colors = (
        (0.88, 0.30, 0.24),
        (0.94, 0.69, 0.20),
        (0.27, 0.62, 0.39),
        (0.30, 0.48, 0.86),
    )

    # One genuine four-high stack, not four blocks glued into a compound.
    for index, color in enumerate(colors):
        position = (
            TRAY_START[0] - 0.035,
            0.0,
            deck_top + half + index * (2.0 * half) + 0.0004,
        )
        compound(
            world,
            "block_{}".format(index + 1),
            position,
            [
                box(
                    (0.0, 0.0, 0.0),
                    (half, half, half),
                    color,
                    friction=0.85,
                    density=240.0,
                )
            ],
            fixed=False,
        )


def _make_food_and_bag(world):
    compound(
        world,
        "bag",
        BAG_CENTER,
        bin_parts(
            BAG_HALF_X,
            BAG_HALF_Y,
            BAG_HEIGHT,
            BAG_COLOR,
            floor=BAG_FLOOR,
            wall=BAG_WALL,
        ),
        fixed=True,
    )

    deck_top = TRAY_START[2] + DECK_HALF_Z
    compound(
        world,
        "apple",
        (TRAY_START[0] - 0.120, -0.065, deck_top + 0.0275),
        [
            sphere(
                (0.0, 0.0, 0.0),
                0.027,
                (0.83, 0.16, 0.12),
                friction=0.32,
                density=650.0,
            )
        ],
        fixed=False,
    )
    compound(
        world,
        "carrot",
        (TRAY_START[0] - 0.025, 0.060, deck_top + 0.0145),
        [
            capsule(
                (0.0, 0.0, 0.0),
                0.014,
                0.042,
                (0.96, 0.43, 0.08),
                friction=0.32,
                density=650.0,
            )
        ],
        fixed=False,
    )
    compound(
        world,
        "cucumber",
        (TRAY_START[0] + 0.100, -0.025, deck_top + 0.0195),
        [
            capsule(
                (0.0, 0.0, 0.0),
                0.019,
                0.057,
                (0.16, 0.43, 0.16),
                friction=0.30,
                density=650.0,
            )
        ],
        fixed=False,
    )


def _approach_and_grasp(motion, pour):
    # Let the tray, loose food, or stack settle under gravity first.
    motion.hold(16)

    tray_position, tray_quaternion = motion.world.get_body_pose("tray")
    tray_rotation = _rotation(tray_quaternion)
    grasp_targets = {}
    for arm in ARMS:
        x, y, z = HANDLE_CENTERS[arm]
        offset = (x, y, z + TOOL_ABOVE_HANDLE)
        position = _add(tray_position, tray_rotation.apply(offset))
        grasp_targets[arm] = (position, down(ARM_YAWS[arm]))

    def elevated_targets(height):
        return {
            arm: (_add(position, (0.0, 0.0, height)), quaternion)
            for arm, (position, quaternion) in grasp_targets.items()
        }

    # Explicitly establish open fingers before descending around the posts.
    # No body is attached; close() below only enables finger contact and
    # commands the opposing jaws to squeeze.
    motion.open("tray", ARMS, steps=8)
    motion.move(
        "clearance above four handles",
        elevated_targets(0.100),
        steps=12 if pour else 16,
    )
    motion.move(
        "approach raised pinch pads",
        elevated_targets(0.030),
        steps=10 if pour else 12,
    )
    motion.move(
        "seat opposing fingers around posts",
        grasp_targets,
        steps=12,
    )
    motion.close("tray", ARMS, steps=24)
    motion.hold(6 if pour else 8)
    return _RigidToolPath(motion)


def _release_and_withdraw(motion, *, settle, withdraw, final_rest):
    # Support the deck on the table before opening all four pinches.
    motion.hold(settle)
    motion.open("tray", ARMS, steps=24)

    targets = {}
    for arm in ARMS:
        position, quaternion = motion.world.get_arm_pose(arm)
        targets[arm] = (_add(position, (0.0, 0.0, 0.100)), quaternion)
    motion.move("withdraw open fingers vertically", targets, steps=withdraw)
    motion.hold(final_rest)


def _keep_expert(motion):
    path = _approach_and_grasp(motion, pour=False)
    start = path.position
    level = path.rotation

    lifted = (start[0], start[1], start[2] + KEEP_LIFT)
    translated = (start[0] + KEEP_SHIFT_X, start[1], lifted[2])
    placed = (
        TRAY_START[0] + KEEP_SHIFT_X,
        TRAY_START[1],
        TABLE_TOP + DECK_HALF_Z,
    )

    # Deliberately slow acceleration protects the single four-block tower.
    path.move("lift loaded tray level", lifted, level, steps=48)
    path.move("carry stack into goal zone", translated, level, steps=64)
    path.move("lower tray onto table", placed, level, steps=48)

    _release_and_withdraw(
        motion, settle=24, withdraw=24, final_rest=72
    )


def _pour_expert(motion):
    path = _approach_and_grasp(motion, pour=True)
    start = path.position
    level = path.rotation

    lifted = (start[0], start[1], start[2] + POUR_LIFT)
    pouring_center = (
        start[0] + POUR_SHIFT_X,
        start[1],
        lifted[2],
    )
    # Positive world-Y rotation lowers the tray's +X edge. Left-multiply
    # so the rotation axis remains world Y, independent of tray orientation.
    tilted = Rotation.from_euler("y", POUR_ANGLE_DEG, degrees=True) * level

    path.move("lift above bag lip", lifted, level, steps=24)
    path.move("align discharge edge over bag opening", pouring_center, level, steps=48)
    path.move("tilt all four pinches together", pouring_center, tilted, steps=40)

    # The bag sits on a lower stand just off the tabletop. Its open rim
    # stays below the discharge lip during the entire tilt.
    motion.hold(POUR_DWELL_STEPS)

    path.move("level empty tray above bag", pouring_center, level, steps=32)

    return_high = (
        TRAY_START[0] + KEEP_SHIFT_X,
        TRAY_START[1],
        lifted[2],
    )
    placed = (
        return_high[0],
        return_high[1],
        TABLE_TOP + DECK_HALF_Z,
    )
    # First retract horizontally at height; only then descend. At this
    # location the tray's +X edge clears the bag's near wall.
    path.move("return empty tray clear of bag", return_high, level, steps=24)
    path.move("set empty tray on table", placed, level, steps=28)

    _release_and_withdraw(
        motion, settle=16, withdraw=22, final_rest=24
    )


def build_tray(world, *, pour=False) -> Layout:
    """Build either the stacked-block carry or loose-food pouring scene."""
    if pour:
        worktable(world, center=(-.225, 0., TABLE_TOP), size=(.95, .69, .045), color=(.60, .49, .37))
        compound(world, "bag_stand", (BAG_CENTER[0], 0., .296),
                 [box((0., 0., 0.), (.092, .075, .299), (.30, .34, .38))], fixed=True)
    else:
        worktable(world, center=(0.0, 0.0, TABLE_TOP), size=TABLE_SIZE, color=(0.60, 0.49, 0.37))
    add_aloha(world, "A", (0.0, -0.67, 0.0), 90.0)
    add_aloha(world, "B", (0.0, 0.67, 0.0), -90.0)

    _make_tray(world, pour)

    goal_x = TRAY_START[0] + KEEP_SHIFT_X
    # This narrow placement band marks the desired tray-center location;
    # a full-footprint zone would already contain the starting tray center
    # for a carry of only 10 cm.
    world.add_region(
        "tray_goal_zone",
        bounds_min=(goal_x - 0.035, -0.250, TABLE_TOP - 0.002),
        bounds_max=(goal_x + 0.035, 0.250, TABLE_TOP + 0.030),
        visual_color=(0.20, 0.65, 0.38),
    )

    if pour:
        _make_food_and_bag(world)
        containers = (
            Container(
                "bag",
                ("apple", "carrot", "cucumber"),
                low=(
                    -BAG_HALF_X + BAG_WALL,
                    -BAG_HALF_Y + BAG_WALL,
                    BAG_FLOOR / 2.0,
                ),
                high=(
                    BAG_HALF_X - BAG_WALL,
                    BAG_HALF_Y - BAG_WALL,
                    BAG_HEIGHT,
                ),
            ),
            Container("tray", ("apple", "carrot", "cucumber"),
                      low=(-0.285, -0.23, DECK_HALF_Z), high=(0.285, 0.23, 0.11)),
        )
        expert = _pour_expert
    else:
        _make_blocks(world)
        containers = (
            Container(
                "tray",
                ("block_1", "block_2", "block_3", "block_4"),
                low=(-0.275, -0.220, DECK_HALF_Z),
                high=(0.275, 0.220, 0.200),
            ),
        )
        expert = _keep_expert

    return Layout(
        expert=expert,
        overview=((1.65, -1.85, 1.85), (0.10, 0.0, 0.83)),
        containers=containers,
    )
