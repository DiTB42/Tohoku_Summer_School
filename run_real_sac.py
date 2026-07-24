"""Run SAC + CPG on the real snake robot using RealSense color tracking.

Color convention:
    RED   = front of the snake head
    BLUE  = rear of the snake head
    GREEN = current waypoint

Safety strategy:
    - R, omega and theta remain fixed at values already tested on the robot.
    - SAC controls only delta.
    - SAC is updated at 2 Hz instead of every camera frame.
    - Brief tracking losses keep the last command.
    - A prolonged tracking loss terminates the test and returns gradually to neutral.
    - Motion starts only after several consecutive valid camera frames.

Supported SAC observation dimensions:
    37 = observation without next_turn
    38 = observation with next_turn
"""

from __future__ import annotations

import argparse
import math
import socket
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from stable_baselines3 import SAC

from real_gait_controller import RealGaitController

from camera_tracking import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FRAME_RATE,
    MIN_ROBOT_AREA,
    MIN_WAYPOINT_AREA,
    get_color_masks,
    find_largest_blob,
    draw_detection,
    pixel_to_3d,
)


# ============================================================
# ROBOT NETWORK
# ============================================================

ROBOT_IP = "10.240.77.197"
ROBOT_PORT = 5000

N_JOINTS = 12
CONTROL_PERIOD = 0.12

WAVE_DIRECTION = 1

HARD_LIMIT = np.pi / 9.2
SAFE_LIMIT = 0.40


# ============================================================
# MODEL
# ============================================================

DEFAULT_MODEL_PATH = Path("models") / "sac_snake_final.zip"

ACTION_NAMES = (
    "R",
    "omega",
    "theta",
    "delta",
)


# ============================================================
# PARAMETERS ALREADY TESTED ON THE REAL ROBOT
# ============================================================

INITIAL_PARAMETERS = np.array(
    [
        0.30,          # R
        2.0,           # omega
        np.pi / 6.0,   # theta
        0.018,         # delta
    ],
    dtype=np.float64,
)


# Intervalli prudenti per il robot reale.
R_MIN = 0.20
R_MAX = 0.40

OMEGA_MIN = 1.70
OMEGA_MAX = 2.30

THETA_MIN = 0.45
THETA_MAX = 0.60

DELTA_MIN = -0.025
DELTA_MAX = 0.035


# Massima variazione per ogni ciclo di controllo.
# Sono volutamente piccoli per evitare spasmi.
MAX_PARAMETER_CHANGE = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.002,
    ],
    dtype=np.float64,
)




# Initial amplitude ramp.
RAMP_DURATION = 3.0


# ============================================================
# SAC AND TRACKING FREQUENCIES
# ============================================================

# Update SAC every 1.4 seconds bro.
# The CPG still runs every CONTROL_PERIOD.
SAC_UPDATE_PERIOD = 1.4

# Motion begins only after this many consecutive valid frames.
MIN_STABLE_TRACKING_FRAMES = 10

# Brief losses do not send zeros.
# If tracking remains absent longer than this, the program stops safely.
TRACKING_GRACE_PERIOD = 8.0

# Camera position smoothing:
# 0 = never update, 1 = no smoothing.
ROBOT_POSITION_FILTER_ALPHA = 0.70
WAYPOINT_POSITION_FILTER_ALPHA = 0.15

ROBOT_DEPTH_FILTER_ALPHA = 0.35
WAYPOINT_DEPTH_FILTER_ALPHA = 0.12


# ============================================================
# CAMERA AND WAYPOINT
# ============================================================

WINDOW_NAME = "Real snake SAC navigation"

# Distance between front marker and waypoint in metres.
WAYPOINT_THRESHOLD_M = 0.15

WP1_OCCLUSION_SWITCH_M = 0.22

PIXELS_PER_WORLD_UNIT = 100.0

# MuJoCo coordinates are assumed to use metres.
METERS_TO_WORLD_UNITS = 1.0


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RealSense color tracking -> SAC -> CPG -> real snake robot"
        )
    )

    parser.add_argument(
        "model_path",
        nargs="?",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the SAC model ZIP file",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Maximum test duration in seconds",
    )

    parser.add_argument(
        "--enable-motion",
        action="store_true",
        help="Actually send commands to the real robot",
    )

    return parser.parse_args()


# ============================================================
# GENERAL UTILITIES
# ============================================================

def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""

    return (
        angle + math.pi
    ) % (
        2.0 * math.pi
    ) - math.pi


def exponential_filter(
    previous_value: np.ndarray | None,
    new_value: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Simple exponential moving average."""

    new_value = np.asarray(
        new_value,
        dtype=np.float64,
    )

    if previous_value is None:
        return new_value.copy()

    previous_value = np.asarray(
        previous_value,
        dtype=np.float64,
    )

    return (
        alpha * new_value
        + (1.0 - alpha) * previous_value
    )


def format_parameters(values: np.ndarray) -> str:
    return "  ".join(
        f"{name}={value:+.4f}"
        for name, value in zip(
            ACTION_NAMES,
            values,
        )
    )


# ============================================================
# UDP ROBOT COMMANDS
# ============================================================

def send_angles(
    sock: socket.socket,
    angles: np.ndarray,
) -> None:
    angles = np.asarray(
        angles,
        dtype=np.float64,
    ).reshape(-1)

    if angles.shape != (N_JOINTS,):
        raise ValueError(
            f"Expected {N_JOINTS} angles, received {angles.shape}"
        )

    if not np.all(np.isfinite(angles)):
        raise ValueError(
            f"Invalid joint command: {angles}"
        )

    angles = np.clip(
        angles,
        -SAFE_LIMIT,
        SAFE_LIMIT,
    )

    angles = np.clip(
        angles,
        -HARD_LIMIT,
        HARD_LIMIT,
    )

    message = ",".join(
        f"{angle:.6f}"
        for angle in angles
    )

    sock.sendto(
        message.encode("utf-8"),
        (ROBOT_IP, ROBOT_PORT),
    )


def send_neutral(
    sock: socket.socket,
    repetitions: int = 5,
) -> None:
    neutral = np.zeros(
        N_JOINTS,
        dtype=np.float64,
    )

    for _ in range(repetitions):
        send_angles(
            sock,
            neutral,
        )

        time.sleep(
            CONTROL_PERIOD
        )


def return_slowly_to_neutral(
    sock: socket.socket,
    current_angles: np.ndarray,
    duration: float = 2.0,
) -> None:
    current_angles = np.asarray(
        current_angles,
        dtype=np.float64,
    ).reshape(N_JOINTS)

    steps = max(
        1,
        int(duration / CONTROL_PERIOD),
    )

    for alpha in np.linspace(
        0.0,
        1.0,
        steps,
    ):
        command = (
            1.0 - alpha
        ) * current_angles

        send_angles(
            sock,
            command,
        )

        time.sleep(
            CONTROL_PERIOD
        )


# ============================================================
# SAC ACTION PROCESSING
# ============================================================

def validate_sac_action(
    action: np.ndarray,
) -> np.ndarray:
    action = np.asarray(
        action,
        dtype=np.float64,
    ).reshape(-1)

    if action.shape != (4,):
        raise RuntimeError(
            "SAC must produce [R, omega, theta, delta]. "
            f"Received shape: {action.shape}"
        )

    if not np.all(np.isfinite(action)):
        raise RuntimeError(
            f"Invalid SAC action: {action}"
        )

    return action


def clamp_sac_action(
    action: np.ndarray,
) -> np.ndarray:
    """Limita tutti e quattro i parametri a intervalli prudenti."""

    action = validate_sac_action(
        action
    )

    return np.array(
        [
            np.clip(action[0], R_MIN, R_MAX),
            np.clip(action[1], OMEGA_MIN, OMEGA_MAX),
            np.clip(action[2], THETA_MIN, THETA_MAX),
            np.clip(action[3], DELTA_MIN, DELTA_MAX),
        ],
        dtype=np.float64,
    )


def update_parameters_slowly(
    current: np.ndarray,
    requested: np.ndarray,
) -> np.ndarray:
    """Rate limit indipendente sui quattro parametri."""

    current = np.asarray(
        current,
        dtype=np.float64,
    ).reshape(4)

    requested = np.asarray(
        requested,
        dtype=np.float64,
    ).reshape(4)

    difference = np.clip(
        requested - current,
        -MAX_PARAMETER_CHANGE,
        MAX_PARAMETER_CHANGE,
    )

    return current + difference

# ============================================================
# REAL OBSERVATION
# ============================================================

def build_real_observation(
    model: SAC,
    front_center: np.ndarray,
    rear_center: np.ndarray,
    waypoint_center: np.ndarray,
    front_xyz: np.ndarray,
    waypoint_xyz: np.ndarray,
    joint_positions: np.ndarray,
    last_action: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """
    Build the SAC observation using the same pixel-based coordinate
    convention that was used before depth was introduced.

    Depth is used only to compute the real metric distance.
    """

    front_center = np.asarray(
        front_center,
        dtype=np.float64,
    ).reshape(2)

    rear_center = np.asarray(
        rear_center,
        dtype=np.float64,
    ).reshape(2)

    waypoint_center = np.asarray(
        waypoint_center,
        dtype=np.float64,
    ).reshape(2)

    front_xyz = np.asarray(
        front_xyz,
        dtype=np.float64,
    ).reshape(3)

    waypoint_xyz = np.asarray(
        waypoint_xyz,
        dtype=np.float64,
    ).reshape(3)

    joint_positions = np.asarray(
        joint_positions,
        dtype=np.float64,
    ).reshape(N_JOINTS)

    last_action = np.asarray(
        last_action,
        dtype=np.float64,
    ).reshape(4)

    # --------------------------------------------------------
    # Joint state
    # --------------------------------------------------------

    joint_pos = joint_positions.copy()

    joint_vel = np.zeros(
        N_JOINTS,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Pixel geometry, identical to the pre-depth version
    # --------------------------------------------------------

    target_vector_px = (
        waypoint_center
        - front_center
    )

    # Image Y grows downward.
    # MuJoCo/world-style Y grows upward.
    head_to_target_xy = np.array(
        [
            float(target_vector_px[0]),
            -float(target_vector_px[1]),
        ],
        dtype=np.float64,
    ) / PIXELS_PER_WORLD_UNIT

    MAX_TARGET_DISTANCE = 3.0

    target_norm = float(
        np.linalg.norm(head_to_target_xy)
    )

    if target_norm > MAX_TARGET_DISTANCE:
        head_to_target_xy = (
            head_to_target_xy
            / target_norm
            * MAX_TARGET_DISTANCE
        )

    head_vector_px = (
        front_center
        - rear_center
    )

    head_norm = float(
        np.linalg.norm(head_vector_px)
    )

    if head_norm <= 1e-6:
        raise RuntimeError(
            "Front and rear markers are too close"
        )

    yaw = math.atan2(
        -float(head_vector_px[1]),
        float(head_vector_px[0]),
    )

    yaw = wrap_angle(yaw)

    target_angle = math.atan2(
        -float(target_vector_px[1]),
        float(target_vector_px[0]),
    )

    heading_error_angle = wrap_angle(
        target_angle - yaw
    )

    orient_z_angle = np.array(
        [
            1.0,
            yaw,
        ],
        dtype=np.float64,
    )

    head_angular_velocity = np.zeros(
        3,
        dtype=np.float64,
    )

    heading_error = np.array(
        [
            math.cos(heading_error_angle),
            math.sin(heading_error_angle),
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Depth only for metric waypoint distance
    # --------------------------------------------------------

    distance_m = float(
        np.linalg.norm(
            waypoint_xyz - front_xyz
        )
    )

    # --------------------------------------------------------
    # Observation
    # --------------------------------------------------------

    expected_shape = model.observation_space.shape

    if expected_shape == (37,):
        observation = np.concatenate(
            [
                joint_pos,
                joint_vel,
                head_to_target_xy,
                orient_z_angle,
                head_angular_velocity,
                heading_error,
                last_action,
            ]
        )

    elif expected_shape == (38,):
        next_turn = np.array(
            [0.0],
            dtype=np.float64,
        )

        observation = np.concatenate(
            [
                joint_pos,
                joint_vel,
                head_to_target_xy,
                orient_z_angle,
                head_angular_velocity,
                heading_error,
                next_turn,
                last_action,
            ]
        )

    else:
        raise RuntimeError(
            "Unsupported observation shape: "
            f"{expected_shape}. Expected (37,) or (38,)."
        )

    observation = observation.astype(
        np.float32
    )

    observation = np.clip(
        observation,
        -1000.0,
        1000.0,
    )

    if observation.shape != expected_shape:
        raise RuntimeError(
            f"Built observation {observation.shape}, "
            f"but model expects {expected_shape}"
        )

    if not np.all(np.isfinite(observation)):
        raise RuntimeError(
            "Observation contains invalid values"
        )

    return (
        observation,
        yaw,
        heading_error_angle,
        distance_m,
    )
    """
    Build the SAC observation from RealSense metric coordinates.

    RealSense coordinates:
        X = right
        Y = down
        Z = distance from camera

    The camera X-Y plane is used for heading.
    Full 3D coordinates are used for waypoint distance.
    """

    front_xyz = np.asarray(
        front_xyz,
        dtype=np.float64,
    ).reshape(3)

    rear_xyz = np.asarray(
        rear_xyz,
        dtype=np.float64,
    ).reshape(3)

    waypoint_xyz = np.asarray(
        waypoint_xyz,
        dtype=np.float64,
    ).reshape(3)

    joint_positions = np.asarray(
        joint_positions,
        dtype=np.float64,
    ).reshape(N_JOINTS)

    last_action = np.asarray(
        last_action,
        dtype=np.float64,
    ).reshape(4)

    # --------------------------------------------------------
    # Joint state
    # --------------------------------------------------------

    joint_pos = joint_positions.copy()

    joint_vel = np.zeros(
        N_JOINTS,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # RealSense metric geometry
    # --------------------------------------------------------

    head_vector_3d = (
        front_xyz
        - rear_xyz
    )

    target_vector_3d = (
        waypoint_xyz
        - front_xyz
    )

    # X-Y are the image-plane metric coordinates returned
    # by the RealSense deprojection.
    head_vector_xy = head_vector_3d[:2]
    target_vector_xy = target_vector_3d[:2]

    head_norm = float(
        np.linalg.norm(head_vector_xy)
    )

    target_xy_norm = float(
        np.linalg.norm(target_vector_xy)
    )

    if head_norm <= 1e-6:
        raise RuntimeError(
            "Front and rear 3D markers are too close"
        )

    if target_xy_norm <= 1e-6:
        heading_error_angle = 0.0
    else:
        yaw = math.atan2(
            float(head_vector_xy[1]),
            float(head_vector_xy[0]),
        )

        target_angle = math.atan2(
            float(target_vector_xy[1]),
            float(target_vector_xy[0]),
        )

        heading_error_angle = wrap_angle(
            target_angle - yaw
        )

    yaw = math.atan2(
        float(head_vector_xy[1]),
        float(head_vector_xy[0]),
    )

    yaw = wrap_angle(yaw)

    # Full metric 3D distance.
    distance_m = float(
        np.linalg.norm(target_vector_3d)
    )

    head_to_target_xy = (
        target_vector_xy
        * METERS_TO_WORLD_UNITS
    )

    MAX_TARGET_DISTANCE = 3.0

    target_norm = float(
        np.linalg.norm(head_to_target_xy)
    )

    if target_norm > MAX_TARGET_DISTANCE:
        head_to_target_xy = (
            head_to_target_xy
            / target_norm
            * MAX_TARGET_DISTANCE
        )

    orient_z_angle = np.array(
        [
            1.0,
            yaw,
        ],
        dtype=np.float64,
    )

    head_angular_velocity = np.zeros(
        3,
        dtype=np.float64,
    )

    heading_error = np.array(
        [
            math.cos(heading_error_angle),
            math.sin(heading_error_angle),
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Observation dimension
    # --------------------------------------------------------

    expected_shape = model.observation_space.shape

    if expected_shape == (37,):
        observation = np.concatenate(
            [
                joint_pos,
                joint_vel,
                head_to_target_xy,
                orient_z_angle,
                head_angular_velocity,
                heading_error,
                last_action,
            ]
        )

    elif expected_shape == (38,):
        next_turn = np.array(
            [0.0],
            dtype=np.float64,
        )

        observation = np.concatenate(
            [
                joint_pos,
                joint_vel,
                head_to_target_xy,
                orient_z_angle,
                head_angular_velocity,
                heading_error,
                next_turn,
                last_action,
            ]
        )

    else:
        raise RuntimeError(
            "Unsupported observation shape: "
            f"{expected_shape}. Expected (37,) or (38,)."
        )

    observation = observation.astype(
        np.float32
    )

    observation = np.clip(
        observation,
        -1000.0,
        1000.0,
    )

    if observation.shape != expected_shape:
        raise RuntimeError(
            f"Built observation {observation.shape}, "
            f"but model expects {expected_shape}"
        )

    if not np.all(np.isfinite(observation)):
        raise RuntimeError(
            "Observation contains invalid values"
        )

    return (
        observation,
        yaw,
        heading_error_angle,
        distance_m,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    if args.duration <= 0.0:
        raise ValueError(
            "--duration must be greater than zero"
        )

    model_path = Path(
        args.model_path
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"SAC model not found: {model_path.resolve()}"
        )


    print(
        f"Loading SAC model: {model_path.resolve()}"
    )

    model = SAC.load(
        str(model_path)
    )

    if model.action_space.shape != (4,):
        raise RuntimeError(
            "Model action shape must be (4,), received "
            f"{model.action_space.shape}"
        )

    if model.observation_space.shape not in (
        (37,),
        (38,),
    ):
        raise RuntimeError(
            "Unsupported model observation shape: "
            f"{model.observation_space.shape}"
        )


    print(
        f"Observation shape: {model.observation_space.shape}"
    )

    print(
        f"Action shape: {model.action_space.shape}"
    )


    # --------------------------------------------------------
    # RealSense configuration
    # --------------------------------------------------------

    pipeline = rs.pipeline()
    camera_config = rs.config()

    camera_config.enable_stream(
        rs.stream.color,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        rs.format.bgr8,
        FRAME_RATE,
    )
    camera_config.enable_stream(
        rs.stream.depth,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        rs.format.z16,
        FRAME_RATE,
    )

    pipeline_started = False
    align_to_color = rs.align(
        rs.stream.color
    )


    # --------------------------------------------------------
    # CPG controller
    # --------------------------------------------------------

    controller = RealGaitController(
        n_joints=N_JOINTS,
        wave_direction=WAVE_DIRECTION,
    )

    controller.set_parameters(
        R=0.0,
        omega=INITIAL_PARAMETERS[1],
        theta=INITIAL_PARAMETERS[2],
        delta=INITIAL_PARAMETERS[3],
    )

    current_parameters = INITIAL_PARAMETERS.copy()
    requested_parameters = INITIAL_PARAMETERS.copy()

    last_action = (
        INITIAL_PARAMETERS.copy()
    )

    last_command = np.zeros(
        N_JOINTS,
        dtype=np.float64,
    )


    # --------------------------------------------------------
    # Tracking state
    # --------------------------------------------------------

    filtered_front: np.ndarray | None = None
    filtered_rear: np.ndarray | None = None
    filtered_waypoint1: np.ndarray | None = None
    filtered_waypoint2: np.ndarray | None = None

    filtered_front_xyz: np.ndarray | None = None
    filtered_rear_xyz: np.ndarray | None = None
    filtered_waypoint1_xyz: np.ndarray | None = None
    filtered_waypoint2_xyz: np.ndarray | None = None

    # 0 = pink waypoint 1
    # 1 = light-yellow waypoint 2
    current_waypoint_index = 0

    consecutive_valid_frames = 0
    motion_started = False
    motion_start_time: float | None = None

    last_valid_tracking_time: float | None = None
    last_sac_update_time = -float("inf")
    last_distance_to_active_waypoint: float | None = None


    # --------------------------------------------------------
    # UDP socket
    # --------------------------------------------------------

    sock: socket.socket | None = None

    if args.enable_motion:
        sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        print(
            "REAL MOTION ENABLED"
        )

    else:
        print(
            "DRY RUN: no commands will be sent to the robot"
        )


    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )


    program_start_time = time.monotonic()

    try:
        pipeline.start(
            camera_config
        )

        pipeline_started = True


        if sock is not None:
            print(
                "Sending initial neutral position..."
            )

            send_neutral(
                sock,
                repetitions=8,
            )


        print(
            "Camera started."
        )

        print(
            "BLUE = front | GREEN/YELLOW variable = rear | "
            "PINK = waypoint 1 | LIGHT YELLOW = waypoint 2"
        )

        print(
            f"Waiting for {MIN_STABLE_TRACKING_FRAMES} "
            "consecutive valid frames before motion."
        )

        print(
            "Q or ESC = stop"
        )


        while True:
            iteration_start = time.monotonic()

            elapsed = (
                iteration_start
                - program_start_time
            )

            if elapsed >= args.duration:
                print(
                    "\nMaximum duration reached."
                )
                break


            # ------------------------------------------------
            # Camera acquisition
            # ------------------------------------------------

            frames = pipeline.wait_for_frames()

            aligned_frames = align_to_color.process(
                frames
            )

            color_frame = (
                aligned_frames.get_color_frame()
            )

            depth_frame = (
                aligned_frames.get_depth_frame()
            )

            if not color_frame or not depth_frame:
                continue


            image = np.asanyarray(
                color_frame.get_data()
            ).copy()

            hsv = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2HSV,
            )

            (
                front_mask,
                rear_mask,
                waypoint1_mask,
                waypoint2_mask,
            ) = get_color_masks(hsv)


            
            raw_front, front_contour, _ = find_largest_blob(
                front_mask,
                MIN_ROBOT_AREA,
            )

            raw_rear, rear_contour, _ = find_largest_blob(
                rear_mask,
                MIN_ROBOT_AREA,
            )

            raw_waypoint1, waypoint1_contour, _ = find_largest_blob(
                waypoint1_mask,
                MIN_WAYPOINT_AREA,
            )

            raw_waypoint2, waypoint2_contour, _ = find_largest_blob(
                waypoint2_mask,
                MIN_WAYPOINT_AREA,
            )

            raw_front_xyz = pixel_to_3d(
                depth_frame,
                raw_front,
            )

            raw_rear_xyz = pixel_to_3d(
                depth_frame,
                raw_rear,
            )

            raw_waypoint1_xyz = pixel_to_3d(
                depth_frame,
                raw_waypoint1,
            )

            raw_waypoint2_xyz = pixel_to_3d(
                depth_frame,
                raw_waypoint2,
            )

            if current_waypoint_index == 0:
                all_markers_visible = (
                    raw_front is not None
                    and raw_rear is not None
                    and raw_waypoint1 is not None
                    and raw_front_xyz is not None
                    and raw_rear_xyz is not None
                    and raw_waypoint1_xyz is not None
                )

            else:
                all_markers_visible = (
                    raw_front is not None
                    and raw_rear is not None
                    and raw_waypoint2 is not None
                    and raw_front_xyz is not None
                    and raw_rear_xyz is not None
                    and raw_waypoint2_xyz is not None
                )


            # ------------------------------------------------
            # Valid tracking
            # ------------------------------------------------

            if all_markers_visible:
                consecutive_valid_frames += 1
                last_valid_tracking_time = iteration_start


                filtered_front = exponential_filter(
                    filtered_front,
                    raw_front,
                    ROBOT_POSITION_FILTER_ALPHA,
                )

                filtered_rear = exponential_filter(
                    filtered_rear,
                    raw_rear,
                    ROBOT_POSITION_FILTER_ALPHA,
                )

                

                if raw_waypoint1 is not None:
                    filtered_waypoint1 = exponential_filter(
                        filtered_waypoint1,
                        raw_waypoint1,
                        WAYPOINT_POSITION_FILTER_ALPHA,
                    )

                if raw_waypoint2 is not None:
                    filtered_waypoint2 = exponential_filter(
                        filtered_waypoint2,
                        raw_waypoint2,
                        WAYPOINT_POSITION_FILTER_ALPHA,
                    )
                    

                filtered_front_xyz = exponential_filter(
                    filtered_front_xyz,
                    raw_front_xyz,
                    ROBOT_DEPTH_FILTER_ALPHA,
                )

                filtered_rear_xyz = exponential_filter(
                    filtered_rear_xyz,
                    raw_rear_xyz,
                    ROBOT_DEPTH_FILTER_ALPHA,
                )

                if raw_waypoint1_xyz is not None:
                    filtered_waypoint1_xyz = exponential_filter(
                        filtered_waypoint1_xyz,
                        raw_waypoint1_xyz,
                        WAYPOINT_DEPTH_FILTER_ALPHA,
                    )

                if raw_waypoint2_xyz is not None:
                    filtered_waypoint2_xyz = exponential_filter(
                        filtered_waypoint2_xyz,
                        raw_waypoint2_xyz,
                        WAYPOINT_DEPTH_FILTER_ALPHA,
                    )

                if (
                    front_contour is not None
                    and filtered_front is not None
                ):
                    draw_detection(
                        image,
                        filtered_front,
                        front_contour,
                        (0, 0, 255),
                        "FRONT",
                    )


                if (
                    rear_contour is not None
                    and filtered_rear is not None
                ):
                    draw_detection(
                        image,
                        filtered_rear,
                        rear_contour,
                        (255, 0, 0),
                        "REAR",
                    )


                if (
                    waypoint1_contour is not None
                    and filtered_waypoint1 is not None
                ):
                    draw_detection(
                        image,
                        filtered_waypoint1,
                        waypoint1_contour,
                        (255, 0, 255),
                        "WP1 PINK",
                    )

                if (
                    waypoint2_contour is not None
                    and filtered_waypoint2 is not None
                ):
                    draw_detection(
                        image,
                        filtered_waypoint2,
                        waypoint2_contour,
                        (0, 255, 255),
                        "WP2 LIGHT YELLOW",
                    )


                if (
                    not motion_started
                    and consecutive_valid_frames
                    >= MIN_STABLE_TRACKING_FRAMES
                ):
                    motion_started = True
                    motion_start_time = iteration_start

                    print(
                        "\nTracking stable. Motion control started."
                    )


                active_target_available = (
                    current_waypoint_index == 0
                    and filtered_waypoint1 is not None
                    and filtered_waypoint1_xyz is not None
                ) or (
                    current_waypoint_index == 1
                    and filtered_waypoint2 is not None
                    and filtered_waypoint2_xyz is not None
                )

                if (
                    motion_started
                    and filtered_front is not None
                    and filtered_rear is not None
                    and filtered_front_xyz is not None
                    and filtered_rear_xyz is not None
                    and active_target_available
                ):
                    if current_waypoint_index == 0:
                        current_target = filtered_waypoint1
                        current_target_xyz = filtered_waypoint1_xyz
                        current_target_name = "WP1"
                    else:
                        current_target = filtered_waypoint2
                        current_target_xyz = filtered_waypoint2_xyz
                        current_target_name = "WP2"
                    
                    

                    (
                        observation,
                        yaw,
                        heading_error,
                        distance_m,
                    ) = build_real_observation(
                        model=model,
                        front_center=filtered_front,
                        rear_center=filtered_rear,
                        waypoint_center=current_target,
                        front_xyz=filtered_front_xyz,
                        waypoint_xyz=current_target_xyz,
                        joint_positions=last_command,
                        last_action=last_action,
                    )

                    last_distance_to_active_waypoint = distance_m
                    distance_px = float(
                        np.linalg.norm(
                            current_target - filtered_front
                        )
                    )

                    head_length_px = float(
                        np.linalg.norm(filtered_front - filtered_rear)
                    )

                    MIN_HEAD_LENGTH_PX = 35.0
                    MAX_HEAD_LENGTH_PX = 220.0

                    if not (
                        MIN_HEAD_LENGTH_PX
                        <= head_length_px
                        <= MAX_HEAD_LENGTH_PX
                    ):
                        print(
                            f"\nInvalid front/rear geometry: "
                            f"{head_length_px:.1f}px - holding last command"
                        )

                        if sock is not None:
                            send_angles(sock, last_command)

                        continue
                    # ----------------------------------------
                    # Update SAC every 1.4 seconds.
                    # ----------------------------------------

                    if (
                        iteration_start
                        - last_sac_update_time
                        >= SAC_UPDATE_PERIOD
                    ):
                        raw_action, _ = model.predict(
                            observation,
                            deterministic=True,
                        )

                        requested_parameters = clamp_sac_action(
                            raw_action
                        )

                      

                        last_sac_update_time = iteration_start

                        print(
                            f"heading={math.degrees(heading_error):+.1f} deg | "
                            f"raw_delta={raw_action[3]:+.4f} | "
                            f"requested_delta={requested_parameters[3]:+.4f} | "
                            f"current_delta={current_parameters[3]:+.4f}"
                        )


                    current_parameters = update_parameters_slowly(
                        current_parameters,
                        requested_parameters,
                    )

                    
                    # ----------------------------------------
                    # Waypoint reached by RED marker
                    # ----------------------------------------

                    WAYPOINT_THRESHOLD_PX_REAL = 55.0

                    waypoint_reached = (
                        distance_px <= WAYPOINT_THRESHOLD_PX_REAL
                        or distance_m <= WAYPOINT_THRESHOLD_M
                    )

                    if waypoint_reached:

                        if current_waypoint_index == 0:
                            current_waypoint_index = 1

                            cv2.putText(
                                image,
                                "WP1 REACHED -> SWITCHING TO WP2",
                                (15, 90),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.70,
                                (0, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )

                            cv2.imshow(
                                WINDOW_NAME,
                                image,
                            )

                            cv2.waitKey(
                                300
                            )

                            print(
                                "\nWaypoint 1 reached. "
                                "Switching to light-yellow waypoint 2."
                            )

                            # Force SAC to immediately recompute the action
                            # using waypoint 2 on the next iteration.
                            last_sac_update_time = -float("inf")

                            # Skip the rest of this iteration.
                            continue

                        else:
                            cv2.putText(
                                image,
                                "FINAL WAYPOINT REACHED",
                                (15, 90),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA,
                            )

                            cv2.imshow(
                                WINDOW_NAME,
                                image,
                            )

                            cv2.waitKey(
                                300
                            )

                            print(
                                "\nWaypoint 2 reached. Navigation completed."
                            )

                            break


                    # ----------------------------------------
                    # CPG update
                    # ----------------------------------------

                    R, omega, theta, delta = current_parameters

                    if motion_start_time is None:
                        motion_elapsed = 0.0
                    else:
                        motion_elapsed = (
                            iteration_start
                            - motion_start_time
                        )

                    ramp = min(
                        motion_elapsed / RAMP_DURATION,
                        1.0,
                    )

                    commanded_R = R * ramp

                    controller.set_parameters(
                        R=commanded_R,
                        omega=omega,
                        theta=theta,
                        delta=delta,
                    )


                    target_q = controller.update(
                        CONTROL_PERIOD
                    )

                    target_q = np.clip(
                        target_q,
                        -SAFE_LIMIT,
                        SAFE_LIMIT,
                    )


                    if sock is not None:
                        send_angles(
                            sock,
                            target_q,
                        )


                    last_command = (
                        target_q.copy()
                    )

                    last_action = (
                        current_parameters.copy()
                    )


                    # ----------------------------------------
                    # Display
                    # ----------------------------------------

                    front_point = tuple(
                        np.round(
                            filtered_front
                        ).astype(int)
                    )

                    rear_point = tuple(
                        np.round(
                            filtered_rear
                        ).astype(int)
                    )

                    waypoint_point = tuple(
                        np.round(
                            current_target
                        ).astype(int)
                    )


                    cv2.arrowedLine(
                        image,
                        rear_point,
                        front_point,
                        (0, 255, 255),
                        3,
                        tipLength=0.25,
                    )


                    cv2.line(
                        image,
                        front_point,
                        waypoint_point,
                        (255, 255, 0),
                        2,
                    )


                    cv2.putText(
                        image,
                        (
                            f"target={current_target_name}  "
                            f"distance={distance_m:.3f}m  "
                            f"error={math.degrees(heading_error):+.1f}deg"
                        ),
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )


                    cv2.putText(
                        image,
                        format_parameters(
                            current_parameters
                        ),
                        (15, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )


                    print(
                        "\r"
                        f"t={elapsed:6.2f}s  "
                        f"target={current_target_name}  "
                        f"distance={distance_m:6.3f}m  "
                        f"error={math.degrees(heading_error):+7.2f}deg  "
                        f"{format_parameters(current_parameters)}",
                        end="",
                        flush=True,
                    )


            # ------------------------------------------------
            # Missing marker
            # ------------------------------------------------

            else:
                consecutive_valid_frames = 0

                if (
                    current_waypoint_index == 0
                    and motion_started
                    and last_distance_to_active_waypoint is not None
                    and last_distance_to_active_waypoint
                    <= WP1_OCCLUSION_SWITCH_M
                    and (
                        raw_waypoint1 is None
                        or raw_waypoint1_xyz is None
                    )
                ):
                    current_waypoint_index = 1
                    last_sac_update_time = -float("inf")
                    last_distance_to_active_waypoint = None

                    print(
                        "\nWP1 disappeared while already close. "
                        "Assuming WP1 reached and switching to WP2."
                    )

                    continue

                missing = []

                if raw_front is None:
                    missing.append("BLUE FRONT")

                if raw_rear is None:
                    missing.append("GREEN REAR")

                if current_waypoint_index == 0:
                    if raw_waypoint1 is None:
                        missing.append("PINK WP1")

                else:
                    if raw_waypoint2 is None:
                        missing.append("YELLOW WP2")

                # Marker detected in RGB but without a valid depth value.

                if raw_front is not None and raw_front_xyz is None:
                    missing.append("FRONT DEPTH")

                if raw_rear is not None and raw_rear_xyz is None:
                    missing.append("REAR DEPTH")

                if (
                    current_waypoint_index == 0
                    and raw_waypoint1 is not None
                    and raw_waypoint1_xyz is None
                ):
                    missing.append("WP1 DEPTH")

                if (
                    current_waypoint_index == 1
                    and raw_waypoint2 is not None
                    and raw_waypoint2_xyz is None
                ):
                    missing.append("WP2 DEPTH")

                cv2.putText(
                    image,
                    "Missing: " + ", ".join(missing),
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )


                if last_valid_tracking_time is None:
                    tracking_loss_duration = float(
                        "inf"
                    )
                else:
                    tracking_loss_duration = (
                        iteration_start
                        - last_valid_tracking_time
                    )


                if not motion_started:
                    cv2.putText(
                        image,
                        "WAITING FOR STABLE TRACKING",
                        (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )


                elif (
                    tracking_loss_duration
                    <= TRACKING_GRACE_PERIOD
                ):
                    # Brief loss: preserve the last target.
                    # Never jump immediately to twelve zeros.
                    if sock is not None:
                        send_angles(
                            sock,
                            last_command,
                        )


                    cv2.putText(
                        image,
                        (
                            "TEMPORARY LOSS - "
                            "HOLDING LAST COMMAND"
                        ),
                        (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                
                    print(
                        "\rTemporary tracking loss: "
                        + ", ".join(missing)
                        + f" ({tracking_loss_duration:.2f}s)"
                        + " - holding last command",
                        end="",
                        flush=True,
                    )


                else:
                    print(
                        "\nTracking lost for too long: "
                        + ", ".join(missing)
                    )

                    print(
                        "Stopping and returning gradually to neutral."
                    )

                    break


            # ------------------------------------------------
            # Window and keyboard
            # ------------------------------------------------

            cv2.imshow(
                WINDOW_NAME,
                image,
            )

            key = cv2.waitKey(
                1
            ) & 0xFF

            if key in (
                ord("q"),
                27,
            ):
                print(
                    "\nStop requested."
                )

                break


            # Keep the joint command update close to CONTROL_PERIOD.
            loop_time = (
                time.monotonic()
                - iteration_start
            )

            remaining = (
                CONTROL_PERIOD
                - loop_time
            )

            if remaining > 0.0:
                time.sleep(
                    remaining
                )


    except KeyboardInterrupt:
        print(
            "\nKeyboard interrupt."
        )


    finally:
        if sock is not None:
            print(
                "Returning gradually to neutral..."
            )

            return_slowly_to_neutral(
                sock,
                last_command,
                duration=2.0,
            )

            send_neutral(
                sock,
                repetitions=5,
            )

            sock.close()

            print(
                "Robot stopped."
            )


        if pipeline_started:
            pipeline.stop()


        cv2.destroyAllWindows()

        print(
            "Camera stopped."
        )


if __name__ == "__main__":
    main()