"""Track colored markers with an Intel RealSense D435.

Color convention:
    BLUE         = front marker
    GREEN        = rear marker
    PINK         = waypoint 1
    LIGHT YELLOW = waypoint 2
Controls:
    Q / ESC = quit
    R       = reset reached waypoint

This file only tracks the robot.
It does not send commands to the robot.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pyrealsense2 as rs


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_RATE = 30
# Depth range accepted for marker measurements.
MIN_VALID_DEPTH_M = 0.10
MAX_VALID_DEPTH_M = 4.00

# Radius of the square neighbourhood used around each centroid.
# 3 means a 7x7 window.
DEPTH_WINDOW_RADIUS = 3

WINDOW_NAME = "Snake color tracking"


# HSV thresholds.
# These may need small adjustments depending on lighting.

# Red needs two hue ranges because it wraps around 0.
# ============================================================
# NEW COLOR CONVENTION
#
# BLUE   = front marker
# YELLOW = rear marker
# PINK   = waypoint
# ============================================================

# Exact reference colors from the provided image:
#
# Blue:   RGB (41, 9, 231)   -> HSV OpenCV approximately (124, 245, 231)
# Yellow: RGB (250, 245, 43) -> HSV OpenCV approximately (29, 211, 250)
# Pink:   RGB (230, 64, 230) -> HSV OpenCV approximately (150, 184, 230)


# Soglie ricavate direttamente dall'immagine della RealSense.
#
# BLUE   = marker anteriore
# YELLOW = marker posteriore
# PINK   = waypoint

BLUE_LOW = np.array(
    [100, 90, 90],
    dtype=np.uint8,
)

BLUE_HIGH = np.array(
    [115, 255, 255],
    dtype=np.uint8,
)


# In realtà usa il vecchio VERDE
YELLOW_LOW = np.array(
    [40, 70, 70],
    dtype=np.uint8,
)

YELLOW_HIGH = np.array(
    [85, 255, 255],
    dtype=np.uint8,
)

PINK_LOW = np.array(
    [130, 55, 70],
    dtype=np.uint8,
)

PINK_HIGH = np.array(
    [173, 255, 255],
    dtype=np.uint8,
)

LIGHT_YELLOW_LOW = np.array(
    [18, 70, 170],
    dtype=np.uint8,
)

LIGHT_YELLOW_HIGH = np.array(
    [38, 255, 255],
    dtype=np.uint8,
)

MIN_ROBOT_AREA = 100.0
MIN_WAYPOINT_AREA = 180.0

WAYPOINT_THRESHOLD_PX = 40.0


reached_waypoint = False


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), dtype=np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return mask


def get_color_masks(
    hsv: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Returns:
        front_mask     -> blue front marker
        rear_mask      -> green/rear marker
        waypoint1_mask -> pink waypoint
        waypoint2_mask -> light-yellow waypoint
    """

    front_mask = cv2.inRange(
        hsv,
        BLUE_LOW,
        BLUE_HIGH,
    )

    rear_mask = cv2.inRange(
        hsv,
        YELLOW_LOW,
        YELLOW_HIGH,
    )

    waypoint_mask = cv2.inRange(
        hsv,
        PINK_LOW,
        PINK_HIGH,
    )

    waypoint2_mask = cv2.inRange(
        hsv,
        LIGHT_YELLOW_LOW,
        LIGHT_YELLOW_HIGH,
    )

    return (
        clean_mask(front_mask),
        clean_mask(rear_mask),
        clean_mask(waypoint_mask),
        clean_mask(waypoint2_mask),
    )

def contour_center(
    contour: np.ndarray,
) -> np.ndarray | None:
    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return None

    center_x = moments["m10"] / moments["m00"]
    center_y = moments["m01"] / moments["m00"]

    return np.array(
        [center_x, center_y],
        dtype=np.float32,
    )


def find_largest_blob(
    mask: np.ndarray,
    min_area: float,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    valid_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= min_area
    ]

    if not valid_contours:
        return None, None, 0.0

    contour = max(
        valid_contours,
        key=cv2.contourArea,
    )

    center = contour_center(contour)

    if center is None:
        return None, None, 0.0

    area = float(
        cv2.contourArea(contour)
    )

    return center, contour, area

def pixel_to_3d(
    depth_frame: rs.depth_frame,
    pixel_center: np.ndarray,
    window_radius: int = DEPTH_WINDOW_RADIUS,
) -> np.ndarray | None:
    """
    Convert a color-image centroid into a RealSense 3D point.

    Returns:
        np.array([X, Y, Z]) in metres.

    RealSense camera coordinates:
        X = right
        Y = down
        Z = forward from camera
    """

    if depth_frame is None or pixel_center is None:
        return None

    pixel_center = np.asarray(
        pixel_center,
        dtype=np.float64,
    ).reshape(2)

    center_x = int(round(float(pixel_center[0])))
    center_y = int(round(float(pixel_center[1])))

    frame_width = depth_frame.get_width()
    frame_height = depth_frame.get_height()

    valid_samples: list[
        tuple[float, int, int]
    ] = []

    for offset_y in range(
        -window_radius,
        window_radius + 1,
    ):
        for offset_x in range(
            -window_radius,
            window_radius + 1,
        ):
            pixel_x = center_x + offset_x
            pixel_y = center_y + offset_y

            if not (
                0 <= pixel_x < frame_width
                and 0 <= pixel_y < frame_height
            ):
                continue

            depth_m = float(
                depth_frame.get_distance(
                    pixel_x,
                    pixel_y,
                )
            )

            if (
                np.isfinite(depth_m)
                and MIN_VALID_DEPTH_M
                <= depth_m
                <= MAX_VALID_DEPTH_M
            ):
                valid_samples.append(
                    (
                        depth_m,
                        pixel_x,
                        pixel_y,
                    )
                )

    if not valid_samples:
        return None

    # Median is more robust than reading only the central pixel.
    median_depth = float(
        np.median(
            [
                sample[0]
                for sample in valid_samples
            ]
        )
    )

    # Use the original centroid with the robust median depth.
    intrinsics = (
        depth_frame.profile
        .as_video_stream_profile()
        .intrinsics
    )

    point_xyz = rs.rs2_deproject_pixel_to_point(
        intrinsics,
        [
            float(pixel_center[0]),
            float(pixel_center[1]),
        ],
        median_depth,
    )

    point_xyz = np.asarray(
        point_xyz,
        dtype=np.float64,
    )

    if (
        point_xyz.shape != (3,)
        or not np.all(np.isfinite(point_xyz))
    ):
        return None

    return point_xyz


def draw_detection(
    image: np.ndarray,
    center: np.ndarray,
    contour: np.ndarray,
    color: tuple[int, int, int],
    label: str,
) -> None:
    point = tuple(
        np.round(center).astype(int)
    )

    cv2.drawContours(
        image,
        [contour],
        -1,
        color,
        2,
    )

    cv2.circle(
        image,
        point,
        6,
        color,
        -1,
    )

    cv2.putText(
        image,
        label,
        (point[0] + 8, point[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def wrap_angle(angle: float) -> float:
    return (
        angle + math.pi
    ) % (
        2.0 * math.pi
    ) - math.pi


def main() -> None:
    global reached_waypoint

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        rs.format.bgr8,
        FRAME_RATE,
    )
    config.enable_stream(
        rs.stream.depth,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        rs.format.z16,
        FRAME_RATE,
    )

    pipeline.start(config)
    align_to_color = rs.align(
        rs.stream.color
    )

    print("Tracking colore avviato.")
    print("BLUE = front")
    print("GREEN = rear")
    print("PINK = waypoint 1")
    print("LIGHT YELLOW = waypoint 2")
    print("R = reset waypoint")
    print("Q / ESC = chiudi")

    try:
        while True:
            frames = pipeline.wait_for_frames()

            aligned_frames = align_to_color.process(
                frames
            )

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

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

            (
            front_center,
            front_contour,
            front_area,
            ) = find_largest_blob(
                front_mask,
                MIN_ROBOT_AREA,
            )

            (
                rear_center,
                rear_contour,
                rear_area,
                ) = find_largest_blob(
                    rear_mask,
                    MIN_ROBOT_AREA,
                )

            
            (
                waypoint1_center,
                waypoint1_contour,
                waypoint1_area,
            ) = find_largest_blob(
                waypoint1_mask,
                MIN_WAYPOINT_AREA,
            )

            (
                waypoint2_center,
                waypoint2_contour,
                waypoint2_area,
            ) = find_largest_blob(
                waypoint2_mask,
                MIN_WAYPOINT_AREA,
            )

            front_xyz = pixel_to_3d(
                depth_frame,
                front_center,
            )

            rear_xyz = pixel_to_3d(
                depth_frame,
                rear_center,
            )

            waypoint1_xyz = pixel_to_3d(
                depth_frame,
                waypoint1_center,
            )

            waypoint2_xyz = pixel_to_3d(
                depth_frame,
                waypoint2_center,
            )

            if (
                front_center is not None
                and front_contour is not None
            ):
                draw_detection(
                    image,
                    front_center,
                    front_contour,
                    (0, 0, 255),
                    "FRONT",
                )

            if (
                rear_center is not None
                and rear_contour is not None
            ):
                draw_detection(
                    image,
                    rear_center,
                    rear_contour,
                    (255, 0, 0),
                    "REAR",
                )

            if (
                waypoint1_center is not None
                and waypoint1_contour is not None
            ):
                draw_detection(
                    image,
                    waypoint1_center,
                    waypoint1_contour,
                    (255, 0, 255),
                    "WAYPOINT 1",
                )

            if (
                waypoint2_center is not None
                and waypoint2_contour is not None
            ):
                draw_detection(
                    image,
                    waypoint2_center,
                    waypoint2_contour,
                    (0, 255, 255),
                    "WAYPOINT 2",
                )

            if (
                front_center is not None
                and rear_center is not None
            ):
                front_point = tuple(
                    np.round(front_center).astype(int)
                )

                rear_point = tuple(
                    np.round(rear_center).astype(int)
                )

                cv2.arrowedLine(
                    image,
                    rear_point,
                    front_point,
                    (0, 255, 255),
                    3,
                    tipLength=0.25,
                )

                head_vector = (
                    front_center
                    - rear_center
                )

                yaw = math.atan2(
                    float(head_vector[1]),
                    float(head_vector[0]),
                )

                cv2.putText(
                    image,
                    f"yaw={math.degrees(yaw):+.1f} deg",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if (
                    waypoint1_center is not None
                    and not reached_waypoint
                ):
                    target_vector = (
                        waypoint1_center
                        - front_center
                    )

                    target_angle = math.atan2(
                        float(target_vector[1]),
                        float(target_vector[0]),
                    )

                    heading_error = wrap_angle(
                        target_angle - yaw
                    )

                    distance_px = float(
                        np.linalg.norm(
                            target_vector
                        )
                    )

                    waypoint_point = tuple(
                        np.round(
                            waypoint1_center
                        ).astype(int)
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
                            f"distance={distance_px:.1f}px  "
                            f"error={math.degrees(heading_error):+.1f} deg"
                        ),
                        (15, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    print(
                        "\r"
                        f"head=({front_center[0]:7.2f}, "
                        f"{front_center[1]:7.2f})  "
                        f"yaw={yaw:+.4f}  "
                        f"distance={distance_px:7.2f}px  "
                        f"error={heading_error:+.4f}",
                        end="",
                        flush=True,
                    )

                    if distance_px <= WAYPOINT_THRESHOLD_PX:
                        reached_waypoint = True
                        print("\nWaypoint raggiunto.")

                elif reached_waypoint:
                    cv2.putText(
                        image,
                        "WAYPOINT REACHED",
                        (15, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            else:
                missing = []

                if front_center is None:
                    missing.append("BLUE FRONT")

                if rear_center is None:
                    missing.append("YELLOW REAR")

                if waypoint1_center is None:
                    missing.append("PINK WAYPOINT 1")

                if waypoint2_center is None:
                    missing.append("LIGHT-YELLOW WAYPOINT 2")

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

            cv2.putText(
                image,
                (
                    f"blue={front_area:.0f}  "
                    f"rear={rear_area:.0f}  "
                    f"pink={waypoint1_area:.0f}  "
                    f"light-yellow={waypoint2_area:.0f}"
                ),
                (15, FRAME_HEIGHT - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            depth_text_lines = []

            if front_xyz is not None:
                depth_text_lines.append(
                    "front xyz="
                    f"({front_xyz[0]:+.3f}, "
                    f"{front_xyz[1]:+.3f}, "
                    f"{front_xyz[2]:+.3f}) m"
                )

            if rear_xyz is not None:
                depth_text_lines.append(
                    "rear xyz="
                    f"({rear_xyz[0]:+.3f}, "
                    f"{rear_xyz[1]:+.3f}, "
                    f"{rear_xyz[2]:+.3f}) m"
                )

            if waypoint1_xyz is not None:
                depth_text_lines.append(
                    "wp1 xyz="
                    f"({waypoint1_xyz[0]:+.3f}, "
                    f"{waypoint1_xyz[1]:+.3f}, "
                    f"{waypoint1_xyz[2]:+.3f}) m"
                )

            if waypoint2_xyz is not None:
                depth_text_lines.append(
                    "wp2 xyz="
                    f"({waypoint2_xyz[0]:+.3f}, "
                    f"{waypoint2_xyz[1]:+.3f}, "
                    f"{waypoint2_xyz[2]:+.3f}) m"
                )

            for line_index, line_text in enumerate(
                depth_text_lines
            ):
                cv2.putText(
                    image,
                    line_text,
                    (
                        15,
                        90 + line_index * 22,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(
                WINDOW_NAME,
                image,
            )

            cv2.imshow(
                "Front blue mask",
                front_mask,
            )

            cv2.imshow(
                "Rear yellow mask",
                rear_mask,
            )

            cv2.imshow(
                "Pink waypoint 1 mask",
                waypoint1_mask,
            )

            cv2.imshow(
                "Light-yellow waypoint 2 mask",
                waypoint2_mask,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (
                ord("q"),
                27,
            ):
                break

            if key == ord("r"):
                reached_waypoint = False
                print("\nWaypoint resettato.")

    finally:
        print()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Tracking terminato.")


if __name__ == "__main__":
    main()