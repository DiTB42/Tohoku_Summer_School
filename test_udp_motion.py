import socket
import time
import numpy as np

ROBOT_IP = "10.240.77.197"
ROBOT_PORT = 5000

sock = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM,
)

for step in range(100):
    phase = 0.15 * step

    angles = np.array(
        [
            0.15 * np.sin(phase + joint * 0.5)
            for joint in range(12)
        ],
        dtype=np.float64,
    )

    message = ",".join(
        f"{angle:.6f}"
        for angle in angles
    )

    sock.sendto(
        message.encode("utf-8"),
        (ROBOT_IP, ROBOT_PORT),
    )

    print(angles)

    time.sleep(0.12)

sock.close()