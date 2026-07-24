import socket
import time
import numpy as np

from real_gait_controller import RealGaitController


ROBOT_IP = "10.240.77.197"
ROBOT_PORT = 5000

N_JOINTS = 12

# Periodo con cui il Raspberry Pi riceve i comandi.
CONTROL_PERIOD = 0.12

# Limite hardware indicato nel test originale.
HARD_LIMIT = np.pi / 9.2  # circa 0.341 rad

AMPLITUDE = 0.30
OMEGA = 2.0
THETA = np.pi / 6


TEST_DELTA = 0.018
TEST_DURATION = 40.0

SAFE_LIMIT = 0.50
# Intensità della sterzata.
# Partiamo con un offset piccolo.
#TURN_DELTA = -0.035

#STRAIGHT_DURATION = 12.0
#TRANSITION_DURATION = 4.0
#TURN_DURATION = 20.0

# Direzione di propagazione dell'onda.
# Cambia da 1 a -1 se l'onda viaggia nella direzione sbagliata.
WAVE_DIRECTION = 1

# Tempo in cui l'ampiezza cresce da zero al valore desiderato.
RAMP_DURATION = 3.0


def send_angles(
    sock: socket.socket,
    angles: np.ndarray,
) -> None:
    """Invia 12 angoli in radianti al robot via UDP."""
    angles = np.asarray(
        angles,
        dtype=np.float64,
    ).reshape(-1)

    if angles.shape != (N_JOINTS,):
        raise ValueError(
            f"Expected {N_JOINTS} joint angles, "
            f"got shape {angles.shape}"
        )

    if not np.all(np.isfinite(angles)):
        raise ValueError(
            f"Invalid joint angles: {angles}"
        )

    # Limite prudente.
    angles = np.clip(
        angles,
        -SAFE_LIMIT,
        SAFE_LIMIT,
    )

    # Limite hardware assoluto.
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
    repetitions: int = 10,
) -> None:
    """Invia ripetutamente la configurazione neutra."""
    neutral = np.zeros(
        N_JOINTS,
        dtype=np.float64,
    )

    for _ in range(repetitions):
        send_angles(sock, neutral)
        time.sleep(CONTROL_PERIOD)


def return_slowly_to_neutral(
    sock: socket.socket,
    current_angles: np.ndarray,
    duration: float = 2.0,
) -> None:
    """Riporta gradualmente tutti i giunti a zero."""
    current_angles = np.asarray(
        current_angles,
        dtype=np.float64,
    )

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

        time.sleep(CONTROL_PERIOD)

def linear_transition(
    elapsed: float,
    start_time: float,
    duration: float,
    start_value: float,
    end_value: float,
) -> float:
    """Interpola gradualmente un parametro tra due valori."""
    alpha = np.clip(
        (elapsed - start_time) / duration,
        0.0,
        1.0,
    )

    return (
        (1.0 - alpha) * start_value
        + alpha * end_value
    )

def main() -> None:
    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    controller = RealGaitController(
        n_joints=N_JOINTS,
        wave_direction=WAVE_DIRECTION,
    )

    # Partiamo da ampiezza zero.
    controller.set_parameters(
        R=0.0,
        omega=OMEGA,
        theta=THETA,
        delta=0.0,
    )

    last_command = np.zeros(
        N_JOINTS,
        dtype=np.float64,
    )

    print("Sending neutral position...")
    send_neutral(
        sock,
        repetitions=10,
    )

    print("Starting straight calibration test.")
    print(f"Target amplitude: {AMPLITUDE:.3f} rad")
    print(f"Omega: {OMEGA:.3f} rad/s")
    print(f"Theta: {THETA:.3f} rad")
    print(f"Test delta: {TEST_DELTA:+.3f} rad")
    print(f"Safe limit: ±{SAFE_LIMIT:.3f} rad")
    print("Press Ctrl+C to stop.")

    start_time = time.monotonic()

    try:
        while True:
            step_start = time.monotonic()
            elapsed = step_start - start_time

            amplitude_ramp = min(
                elapsed / RAMP_DURATION,
                1.0,
            )

            current_amplitude = (
                AMPLITUDE * amplitude_ramp
            )

            if elapsed >= TEST_DURATION:
                print("Straight calibration test completed.")
                break

            current_delta = TEST_DELTA

            controller.set_parameters(
                R=current_amplitude,
                omega=OMEGA,
                theta=THETA,
                delta=current_delta,
            )

            target_q = controller.update(
                CONTROL_PERIOD
            )

            target_q = np.clip(
                target_q,
                -SAFE_LIMIT,
                SAFE_LIMIT,
            )

            send_angles(
                sock,
                target_q,
            )

            last_command = target_q.copy()

            print(
                f"t={elapsed:6.2f}s  "
                f"R={current_amplitude:.3f}  "
                f"delta={current_delta:+.3f}  "
                f"q=[{target_q.min():+.3f}, "
                f"{target_q.max():+.3f}]"
            )

            loop_elapsed = (
                time.monotonic()
                - step_start
            )

            remaining = (
                CONTROL_PERIOD
                - loop_elapsed
            )

            if remaining > 0:
                time.sleep(remaining)
     

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        print(
            "Returning slowly to neutral..."
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

        print("Robot stopped.")


if __name__ == "__main__":
    main()