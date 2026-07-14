import numpy as np
import mujoco
import mujoco.viewer
import os
import time

class PaperCPG:
    """Paper-based CPG implementation following Equation 9"""
    def __init__(self, n_joints=12, dt=0.02): #CPG generate nerate output signals at 50Hz
        self.n_joints = n_joints
        self.dt = dt

        # CPG internal states (Equation 9)
        self.phi = np.zeros(n_joints)      # Phase variables φ
        self.r = np.ones(n_joints)         # Amplitude variables r
        self.r_dot = np.zeros(n_joints)    # Amplitude derivatives ṙ

        # CPG parameters
        self.R = np.ones(n_joints)                         # Target amplitude R
        self.omega =  0.05 * np.ones(n_joints)             # Base omega ω
        self.theta = self._create_phase_shifts(np.pi/4)    # Phase shift pattern θ (n-1 elements)
        self.delta = np.zeros(n_joints)                    # Offset δ

        # CPG hyper parameters
        self.a = 50.0                        # Convergence rate a
        self.mu = 2.0 * np.ones(n_joints)    # Coupling strength μ

        # Coupling matrices A and B (Equations 10, 11)
        self.A = self._create_coupling_matrix_A()
        self.B = self._create_coupling_matrix_B()

    def _create_coupling_matrix_A(self):
        """Create coupling matrix A (Equation 10)"""
        A = np.zeros((self.n_joints, self.n_joints))
        for i in range(self.n_joints-1):
            A[i, i+1] = self.mu[i]      # Upper diagonal
            A[i+1, i] = self.mu[i+1]    # Lower diagonal
            A[i, i] -= self.mu[i]       # Diagonal (negative coupling)
            if i > 0:
                A[i, i] -= self.mu[i]
        # Last element
        A[self.n_joints-1, self.n_joints-1] = -self.mu[self.n_joints-1]
        return A

    def _create_coupling_matrix_B(self):
        """Create coupling matrix B (Equation 11)"""
        B = np.zeros((self.n_joints, self.n_joints-1))
        for i in range(self.n_joints-1):
            B[i, i] = 1.0
            B[i+1, i] = -1.0
        return B

    def _create_phase_shifts(self, phase_shift):
        """Create phase shift pattern for snake locomotion (paper IV.D.3: a
        single value theta duplicated across channels, not an increasing series)"""
        return np.full(self.n_joints - 1, phase_shift)

    def _r_system(self, r, r_dot):
        r_ddot = self.a * (self.a/4 * (self.R - r) - r_dot)
        return r_dot, r_ddot

    def update(self):
        """Update CPG states using paper's dynamics (Equation 9)"""
        # Phase dynamics: φ̇ = ω + A·φ + B·θ
        phi_dot = self.omega + np.dot(self.A, self.phi) + np.dot(self.B, self.theta)

        # Amplitude dynamics: r̈ = a·[a/4·(R - r) - ṙ]
        r_ddot = self.a * (self.a/4 * (self.R - self.r) - self.r_dot)

        # Euler integration
        self.phi += phi_dot * self.dt
        self.r_dot += r_ddot * self.dt
        self.r += self.r_dot * self.dt

        # Output: x = r·sin(φ) + δ (Equation 9)
        x = self.r * np.sin(self.phi) + self.delta

        return x

    def reset_state(self):
        """Reset the CPG's internal state (phase and amplitude dynamics) to
        their initial values. Call this at the start of every episode so
        that identical observations/actions produce identical joint targets
        regardless of what happened in the previous episode."""
        self.phi = np.zeros(self.n_joints)
        self.r = np.ones(self.n_joints)
        self.r_dot = np.zeros(self.n_joints)

    def set_parameters(self, R=1.0, omega=0.05, theta=np.pi/4, delta=0.0):
        """Set CPG parameters (maps to paper's parameters)"""
        self.R = R * np.ones(self.n_joints)              # Target amplitude R
        self.omega = omega * np.ones(self.n_joints)      # Base omega ω
        self.theta = self._create_phase_shifts(theta)    # Phase shift pattern θ (n-1 elements)
        self.delta = delta * np.ones(self.n_joints)      # Offset δ


    def set_hyper_parameters(self, convergence_rate=50.0, coupling_strength=2.0):
        """Set advanced parameters specific to paper's CPG"""
        self.a = convergence_rate
        self.mu = coupling_strength * np.ones(self.n_joints)
        # Recreate coupling matrices with new parameters
        self.A = self._create_coupling_matrix_A()

def Target_q_CPG(cpg):
    """Generate target joint positions using CPG (works with both SimpleCPG and PaperCPG)"""
    return cpg.update()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    model = mujoco.MjModel.from_xml_path(os.path.join(repo_root, "scenes", "scene.xml"))
    data = mujoco.MjData(model)

    # Initialize paper-based CPG
    n_joints = 12         # Number of joints in snake robots
    CPG_frequency = 50    # CPGs generate outputsignals at 50Hz to control the motors.
    # pid_controllers = [PIDController(kp=10.0, ki=0.1, kd=0.5, dt=1.0/CPG_frequency) for _ in range(n_joints)]
    # dt MUST match the physics timestep (model.opt.timestep, default 0.002s),
    # because cpg.update() is called once per mj_step below. Using 1/CPG_frequency
    # (0.02s) advanced the phase 10x too fast per physics step -> frantic wiggle.
    cpg = PaperCPG(n_joints=n_joints, dt=model.opt.timestep)

    # Set advanced parameters for paper CPG
    cpg.set_hyper_parameters(
        convergence_rate= 100.0,    # 'a' parameter
        coupling_strength= 2.0     # 'μ' parameter
    )

    # Set basic CPG parameters for snake locomotion
    cpg.set_parameters(
        R=0.4,            # Joint amplitude (rad; ~23 deg. R=2 was ~115 deg -> unphysical)
        omega=2.0,       # Oscillation omega (rad/s; period = 2*pi/omega)
        theta=np.pi/6,    # Phase shift between joints (wavelength)
        delta=0         # Joint offset (0 = straight; nonzero = turn)
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Setting camera
        viewer.cam.type = 1  # 0:free 1:track 2:fixed
        viewer.cam.trackbodyid = 0
        viewer.cam.distance = 2

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(2)

        start = time.time()
        step = 0

        print(f"CPG-based snake robot control started...")
        print(f"Advanced parameters: convergence_rate={cpg.a:.1f}, coupling_strength={cpg.mu[0]:.1f}")

        while viewer.is_running():
            step_start = time.time()
            step += 1

            # Generate control signals using CPG
            target_positions = Target_q_CPG(cpg)

            # Apply control signals to joints
            data.ctrl[:] = target_positions
            # for i in range(n_joints):
            #     current_pos = data.qpos[i]
            #     torque = pid_controllers[i].compute(target_positions[i], current_pos)
            #     data.ctrl[i] = torque
            # Print status every 200 steps (4 seconds)
            if step % 200 == 0:
                t = step * model.opt.timestep
                print(f"Time: {t:.2f}s, Step: {step}, CPG output range: [{np.min(target_positions):.3f}, {np.max(target_positions):.3f}]")
            cpg.set_parameters(
                R=0.4,            # Joint amplitude (rad; ~23 deg. R=2 was ~115 deg -> unphysical)
                omega=2.0,       # Oscillation omega (rad/s; period = 2*pi/omega)
                theta=np.pi/6,    # Phase shift between joints (wavelength)
                delta=0         # Joint offset (0 = straight; nonzero = turn)
            )
            mujoco.mj_step(model, data)
            viewer.sync()

            # Real-time control
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
