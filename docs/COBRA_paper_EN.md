# Paper Summary (Revised for Coursework)
## Hierarchical RL-Guided Large-scale Navigation of a Snake Robot

Shuo Jiang, Adarsh Salagame, Alireza Ramezani, Lawson Wong (Northeastern University), arXiv:2312.03223v1, 2023

**Revision policy**
- Hardware (head/tail locking mechanism, tumbling mode) is out of scope for this implementation → removed
- The physics simulator used is MuJoCo → since MuJoCo internally handles paper §III (Euler-Lagrange equations of motion, Stribeck contact model), those details are omitted. Only the MuJoCo configuration notes relevant to it are kept in §5
- The coursework robot has **yaw-axis rotation only** → a single CPG suffices, and the action space/gait differ from the paper. The relevant parts are rewritten in §3, §4
- The meaning of the CPG equations is explained carefully in §2

---

## 1. Core Logic of the Research (what implementers must grasp)

**Problem**: autonomously navigate a large-scale, complex environment (a maze) with a high-DOF snake robot.

**Limits of naive approaches**:
- Fixed gait → only works for specific terrain
- MPC → accurate modeling of contact-rich dynamics is difficult
- Direct RL in joint space → high-dimensional continuous action space + long episodes → learning becomes hopelessly slow (the paper's comparison experiment shows **526 hours vs. 12.25 hours**)

**Proposal**: decompose into 4 layers.

```
[A*] Global Path Planning  … map → sequence of waypoints
   ↓
[RL] Local Navigation      … state → CPG parameters (low-dimensional, low-frequency)
   ↓
[CPG] Gait Generation      … CPG parameters → target angle trajectory for each joint (sinusoid)
   ↓
[PID] Gait Control         … target angle → torque
```

**Why this decomposition works (the core point to explain to students)**:
1. **RL's action space becomes dramatically smaller.** Instead of directly outputting joint angles, RL only needs to output "gait parameters" (amplitude, frequency, phase difference, offset). The CPG guarantees the strong prior of periodic motion.
2. **RL's decision-making frequency drops** (0.5 Hz vs. CPG output at 50 Hz). The number of RL steps per episode (rollout length) shrinks to 1/100, speeding up learning.
3. **Thanks to the Global Planner, RL only has to solve the local problem of "go to the nearby waypoint."** Since this local policy is environment-independent, **it can transfer zero-shot to a new maze** (as long as a map is available, A* produces the path, and the local policy is reused).

---

## 2. Explanation of the CPG (Central Pattern Generator) ★ Important

A CPG is a model of the neural circuit in a biological spinal cord that generates rhythmic signals. Here it should be understood as "**a dynamical system that, given a small number of external parameters, emits smooth sinusoidal trajectories for all joints**." Equation (9) of the paper is the following set of three:

$$
\dot\varphi = \omega + A\varphi + B\theta \qquad \text{(phase dynamics)}
$$
$$
\ddot r = a\left[\frac{a}{4}(R-r) - \dot r\right] \qquad \text{(amplitude dynamics)}
$$
$$
x = r\cdot\sin(\varphi) + \delta \qquad \text{(output)}
$$

$\varphi, r \in \mathbb{R}^n$ are the internal state ($n$ = number of joints), and $x\in\mathbb{R}^n$ is the target angle of each joint. $R,\omega,\theta,\delta$ are external inputs (i.e., what RL decides).

### 2.1 The output equation: why $x = r\sin\varphi + \delta$

The basis of snake propulsion is the **serpenoid curve**: if each joint traces
$$
x_i(t) = A\sin(\omega t + (i-1)\beta) + \delta
$$
— that is, "**sinusoids of the same amplitude and frequency, with adjacent joints offset by a constant phase difference $\beta$**" — a traveling wave is generated along the body, and anisotropic friction with the ground produces forward motion.

In other words, only 4 quantities are needed:

| Quantity | Symbol | Physical meaning |
|---|---|---|
| Amplitude | $R$ | Magnitude of the body's undulation (larger → sharper bending) |
| Frequency | $\omega$ | Speed of the undulation (sign reverses forward/backward) |
| Phase difference | $\theta$ | Wavelength of the traveling wave. How many waves fit on the body |
| Offset | $\delta$ | Shifts the sinusoid's center → the whole body bends into an arc → produces **turning** |

Rather than hard-coding these four quantities directly, the CPG **realizes them as the equilibrium point of a differential equation**. Because of this, even if the input changes abruptly, the output transitions continuously and smoothly to the new gait (this is the CPG's biggest advantage — if you compute the sinusoid directly from a formula, the joint angle jumps discontinuously the instant a parameter changes).

### 2.2 The amplitude equation: why is it second-order?

Writing out $\ddot r = a\left[\frac{a}{4}(R-r) - \dot r\right]$ gives
$$
\ddot r + a\dot r + \frac{a^2}{4} r = \frac{a^2}{4}R
$$
The characteristic equation is $s^2 + as + a^2/4 = (s + a/2)^2 = 0$, a **repeated root $s = -a/2$ → critically damped**.

In other words, this equation is "**a second-order system in which the amplitude $r$ converges to the target amplitude $R$ as fast as possible without overshoot**." $a$ is the hyperparameter determining the speed of convergence, with a time constant of roughly $2/a$. Even if RL changes the amplitude command $R$ in a step, the actual amplitude $r$ tracks it smoothly.

### 2.3 The phase equation: why the matrices $A$, $B$ (the most confusing part)

$$\dot\varphi = \omega + A\varphi + B\theta$$

Start by **pinning down the type (dimension) of each symbol** in this equation. If this is left vague, nothing that follows will make sense.

| Symbol | Type | Meaning |
|---|---|---|
| $\varphi$ | $\mathbb{R}^n$ | The **phase** of each oscillator (= each joint). An angle that grows with time |
| $\omega$ | $\mathbb{R}^n$ | The **natural angular velocity** of each oscillator (commanded by RL; the same value is put in every component) |
| $\theta$ | $\mathbb{R}^{n-1}$ | The **target phase difference between adjacent oscillators**. With $n$ joints there are $n-1$ "adjacent pairs," hence $n-1$ dimensions |
| $A$ | $\mathbb{R}^{n\times n}$ | Coupling matrix between oscillators (Eq. 10) |
| $B$ | $\mathbb{R}^{n\times(n-1)}$ | Matrix distributing $\theta$ to each oscillator (Eq. 11) |
| $\mu_i$ | scalar | Coupling strength (hyperparameter) |

#### (a) Definitions of $A$ and $B$

**$A$ (Eq. 10, an $n \times n$ tridiagonal matrix)**:

$$
A = \begin{bmatrix}
-\mu_1 & \mu_1 & & & \\
\mu_2 & -2\mu_2 & \mu_2 & & \\
& \ddots & \ddots & \ddots & \\
& & \mu_{n-1} & -2\mu_{n-1} & \mu_{n-1} \\
& & & \mu_n & -\mu_n
\end{bmatrix}
$$

Reading row $i$ (an interior row, $2 \le i \le n-1$): column $i-1$ holds $\mu_i$, column $i$ holds $-2\mu_i$, and column $i+1$ holds $\mu_i$. That is,
$$(A\varphi)_i = \mu_i\big(\varphi_{i-1} - 2\varphi_i + \varphi_{i+1}\big)$$
At both ends ($i=1, n$), there is only one neighbor, so instead of $-2\mu$ it becomes $-\mu$:
$$(A\varphi)_1 = \mu_1(\varphi_2 - \varphi_1), \qquad (A\varphi)_n = \mu_n(\varphi_{n-1} - \varphi_n)$$

**$B$ (Eq. 11, an $n \times (n-1)$ matrix)**:

$$
B = \begin{bmatrix}
1 & & & \\
-1 & 1 & & \\
& -1 & \ddots & \\
& & \ddots & 1 \\
& & & -1 \\
\end{bmatrix}
$$

Row $i$ has $-1$ in column $i-1$ and $+1$ in column $i$ (out-of-range entries are ignored). Hence
$$(B\theta)_1 = \theta_1, \qquad (B\theta)_i = \theta_i - \theta_{i-1}, \qquad (B\theta)_n = -\theta_{n-1}$$

#### (b) Expanding to a single line

Putting this together, the phase of the $i$-th oscillator is
$$
\boxed{\ \dot\varphi_i = \omega_i + \mu_i\big(\varphi_{i-1} - 2\varphi_i + \varphi_{i+1}\big) + \big(\theta_i - \theta_{i-1}\big)\ }
$$
which can be written **using only itself and its left/right neighbors**. The structure is a chain of $n$ oscillators, each talking only to its neighbors.

#### (c) What "diffusive coupling" means

The term $\varphi_{i-1} - 2\varphi_i + \varphi_{i+1}$ in $A\varphi$ is exactly the **finite-difference approximation of a second derivative** ($\partial^2\varphi/\partial i^2$). Spatially discretizing the heat equation $\partial u/\partial t = \kappa\,\partial^2 u/\partial x^2$ produces exactly the same form. Hence it is called "diffusion."

Intuitively, think of it as **springs connecting adjacent oscillators**.

- If $\varphi_i$ lags behind the average of its two neighbors, $\varphi_{i-1}-2\varphi_i+\varphi_{i+1} > 0$, so $\dot\varphi_i$ increases and it **tries to catch up**
- If it is too far ahead, it is conversely **pulled back**
- As a result, left alone, everyone's phase converges (synchronizes)

An important property of this matrix is that **each row sums to zero**:
$$-\mu_1 + \mu_1 = 0,\qquad \mu_i - 2\mu_i + \mu_i = 0,\qquad \mu_n - \mu_n = 0$$
Row sum zero ⟺ for the all-ones vector $\mathbf{1}=[1,\dots,1]^\top$, $A\mathbf{1} = 0$. That is, **the coupling term exerts no force whatsoever on the motion of "shifting everyone's phase together by the same amount."**

This is essential to how it works. Splitting the phase into "overall rotation (common advance shared by everyone)" and "relative phase (difference from neighbors),"

- **Overall rotation** is never interfered with by $A$ and is determined solely by $\omega$ → all joints **necessarily oscillate at the same frequency $\omega$**
- Only the **relative phase** is regulated by $A$

The equation neatly separates "matching frequency" from "shaping phase differences" in a single expression. (A tridiagonal coupling matrix with zero row sums like this is called a **graph Laplacian** in graph theory. $A$ is the Laplacian of a "path graph with $n$ nodes in a row," multiplied by $-\mu_i$.)

#### (d) The role of $B\theta$: injecting the "target value" of the phase difference

With $A$ alone, everyone's phase would **completely coincide** (all joints bend the same direction at the same time = no traveling wave = no forward motion). What we want is a state where "adjacent joints are offset by $\theta$." So $B\theta$ **supplies the target offset as a bias**.

Indeed, regrouping the coupling term gives (this matches $A\varphi + B\theta$ exactly when $\mu_i=1$; even for $\mu_i \ne 1$ the intent is the same, and this form is common in the original CPG literature):

$$
\dot\varphi_i = \omega_i + \mu_i\Big[\underbrace{(\varphi_{i-1}-\varphi_i - \theta_{i-1})}_{\text{error from left neighbor's offset}} + \underbrace{(\varphi_{i+1}-\varphi_i + \theta_i)}_{\text{error from right neighbor's offset}}\Big]
$$

Each bracket is an **error**: "actual phase difference − target phase difference." The equilibrium is where the error is zero, at which point
$$
\varphi_i - \varphi_{i+1} = \theta_i
$$
that is, **$\theta_i$ is exactly the phase difference (phase lag) by which joint $i$ leads joint $i+1$**. The reason $B$ has the $[+1, -1]$ difference structure is to convert the $n-1$ "quantities on links" $\theta$ into the $n$ "quantities on nodes" $\varphi$ (what graph theory calls the **incidence matrix**).

> **Summarizing with the spring analogy**: $A$ is "a spring trying to pull neighbors together," and $B\theta$ is "that spring's natural length." If the natural length is zero ($\theta=0$), everyone sticks together in the same phase; if the natural length is set to $\theta$, adjacent joints settle into equilibrium exactly $\theta$ apart.

#### (e) Conclusion

In short, the phase dynamics form

> **a synchronization circuit that spins all oscillators at a common frequency $\omega$ while automatically aligning the phase difference between neighbors to $\theta$.**

Once aligned, the joint angle $x_i = r_i\sin(\varphi_i)+\delta_i$ becomes "a traveling wave with a constant phase difference propagating through the body," which is the source of propulsion. Even with random initial phases, it spontaneously converges to the correct gait, and even if $\theta$ is changed mid-motion, it transitions smoothly to the new wavelength — this is the practical benefit of using a CPG.

> **Implementation tip**: since $\varphi$ increases monotonically if left alone, it's fine to take it mod $2\pi$ (only $\sin$ of it is used, so the output is unaffected). However, when computing the coupling term $\varphi_{i-1}-\varphi_i$, if the phase difference jumps by $2\pi$ due to mod wraparound, the coupling breaks. After taking the difference, normalize it into $[-\pi,\pi)$ with `np.arctan2(np.sin(d), np.cos(d))`. This is a spot students are guaranteed to trip on.

---

## 3. Reinterpreting for a Yaw-Only Robot ★ Core of the coursework

The paper's COBRA has both pitch and yaw joints and used **two CPGs** (one for pitch, one for yaw) to realize **sidewinding**.

Since the coursework robot has **yaw axis only**:

| Item | Paper (pitch+yaw) | Coursework (yaw only) |
|---|---|---|
| Number of CPGs | 2 (pitch, yaw) | **1** |
| Gait | Sidewinding | **Lateral undulation (serpentine)** |
| Action space | $\mathbb{R}^7$ ($R_1,R_2,\omega,\theta_1,\theta_2,\delta_1,\delta_2$) | **$\mathbb{R}^4$** ($R, \omega, \theta, \delta$) |
| Frequency constraint | The two CPGs' frequencies must match ($\omega_1=\omega_2$) | **Constraint disappears since there's only one CPG** |

**Action space (coursework version)** — ranges follow the paper's values:
- Amplitude $R \in [0, 1.5]$
- Frequency $\omega \in [-0.1, 0.1]$
- Phase difference $\theta \in [-\pi, \pi]$
- Offset $\delta \in [-0.1, 0.1]$

→ The Actor's output layer becomes tanh(4) (changed from tanh(7) in the paper's Fig. 6).

**Note on the propulsion principle**: sidewinding is a 3D gait that lifts the body and moves the contact points, but planar lateral undulation **absolutely cannot move forward without anisotropic ground friction (low resistance in the direction of travel, high resistance sideways)**. This is the point students get stuck on most in the coursework; the workaround is described in §5.

**State space (proposed coursework version)**: the paper uses 21 dimensions (joint positions $\mathbb{R}^n$ + IMU $\mathbb{R}^3$ + translational displacement $\mathbb{R}^3$ + relative rotation axis-angle $\mathbb{R}^4$). For planar motion this can be reduced as follows:
- Joint angles $\mathbb{R}^n$ ($n$ = number of yaw joints)
- IMU (body angular velocity/acceleration) $\mathbb{R}^3$ ($\mathbb{R}^2$–$\mathbb{R}^3$ for planar motion)
- Waypoint's relative position as seen from the robot's frame $\mathbb{R}^2$
- Waypoint's relative heading as seen from the robot's frame (expressing it as $\sin\psi, \cos\psi$ in 2 dimensions avoids the angular discontinuity) $\mathbb{R}^2$

※ As the paper argues, restrict everything to **ego-centric observations (obtained from the robot's own sensors)**. Not putting information equivalent to external motion capture (absolute position in world coordinates) into the state is this paper's design philosophy.

---

## 4. RL Component (as in the paper, only the action dimension changes)

### 4.1 Global Path Planning
- Search for the shortest path on an occupancy grid map with **A\***
- Place waypoints along the path so that the distance between adjacent waypoints stays within a fixed range
- Mazes are randomly generated with **Kruskal's algorithm** (paper Fig. 8). Randomizing during training enables zero-shot transfer to new environments

### 4.2 Reward function (Eq. 12)
$$
r_1 = \frac{1}{0.1 + d_t}, \qquad
r_2 = d_{t-1} - d_t, \qquad
r_3 = \|a_t - a_{t-1}\|_2
$$
- $d_t$: displacement (distance) between the robot frame and the waypoint frame
- $r_1$: larger reward the closer to the goal. When far away, $r_1 \to 0$ and has little effect
- $r_2$: reward for the speed of approaching the goal. Near the goal, $r_2 \to 0$. → **$r_1$ and $r_2$ are complementary** ($r_2$ dominates when far, $r_1$ dominates when close)
- $r_3$: **penalty on the change in action (CPG parameters) between consecutive decision steps.** Ensures smooth transitions between gaits
- ⚠️ **The paper does not specify the weighting coefficients for $r_1,r_2,r_3$** (the total reward should take the form $w_1 r_1 + w_2 r_2 - w_3 r_3$; note the sign of $r_3$ — it's a penalty, hence subtracted). This is the part students are meant to design and tune in the coursework.

### 4.3 Training setup
| Item | Value |
|---|---|
| RL algorithm | **DDPG** |
| RL decision frequency | **0.5 Hz** (acts only once every 2 seconds) |
| CPG output / motor control frequency | **50 Hz** |
| Episode length | 160 seconds |
| Goal relocation range | 8m × 8m |
| Total episodes | 40,000 (training takes a few hours) |

**Design intent behind the frequencies (important)**: since the CPG produces a rhythmic gait, **the gait needs to keep running unchanged for a while before it becomes effective**. If RL rewrote the CPG parameters every step, no gait could form. Hence RL:CPG = 1:100 in frequency ratio. In implementation this becomes a loop of "RL outputs one action → hold that action for 100 control steps (= 2 seconds) while running CPG+PID → put the resulting transitioned state and accumulated reward into the replay buffer as one transition."

### 4.4 Network (paper Fig. 6, only the action dimension changes)
- **Actor**: State → ReLU(512) → ReLU(256) → ReLU(128) → tanh(**4**) ※ the tanh output is linearly scaled to each parameter's range
- **Critic**: (State, Action) → ReLU(512) → ReLU(512) → ReLU(256) → Linear(1)

### 4.5 Gait Control
- **PID control** for each joint. Tracks the target joint angle $x_i$ produced by the CPG to generate torque $u$
- In MuJoCo, using a `position` actuator (roughly equivalent to PD internally) can substitute for this in most cases. `kp`, `kv` tuning is required

---

## 5. Notes on the MuJoCo Implementation

Since MuJoCo solves the physical model (equations of motion, contact forces) for you, implementing paper §III is unnecessary. However, the following are points about **how to reproduce, in MuJoCo, what the paper's contact model (Eq. 8) was doing**.

1. **Anisotropic friction is the most important thing.** The paper's ground contact model includes direction-dependent friction, and without it lateral undulation will not move forward (it will just wiggle in place). There are mainly two options in MuJoCo:
   - **Attach passive wheels to each link**: the most common approach on real snake robots. Free to roll in the rolling direction, constrained sideways. In the model, attach a thin cylinder/capsule with a rotational joint.
   - **Vary the sliding friction coefficient by direction in the contact**: MuJoCo allows setting separate sliding friction coefficients for the two tangential directions of the contact frame (requires the `friction` setting on `geom`/`pair` and specifying `condim`). However, this depends on how the contact frame's orientation is determined, so **be sure to check the behavior in the MuJoCo documentation before configuring this** (worth double-checking yourself too).
2. **Stribeck friction** (the smooth transition from static to kinetic friction) is not in MuJoCo's standard friction model. It's usually fine to ignore, but it's worth telling students that low-speed behavior won't strictly match the paper.
3. Depending on the combination of **PID gains** and CPG amplitude/frequency, the target angle may not be tracked and the gait may fall apart. The first milestone should probably be to remove RL, **fix the CPG parameters by hand, and check that the robot can go straight and turn.**

---

## 6. Recommended Implementation Milestones (suggested coursework progression)

1. **Build the MuJoCo model**: a snake robot with $n$ yaw joints + anisotropic friction (passive wheels or friction settings). Position actuators.
2. **Implement the CPG standalone**: implement Eqs. (9)–(11) (single-system version) in Python. Without RL, give $R, \omega, \theta, \delta$ by hand and confirm **it moves forward and turns via $\delta$**. ← If this doesn't pass, moving forward is pointless
3. **A\* + waypoint generation**: generate a maze with Kruskal's algorithm → occupancy grid → A\* → evenly spaced waypoints
4. **Turn into a Gym environment**: a loop with state (ego-centric), action ($\mathbb{R}^4$), reward (Eq. 12), RL at 0.5Hz / control at 50Hz
5. **DDPG training**: learn local navigation to a single waypoint
6. **Integration and zero-shot evaluation**: evaluate by combining A\* with the trained local policy on unseen mazes

### Places students are likely to get stuck (heads-up notes)
- No forward motion because there's no anisotropic friction (→ §5-1)
- The coupling term breaks due to the phase mod operation (→ §2.3 Tips)
- Not implementing the RL:CPG frequency ratio (1:100) and instead issuing an action every step, so no gait forms
- Reward weight design (not specified in the paper)
- Feeding the relative heading in as a raw angle into the state, causing training instability from the $\pm\pi$ discontinuity (→ use $\sin,\cos$ representation)

---

*Source: S. Jiang, A. Salagame, A. Ramezani, L. Wong, "Hierarchical RL-Guided Large-scale Navigation of a Snake Robot," arXiv:2312.03223v1 [cs.RO], 2023.*
