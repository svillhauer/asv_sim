import gymnasium as gym
from gymnasium import spaces
import numpy as np


class AsvTwoGliderEnv(gym.Env):
    """
    D-Optimal EKF-based ASV positioning for tracking TWO gliders.
    
    The two gliders move in parallel lines, separated by a configurable
    distance (glider_separation). Each glider has its own independent EKF.
    The reward is the SUM of D-optimal criteria for both gliders, forcing
    the ASV to balance tracking both targets.
    
    Current-compensated EKF:
      - When a glider is IN comm range, it transmits its depth-averaged
        current (DAC) to the ASV, and the EKF predict step uses the
        glider's actual local current for drift compensation.
      - When a glider is OUT of comm range, the EKF falls back to the
        ASV's own ADCP reading as a proxy for the glider's current.
    
    ─────────────────────────────────────────────────────────────────────
    State  (20-dimensional, normalized to [-1, 1]):
    
      ASV state (3 dims):
        [0]  ASV heading                        (/ pi)
        [1]  ASV last commanded surge           (normalized)
        [2]  ASV last commanded yaw rate        (normalized)
    
      Glider 1 — relative to ASV (7 dims):
        [3]  estimated distance to G1           (normalized to world_size)
        [4]  range rate to G1                   (normalized to 3 m/s)
        [5]  Relative X (est) to G1             (normalized to world_size)
        [6]  Relative Y (est) to G1             (normalized to world_size)
        [7]  EKF Unc Major Axis G1              (normalized to obs_unc_scale)
        [8]  EKF Unc Minor Axis G1              (normalized to obs_unc_scale)
        [9]  EKF Unc Orientation Angle G1       (/ pi)
    
      Glider 2 — relative to ASV (7 dims):
        [10] estimated distance to G2           (normalized to world_size)
        [11] range rate to G2                   (normalized to 3 m/s)
        [12] Relative X (est) to G2             (normalized to world_size)
        [13] Relative Y (est) to G2             (normalized to world_size)
        [14] EKF Unc Major Axis G2              (normalized to obs_unc_scale)
        [15] EKF Unc Minor Axis G2              (normalized to obs_unc_scale)
        [16] EKF Unc Orientation Angle G2       (/ pi)
    
      Shared glider info (1 dim):
        [17] estimated glider heading           (/ pi) — same for both
        
      ASV Environment Sensors (2 dims):
        [18] ASV local current X (ADCP)         (normalized to current_max)
        [19] ASV local current Y (ADCP)         (normalized to current_max)
    ─────────────────────────────────────────────────────────────────────
    
    Action  (2-D continuous [-1, 1]):
        [0] Commanded Surge
        [1] Commanded Yaw Rate
    ─────────────────────────────────────────────────────────────────────
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = config or {}

        # --- World ---
        self.dt = float(cfg.get("dt", 0.5))
        self.max_steps = int(cfg.get("max_steps", 1000))
        self.world_size = float(cfg.get("world_size", 5000.0))

        # --- ASV Kinematics ---
        self.max_cmd = float(cfg.get("max_cmd", 2.0))
        self.max_yaw_rate = float(cfg.get("max_yaw_rate", 0.15))

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # --- Observation: 20-dim (18 original + 2 ADCP) ---
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(20,), dtype=np.float32)

        # --- Glider parameters ---
        self.glider_speed_mean = float(cfg.get("glider_speed", 0.30))
        self.glider_depth = float(cfg.get("glider_depth", 100.0))
        # Distance between the two parallel glider tracks (perpendicular to heading)
        self.glider_separation = float(cfg.get("glider_separation", 400.0))

        # --- EKF & Acoustic parameters ---
        self.comm_range = float(cfg.get("comm_range", 800.0))
        self.q_pos_std = float(cfg.get("q_pos_std", 0.1))
        self.q_vel_std = float(cfg.get("q_vel_std", 0.001))
        self.r_noise_std = float(cfg.get("r_noise_std", 15.0))
        self.max_pos_std = float(cfg.get("max_pos_std", 2000.0))

        self.obs_unc_scale = float(cfg.get("obs_unc_scale", 100.0))

        # --- Reward weights ---
        self.w_unc = float(cfg.get("w_unc", 1.0))
        self.w_comm = float(cfg.get("w_comm", 0.3))
        self.w_ctrl = float(cfg.get("w_ctrl", 0.005))

        # --- Spawn / Curriculum ---
        self.spawn_range = float(cfg.get("spawn_range", 200.0))
        self.ekf_init_err = float(cfg.get("ekf_init_err", 100.0))

        # --- Ocean Currents ---
        self.current_max = float(cfg.get("current_max", 0.0))           # max current speed (m/s), 0 = disabled
        self.current_stripe_width = float(cfg.get("current_stripe_width", 1000.0))  # stripe width (m)
        self.current_num_stripes = int(cfg.get("current_num_stripes", 60))  # enough to cover world
        self.current_velocities = np.zeros((self.current_num_stripes, 2), dtype=np.float32)

        # --- ASV Internal state ---
        self.asv_pos = np.zeros(2, dtype=np.float32)
        self.asv_heading = 0.0
        self.last_cmd = np.zeros(2, dtype=np.float32)

        # --- Glider 1 ---
        self.g1_pos = np.zeros(2, dtype=np.float32)
        self.g1_vel = np.zeros(2, dtype=np.float32)
        self.g1_x = np.zeros(4, dtype=np.float32)        # EKF state
        self.g1_P = np.eye(4, dtype=np.float32)           # EKF covariance
        self.g1_est_pos = np.zeros(2, dtype=np.float32)
        self.g1_est_vel = np.zeros(2, dtype=np.float32)
        self.g1_unc = 0.0
        self.prev_d1_est = 0.0

        # --- Glider 2 ---
        self.g2_pos = np.zeros(2, dtype=np.float32)
        self.g2_vel = np.zeros(2, dtype=np.float32)
        self.g2_x = np.zeros(4, dtype=np.float32)
        self.g2_P = np.eye(4, dtype=np.float32)
        self.g2_est_pos = np.zeros(2, dtype=np.float32)
        self.g2_est_vel = np.zeros(2, dtype=np.float32)
        self.g2_unc = 0.0
        self.prev_d2_est = 0.0

        # --- EKF Matrices (shared, since both gliders have same dynamics) ---
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        self.Q = np.diag([
            self.q_pos_std ** 2, self.q_pos_std ** 2,
            self.q_vel_std ** 2, self.q_vel_std ** 2
        ]).astype(np.float32)

        self.R = np.array([[self.r_noise_std ** 2]], dtype=np.float32)

        self._rng = np.random.default_rng()
        self.t = 0.0
        self.step_count = 0

    # ─── Utilities ────────────────────────────────────────────────────

    @staticmethod
    def _distance(a, b):
        return float(np.linalg.norm(b - a))

    def _get_current(self, pos):
        """Return the 2D current velocity [cx, cy] at a given world position."""
        half_world = self.current_num_stripes * self.current_stripe_width / 2.0
        stripe_idx = int((pos[0] + half_world) / self.current_stripe_width)
        stripe_idx = np.clip(stripe_idx, 0, self.current_num_stripes - 1)
        return self.current_velocities[stripe_idx]

    def _clamp_covariance(self, P):
        max_var = self.max_pos_std ** 2
        P_clamped = P.copy()
        for i in range(2):
            if P_clamped[i, i] > max_var:
                P_clamped[i, :] = 0.0
                P_clamped[:, i] = 0.0
                P_clamped[i, i] = max_var
        return P_clamped

    # ─── EKF operations (generic, work on any glider's state) ─────────

    def _ekf_predict(self, x, P, current=None):
        """Predict step with optional current compensation. Returns new (x, P).
        
        If current is provided (from ADCP or glider-transmitted DAC),
        the predicted position is offset by the current drift.
        """
        x_new = self.F @ x
        if current is not None:
            x_new[0] += current[0] * self.dt
            x_new[1] += current[1] * self.dt
        P_new = self._clamp_covariance(self.F @ P @ self.F.T + self.Q)
        return x_new, P_new

    def _ekf_update(self, x, P, glider_true_pos):
        """Range-only measurement update. Returns new (x, P)."""
        true_dist = self._distance(self.asv_pos, glider_true_pos)
        slant_range = np.sqrt(true_dist ** 2 + self.glider_depth ** 2)
        z = slant_range + self._rng.normal(0, self.r_noise_std)

        dx = x[0] - self.asv_pos[0]
        dy = x[1] - self.asv_pos[1]
        z_hat = np.sqrt(dx ** 2 + dy ** 2 + self.glider_depth ** 2)
        z_hat_safe = max(z_hat, 1e-6)

        H = np.array([[dx / z_hat_safe, dy / z_hat_safe, 0.0, 0.0]], dtype=np.float32)
        y = np.array([z - z_hat_safe], dtype=np.float32)

        S = H @ P @ H.T + self.R
        K = P @ H.T @ np.linalg.inv(S)

        x_new = x + (K @ y).flatten()
        I_KH = np.eye(4, dtype=np.float32) - K @ H
        P_new = I_KH @ P @ I_KH.T + K @ self.R @ K.T
        P_new = 0.5 * (P_new + P_new.T)

        for i in range(4):
            if P_new[i, i] < 1e-6:
                P_new[i, i] = 1e-6

        return x_new, P_new

    def _sync_glider_estimates(self):
        """Update derived quantities from EKF states."""
        self.g1_est_pos = self.g1_x[0:2].copy()
        self.g1_est_vel = self.g1_x[2:4].copy()
        self.g1_unc = float(np.sqrt(max(self.g1_P[0, 0] + self.g1_P[1, 1], 0.0)))

        self.g2_est_pos = self.g2_x[0:2].copy()
        self.g2_est_vel = self.g2_x[2:4].copy()
        self.g2_unc = float(np.sqrt(max(self.g2_P[0, 0] + self.g2_P[1, 1], 0.0)))

    # ─── Covariance → observation helper ──────────────────────────────

    def _cov_to_obs(self, P):
        """Extract [major_norm, minor_norm, angle/pi] from 4x4 P."""
        P_pos = P[0:2, 0:2]
        eigvals, eigvecs = np.linalg.eigh(P_pos)
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        major_std = np.sqrt(max(eigvals[0], 0.0))
        minor_std = np.sqrt(max(eigvals[1], 0.0))
        major_angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

        major_norm = np.clip(major_std / self.obs_unc_scale, 0.0, 1.0)
        minor_norm = np.clip(minor_std / self.obs_unc_scale, 0.0, 1.0)

        return major_norm * 2.0 - 1.0, minor_norm * 2.0 - 1.0, major_angle / np.pi

    # ─── Reset ────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        # ── Generate ocean currents for this episode ──
        self.current_velocities = np.zeros((self.current_num_stripes, 2), dtype=np.float32)
        if self.current_max > 0:
            current = self._rng.uniform(-self.current_max, self.current_max, size=2)
            for i in range(self.current_num_stripes):
                self.current_velocities[i] = current
                step = self._rng.normal(0, self.current_max * 0.3, size=2)
                current = np.clip(current + step, -self.current_max, self.current_max)

        # Midpoint between the two gliders
        center = self._rng.uniform(-500, 500, size=2).astype(np.float32)
        base_angle = self._rng.uniform(0, 2 * np.pi)
        heading_vec = np.array([np.cos(base_angle), np.sin(base_angle)], dtype=np.float32)
        # Perpendicular to heading (for separation)
        perp_vec = np.array([-np.sin(base_angle), np.cos(base_angle)], dtype=np.float32)

        # Glider 1: offset left of center by half the separation
        self.g1_pos = center + perp_vec * (self.glider_separation / 2.0)
        self.g1_vel = heading_vec * self.glider_speed_mean

        # Glider 2: offset right of center by half the separation
        self.g2_pos = center - perp_vec * (self.glider_separation / 2.0)
        self.g2_vel = heading_vec * self.glider_speed_mean  # same heading, parallel

        # ASV: spawn near the midpoint
        self.asv_pos = center + self._rng.uniform(
            -self.spawn_range, self.spawn_range, size=2
        ).astype(np.float32)
        self.asv_heading = self._rng.uniform(-np.pi, np.pi)

        # EKF for Glider 1
        err1 = self._rng.normal(0, self.ekf_init_err, size=2)
        self.g1_x = np.array([
            self.g1_pos[0] + err1[0], self.g1_pos[1] + err1[1],
            self.g1_vel[0], self.g1_vel[1]
        ], dtype=np.float32)
        self.g1_P = np.diag([
            self.ekf_init_err ** 2, self.ekf_init_err ** 2, 0.001, 0.001
        ]).astype(np.float32)

        # EKF for Glider 2
        err2 = self._rng.normal(0, self.ekf_init_err, size=2)
        self.g2_x = np.array([
            self.g2_pos[0] + err2[0], self.g2_pos[1] + err2[1],
            self.g2_vel[0], self.g2_vel[1]
        ], dtype=np.float32)
        self.g2_P = np.diag([
            self.ekf_init_err ** 2, self.ekf_init_err ** 2, 0.001, 0.001
        ]).astype(np.float32)

        self._sync_glider_estimates()

        self.prev_d1_est = self._distance(self.asv_pos, self.g1_est_pos)
        self.prev_d2_est = self._distance(self.asv_pos, self.g2_est_pos)
        self.last_cmd = np.array([0.0, 0.0], dtype=np.float32)
        self.t = 0.0
        self.step_count = 0

        return self._get_obs(), {}

    # ─── Observation ──────────────────────────────────────────────────

    def _get_obs(self):
        max_dist = self.world_size
        max_rate = 3.0

        # --- Glider 1 obs ---
        d1_est = self._distance(self.asv_pos, self.g1_est_pos)
        d1_rate = (d1_est - self.prev_d1_est) / self.dt if self.step_count > 0 else 0.0
        self.prev_d1_est = d1_est
        los1 = self.g1_est_pos - self.asv_pos
        g1_major, g1_minor, g1_angle = self._cov_to_obs(self.g1_P)

        # --- Glider 2 obs ---
        d2_est = self._distance(self.asv_pos, self.g2_est_pos)
        d2_rate = (d2_est - self.prev_d2_est) / self.dt if self.step_count > 0 else 0.0
        self.prev_d2_est = d2_est
        los2 = self.g2_est_pos - self.asv_pos
        g2_major, g2_minor, g2_angle = self._cov_to_obs(self.g2_P)

        # Glider heading (same for both since parallel)
        g_heading = np.arctan2(self.g1_est_vel[1], self.g1_est_vel[0]) \
            if np.linalg.norm(self.g1_est_vel) > 1e-6 else 0.0

        # --- ASV ADCP (Local Current) ---
        asv_current = self._get_current(self.asv_pos)
        norm_factor = max(self.current_max, 1e-6) # Prevent division by zero if currents are off
        cx_norm = np.clip(asv_current[0] / norm_factor, -1.0, 1.0)
        cy_norm = np.clip(asv_current[1] / norm_factor, -1.0, 1.0)

        obs = np.array([
            # ASV state (3) - Padding removed
            self.asv_heading / np.pi,
            self.last_cmd[0],
            self.last_cmd[1],

            # Glider 1 (7)
            np.clip((d1_est / max_dist) * 2.0 - 1.0, -1, 1),
            np.clip(d1_rate / max_rate, -1, 1),
            np.clip(los1[0] / max_dist, -1, 1),
            np.clip(los1[1] / max_dist, -1, 1),
            g1_major,
            g1_minor,
            g1_angle,

            # Glider 2 (7)
            np.clip((d2_est / max_dist) * 2.0 - 1.0, -1, 1),
            np.clip(d2_rate / max_rate, -1, 1),
            np.clip(los2[0] / max_dist, -1, 1),
            np.clip(los2[1] / max_dist, -1, 1),
            g2_major,
            g2_minor,
            g2_angle,

            # Shared (1)
            g_heading / np.pi,

            # ASV Environment Sensors (2)
            cx_norm,
            cy_norm
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    # ─── Step ─────────────────────────────────────────────────────────

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.last_cmd = action.copy()

        # 1. ASV Dynamics
        surge = action[0] * self.max_cmd
        yaw_rate = action[1] * self.max_yaw_rate
        self.asv_heading += yaw_rate * self.dt
        self.asv_heading = (self.asv_heading + np.pi) % (2 * np.pi) - np.pi

        vx = surge * np.cos(self.asv_heading)
        vy = surge * np.sin(self.asv_heading)
        # Add ocean current at ASV position
        asv_current = self._get_current(self.asv_pos)
        self.asv_pos = self.asv_pos + np.array([vx + asv_current[0], vy + asv_current[1]], dtype=np.float32) * self.dt

        # 2. Glider dynamics (both move in parallel straight lines + current drift)
        g1_current = self._get_current(self.g1_pos)
        g2_current = self._get_current(self.g2_pos)
        self.g1_pos = self.g1_pos + (self.g1_vel + g1_current) * self.dt
        self.g2_pos = self.g2_pos + (self.g2_vel + g2_current) * self.dt

        # 3. Compute distances (needed to decide current source for EKF)
        true_dist1 = self._distance(self.asv_pos, self.g1_pos)
        true_dist2 = self._distance(self.asv_pos, self.g2_pos)

        # 4. EKF Predict — with current compensation
        #    In comm range: glider transmits its local DAC (depth-averaged current)
        #    Out of range:  fall back to ASV's own ADCP reading
        asv_current = self._get_current(self.asv_pos)

        if true_dist1 <= self.comm_range:
            g1_current_for_ekf = self._get_current(self.g1_pos)   # glider-transmitted DAC
        else:
            g1_current_for_ekf = asv_current                       # ASV ADCP fallback

        if true_dist2 <= self.comm_range:
            g2_current_for_ekf = self._get_current(self.g2_pos)   # glider-transmitted DAC
        else:
            g2_current_for_ekf = asv_current                       # ASV ADCP fallback

        self.g1_x, self.g1_P = self._ekf_predict(self.g1_x, self.g1_P, g1_current_for_ekf)
        self.g2_x, self.g2_P = self._ekf_predict(self.g2_x, self.g2_P, g2_current_for_ekf)

        # 5. EKF Update (if in comm range)
        if true_dist1 <= self.comm_range:
            self.g1_x, self.g1_P = self._ekf_update(self.g1_x, self.g1_P, self.g1_pos)
        if true_dist2 <= self.comm_range:
            self.g2_x, self.g2_P = self._ekf_update(self.g2_x, self.g2_P, self.g2_pos)

        self._sync_glider_estimates()
        self.t += self.dt
        self.step_count += 1

        # ─── Reward: Sum of D-optimal for both gliders ────────────────

        det_P1 = max(float(np.linalg.det(self.g1_P[0:2, 0:2])), 1e-10)
        det_P2 = max(float(np.linalg.det(self.g2_P[0:2, 0:2])), 1e-10)
        r_unc1 = -np.log(det_P1)
        r_unc2 = -np.log(det_P2)
        r_unc = min(r_unc1, r_unc2)

        r_ctrl = -float(np.sum(action ** 2))

        # Comm penalty for each glider independently
        r_comm = 0.0
        for excess in [true_dist1 - self.comm_range, true_dist2 - self.comm_range]:
            if excess > 0:
                r_comm -= (excess / self.comm_range)

        reward = (
            self.w_unc * r_unc
            + self.w_comm * r_comm
            + self.w_ctrl * r_ctrl
        )

        terminated = False
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), float(reward), terminated, truncated, {
            "r_unc": r_unc,
            "r_unc1": r_unc1,
            "r_unc2": r_unc2,
            "r_comm": r_comm,
            "r_ctrl": r_ctrl,
            "unc1": self.g1_unc,
            "unc2": self.g2_unc,
            "det_P1": det_P1,
            "det_P2": det_P2,
            "dist1": true_dist1,
            "dist2": true_dist2,
        }

    # ─── Position accessor ────────────────────────────────────────────

    def get_positions(self):
        return {
            "asv": self.asv_pos.copy(),
            "glider1": self.g1_pos.copy(),
            "glider2": self.g2_pos.copy(),
            "glider1_est": self.g1_est_pos.copy(),
            "glider2_est": self.g2_est_pos.copy(),
        }

    def get_current_field(self):
        """Return current stripe data for visualization."""
        return {
            "velocities": self.current_velocities.copy(),
            "stripe_width": self.current_stripe_width,
            "num_stripes": self.current_num_stripes,
            "max_speed": self.current_max,
        }


if __name__ == "__main__":
    env = AsvTwoGliderEnv(config={
        "glider_separation": 400.0,
        "comm_range": 800.0,
        "spawn_range": 200.0,
        "current_max": 0.2, 
    })
    obs, _ = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Initial obs (20-Dim): {np.round(obs, 3)}")

    positions = env.get_positions()
    g1g2_dist = np.linalg.norm(positions["glider1"] - positions["glider2"])
    print(f"\nGlider separation: {g1g2_dist:.1f}m")
    print(f"ASV to G1: {np.linalg.norm(positions['asv'] - positions['glider1']):.1f}m")
    print(f"ASV to G2: {np.linalg.norm(positions['asv'] - positions['glider2']):.1f}m")

    obs, reward, _, _, info = env.step([1.0, 0.5])
    print(f"\nReward: {reward:.3f}")
    print(f"  G1: det_P={info['det_P1']:.1f}, unc={info['unc1']:.1f}m, dist={info['dist1']:.0f}m")
    print(f"  G2: det_P={info['det_P2']:.1f}, unc={info['unc2']:.1f}m, dist={info['dist2']:.0f}m")