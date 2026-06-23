"""
EkfCore: pure-Python port of AsvTwoGliderEnv's EKF + observation builder.

No ROS / Gazebo dependencies — validates cleanly against the original env on a
fixed seed.  The bridge node wraps this class and switches between two modes:
  kinematic (step 2): EkfCore drives ASV position with the same simple dynamics
                      as the Python env → used for bit-exact seed validation.
  gazebo   (step 3+): bridge node feeds real /wamv/pose into set_asv_state().
"""

import numpy as np

# Exact training config for run 20km_ciric_version1
TRAINING_CONFIG = {
    "dt":                  300,
    "max_steps":           288,
    "max_cmd":             2.5,
    "max_yaw_rate":        0.05,
    "world_size":          50000.0,
    "glider_speed":        0.30,
    "glider_depth":        100.0,
    "glider_separation":   20000.0,
    "comm_range":          8000.0,
    "r_noise_std":         15.0,
    "q_pos_std":           0.15,
    "q_vel_std":           0.002,
    "max_pos_std":         20000.0,
    "obs_unc_scale":       10000.0,
    "spawn_range":         2000.0,
    "ekf_init_err":        1000.0,
    "current_max":         0.2,
    "current_stripe_width": 10000.0,
    "current_num_stripes": 60,
}


class EkfCore:
    """
    Mirrors AsvTwoGliderEnv's state, EKF, and observation builder exactly.

    Public API
    ----------
    reset(seed)          → obs (np.float32, shape (20,))
    step(action)         → obs  [kinematic mode only; not used when Gazebo drives ASV]
    get_obs()            → obs  [can be called after set_asv_state() in gazebo mode]
    set_asv_state(pos, heading, last_cmd)  → for gazebo mode
    """

    def __init__(self, config: dict | None = None):
        cfg = {**TRAINING_CONFIG, **(config or {})}

        self.dt                   = float(cfg["dt"])
        self.max_cmd              = float(cfg["max_cmd"])
        self.max_yaw_rate         = float(cfg["max_yaw_rate"])
        self.world_size           = float(cfg["world_size"])
        self.glider_speed_mean    = float(cfg["glider_speed"])
        self.glider_depth         = float(cfg["glider_depth"])
        self.glider_separation    = float(cfg["glider_separation"])
        self.comm_range           = float(cfg["comm_range"])
        self.q_pos_std            = float(cfg["q_pos_std"])
        self.q_vel_std            = float(cfg["q_vel_std"])
        self.r_noise_std          = float(cfg["r_noise_std"])
        self.max_pos_std          = float(cfg["max_pos_std"])
        self.obs_unc_scale        = float(cfg["obs_unc_scale"])
        self.spawn_range          = float(cfg["spawn_range"])
        self.ekf_init_err         = float(cfg["ekf_init_err"])
        self.current_max          = float(cfg["current_max"])
        self.current_stripe_width = float(cfg["current_stripe_width"])
        self.current_num_stripes  = int(cfg["current_num_stripes"])

        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1,       0],
            [0, 0, 0,       1],
        ], dtype=np.float32)

        self.Q = np.diag([
            self.q_pos_std ** 2, self.q_pos_std ** 2,
            self.q_vel_std ** 2, self.q_vel_std ** 2,
        ]).astype(np.float32)

        self.R = np.array([[self.r_noise_std ** 2]], dtype=np.float32)

        # State variables (initialised by reset)
        self.asv_pos      = np.zeros(2, dtype=np.float32)
        self.asv_heading  = 0.0
        self.last_cmd     = np.zeros(2, dtype=np.float32)
        self.current_velocities = np.zeros((self.current_num_stripes, 2), dtype=np.float32)

        self.g1_pos = np.zeros(2, dtype=np.float32)
        self.g1_vel = np.zeros(2, dtype=np.float32)
        self.g1_x   = np.zeros(4, dtype=np.float32)
        self.g1_P   = np.eye(4,   dtype=np.float32)
        self.g1_est_pos = np.zeros(2, dtype=np.float32)
        self.g1_est_vel = np.zeros(2, dtype=np.float32)
        self.prev_d1_est = 0.0

        self.g2_pos = np.zeros(2, dtype=np.float32)
        self.g2_vel = np.zeros(2, dtype=np.float32)
        self.g2_x   = np.zeros(4, dtype=np.float32)
        self.g2_P   = np.eye(4,   dtype=np.float32)
        self.g2_est_pos = np.zeros(2, dtype=np.float32)
        self.g2_est_vel = np.zeros(2, dtype=np.float32)
        self.prev_d2_est = 0.0

        self.step_count = 0
        self._rng = np.random.default_rng()

    # ── Utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def _distance(a, b):
        return float(np.linalg.norm(b - a))

    def _get_current(self, pos):
        half_world = self.current_num_stripes * self.current_stripe_width / 2.0
        stripe_idx = int((pos[0] + half_world) / self.current_stripe_width)
        stripe_idx = np.clip(stripe_idx, 0, self.current_num_stripes - 1)
        return self.current_velocities[stripe_idx]

    def _clamp_covariance(self, P):
        max_var = self.max_pos_std ** 2
        P_c = P.copy()
        for i in range(2):
            if P_c[i, i] > max_var:
                P_c[i, :]  = 0.0
                P_c[:, i]  = 0.0
                P_c[i, i]  = max_var
        return P_c

    # ── EKF ───────────────────────────────────────────────────────────────

    def _ekf_predict(self, x, P, current=None):
        x_new = self.F @ x
        if current is not None:
            x_new[0] += current[0] * self.dt
            x_new[1] += current[1] * self.dt
        P_new = self._clamp_covariance(self.F @ P @ self.F.T + self.Q)
        return x_new, P_new

    def _ekf_update(self, x, P, glider_true_pos):
        true_dist  = self._distance(self.asv_pos, glider_true_pos)
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

        x_new  = x + (K @ y).flatten()
        I_KH   = np.eye(4, dtype=np.float32) - K @ H
        P_new  = I_KH @ P @ I_KH.T + K @ self.R @ K.T
        P_new  = 0.5 * (P_new + P_new.T)

        for i in range(4):
            if P_new[i, i] < 1e-6:
                P_new[i, i] = 1e-6

        return x_new, P_new

    def _sync_estimates(self):
        self.g1_est_pos = self.g1_x[0:2].copy()
        self.g1_est_vel = self.g1_x[2:4].copy()
        self.g2_est_pos = self.g2_x[0:2].copy()
        self.g2_est_vel = self.g2_x[2:4].copy()

    def _cov_to_obs(self, P):
        P_pos = P[0:2, 0:2]
        eigvals, eigvecs = np.linalg.eigh(P_pos)
        idx    = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        major_std   = np.sqrt(max(eigvals[0], 0.0))
        minor_std   = np.sqrt(max(eigvals[1], 0.0))
        major_angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

        major_norm = np.clip(major_std / self.obs_unc_scale, 0.0, 1.0)
        minor_norm = np.clip(minor_std / self.obs_unc_scale, 0.0, 1.0)

        return major_norm * 2.0 - 1.0, minor_norm * 2.0 - 1.0, major_angle / np.pi

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self, seed=None):
        self._rng = np.random.default_rng(seed)

        # Ocean current field
        self.current_velocities = np.zeros((self.current_num_stripes, 2), dtype=np.float32)
        if self.current_max > 0:
            current = self._rng.uniform(-self.current_max, self.current_max, size=2)
            for i in range(self.current_num_stripes):
                self.current_velocities[i] = current
                step = self._rng.normal(0, self.current_max * 0.3, size=2)
                current = np.clip(current + step, -self.current_max, self.current_max)

        center     = self._rng.uniform(-500, 500, size=2).astype(np.float32)
        base_angle = self._rng.uniform(0, 2 * np.pi)
        heading_vec = np.array([np.cos(base_angle), np.sin(base_angle)], dtype=np.float32)
        perp_vec    = np.array([-np.sin(base_angle), np.cos(base_angle)], dtype=np.float32)

        self.g1_pos = center + perp_vec * (self.glider_separation / 2.0)
        self.g1_vel = heading_vec * self.glider_speed_mean
        self.g2_pos = center - perp_vec * (self.glider_separation / 2.0)
        self.g2_vel = heading_vec * self.glider_speed_mean

        self.asv_pos = center + self._rng.uniform(
            -self.spawn_range, self.spawn_range, size=2
        ).astype(np.float32)
        self.asv_heading = self._rng.uniform(-np.pi, np.pi)

        err1 = self._rng.normal(0, self.ekf_init_err, size=2)
        self.g1_x = np.array([
            self.g1_pos[0] + err1[0], self.g1_pos[1] + err1[1],
            self.g1_vel[0], self.g1_vel[1],
        ], dtype=np.float32)
        self.g1_P = np.diag([
            self.ekf_init_err ** 2, self.ekf_init_err ** 2, 0.001, 0.001,
        ]).astype(np.float32)

        err2 = self._rng.normal(0, self.ekf_init_err, size=2)
        self.g2_x = np.array([
            self.g2_pos[0] + err2[0], self.g2_pos[1] + err2[1],
            self.g2_vel[0], self.g2_vel[1],
        ], dtype=np.float32)
        self.g2_P = np.diag([
            self.ekf_init_err ** 2, self.ekf_init_err ** 2, 0.001, 0.001,
        ]).astype(np.float32)

        self._sync_estimates()
        self.prev_d1_est = self._distance(self.asv_pos, self.g1_est_pos)
        self.prev_d2_est = self._distance(self.asv_pos, self.g2_est_pos)
        self.last_cmd    = np.array([0.0, 0.0], dtype=np.float32)
        self.step_count  = 0

        return self.get_obs()

    # ── Observation ───────────────────────────────────────────────────────

    def get_obs(self):
        max_dist = self.world_size
        max_rate = 3.0

        d1_est   = self._distance(self.asv_pos, self.g1_est_pos)
        d1_rate  = (d1_est - self.prev_d1_est) / self.dt if self.step_count > 0 else 0.0
        self.prev_d1_est = d1_est
        los1     = self.g1_est_pos - self.asv_pos
        g1_major, g1_minor, g1_angle = self._cov_to_obs(self.g1_P)

        d2_est   = self._distance(self.asv_pos, self.g2_est_pos)
        d2_rate  = (d2_est - self.prev_d2_est) / self.dt if self.step_count > 0 else 0.0
        self.prev_d2_est = d2_est
        los2     = self.g2_est_pos - self.asv_pos
        g2_major, g2_minor, g2_angle = self._cov_to_obs(self.g2_P)

        g_heading = (
            np.arctan2(self.g1_est_vel[1], self.g1_est_vel[0])
            if np.linalg.norm(self.g1_est_vel) > 1e-6
            else 0.0
        )

        asv_current = self._get_current(self.asv_pos)
        norm_factor = max(self.current_max, 1e-6)
        cx_norm = np.clip(asv_current[0] / norm_factor, -1.0, 1.0)
        cy_norm = np.clip(asv_current[1] / norm_factor, -1.0, 1.0)

        obs = np.array([
            self.asv_heading / np.pi,
            self.last_cmd[0],
            self.last_cmd[1],

            np.clip((d1_est / max_dist) * 2.0 - 1.0, -1, 1),
            np.clip(d1_rate / max_rate, -1, 1),
            np.clip(los1[0] / max_dist, -1, 1),
            np.clip(los1[1] / max_dist, -1, 1),
            g1_major, g1_minor, g1_angle,

            np.clip((d2_est / max_dist) * 2.0 - 1.0, -1, 1),
            np.clip(d2_rate / max_rate, -1, 1),
            np.clip(los2[0] / max_dist, -1, 1),
            np.clip(los2[1] / max_dist, -1, 1),
            g2_major, g2_minor, g2_angle,

            g_heading / np.pi,
            cx_norm,
            cy_norm,
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    # ── Kinematic step (matches Python env, used for validation) ──────────

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.last_cmd = action.copy()

        surge    = action[0] * self.max_cmd
        yaw_rate = action[1] * self.max_yaw_rate
        self.asv_heading += yaw_rate * self.dt
        self.asv_heading  = (self.asv_heading + np.pi) % (2 * np.pi) - np.pi

        vx = surge * np.cos(self.asv_heading)
        vy = surge * np.sin(self.asv_heading)
        asv_current      = self._get_current(self.asv_pos)
        self.asv_pos     = self.asv_pos + np.array(
            [vx + asv_current[0], vy + asv_current[1]], dtype=np.float32
        ) * self.dt

        g1_current = self._get_current(self.g1_pos)
        g2_current = self._get_current(self.g2_pos)
        self.g1_pos = self.g1_pos + (self.g1_vel + g1_current) * self.dt
        self.g2_pos = self.g2_pos + (self.g2_vel + g2_current) * self.dt

        true_dist1 = self._distance(self.asv_pos, self.g1_pos)
        true_dist2 = self._distance(self.asv_pos, self.g2_pos)

        asv_current = self._get_current(self.asv_pos)
        g1_cur_ekf  = self._get_current(self.g1_pos) if true_dist1 <= self.comm_range else asv_current
        g2_cur_ekf  = self._get_current(self.g2_pos) if true_dist2 <= self.comm_range else asv_current

        self.g1_x, self.g1_P = self._ekf_predict(self.g1_x, self.g1_P, g1_cur_ekf)
        self.g2_x, self.g2_P = self._ekf_predict(self.g2_x, self.g2_P, g2_cur_ekf)

        if true_dist1 <= self.comm_range:
            self.g1_x, self.g1_P = self._ekf_update(self.g1_x, self.g1_P, self.g1_pos)
        if true_dist2 <= self.comm_range:
            self.g2_x, self.g2_P = self._ekf_update(self.g2_x, self.g2_P, self.g2_pos)

        self._sync_estimates()
        self.step_count += 1

        return self.get_obs()

    # ── Gazebo-mode entry point (step 3+) ─────────────────────────────────

    def set_asv_state(self, pos: np.ndarray, heading: float, last_cmd: np.ndarray):
        """Inject real ASV state; does NOT tick gliders or EKF."""
        self.asv_pos     = np.asarray(pos,      dtype=np.float32)
        self.asv_heading = float(heading)
        self.last_cmd    = np.asarray(last_cmd,  dtype=np.float32)

    def step_gazebo(self, gz_pos: np.ndarray, gz_yaw: float, action: np.ndarray):
        """
        Gazebo mode: ASV position/heading come from GPS+IMU;
        glider dynamics and EKF still run with the training config's dt.
        """
        self.asv_pos     = np.asarray(gz_pos, dtype=np.float32)
        self.asv_heading = float(gz_yaw)
        self.last_cmd    = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        g1_current = self._get_current(self.g1_pos)
        g2_current = self._get_current(self.g2_pos)
        self.g1_pos = self.g1_pos + (self.g1_vel + g1_current) * self.dt
        self.g2_pos = self.g2_pos + (self.g2_vel + g2_current) * self.dt

        true_dist1 = self._distance(self.asv_pos, self.g1_pos)
        true_dist2 = self._distance(self.asv_pos, self.g2_pos)

        asv_current = self._get_current(self.asv_pos)
        g1_cur_ekf  = self._get_current(self.g1_pos) if true_dist1 <= self.comm_range else asv_current
        g2_cur_ekf  = self._get_current(self.g2_pos) if true_dist2 <= self.comm_range else asv_current

        self.g1_x, self.g1_P = self._ekf_predict(self.g1_x, self.g1_P, g1_cur_ekf)
        self.g2_x, self.g2_P = self._ekf_predict(self.g2_x, self.g2_P, g2_cur_ekf)

        if true_dist1 <= self.comm_range:
            self.g1_x, self.g1_P = self._ekf_update(self.g1_x, self.g1_P, self.g1_pos)
        if true_dist2 <= self.comm_range:
            self.g2_x, self.g2_P = self._ekf_update(self.g2_x, self.g2_P, self.g2_pos)

        self._sync_estimates()
        self.step_count += 1
        return self.get_obs()
