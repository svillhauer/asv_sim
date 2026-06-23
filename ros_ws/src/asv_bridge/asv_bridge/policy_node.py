"""
Policy Node — SAC actor + inner-loop velocity controller.

Outer loop (every 300 s, triggered by /asv_bridge/obs_normalized):
  SAC inference → [surge_cmd (m/s), yaw_rate_cmd (rad/s)]

Inner loop (10 Hz timer):
  PI(surge) + P(yaw_rate) → differential thrust
  Feedback: surge estimated from GPS, yaw_rate from IMU.

This lets the WAM-V track the commanded velocity for the full 300 s
between EKF ticks, matching the kinematic assumption the policy was
trained under (constant velocity over dt=300 s).

Publishes:
  /asv_bridge/action           Float32MultiArray  [surge, yaw_rate] in [-1,1]
  /wamv/thrusters/left/thrust  Float64  Newtons
  /wamv/thrusters/right/thrust Float64  Newtons
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Imu, NavSatFix


_R_EARTH = 6_371_000.0   # metres

_ACT_FNS = {
    "ReLU":  lambda x: np.maximum(0.0, x),
    "Tanh":  np.tanh,
    "ELU":   lambda x: np.where(x > 0, x, np.exp(x) - 1.0),
    "GELU":  lambda x: x * 0.5 * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3))),
}


class SACActorNumpy:
    """Deterministic SAC actor — plain numpy, no SB3 dependency."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        n = int(d["n_layers"])
        acts = list(d["act_names"])
        self._layers = [
            (d[f"mlp_w{i}"].astype(np.float32),
             d[f"mlp_b{i}"].astype(np.float32),
             _ACT_FNS.get(acts[i] if i < len(acts) else "ReLU", _ACT_FNS["ReLU"]))
            for i in range(n)
        ]
        self._mu_w = d["mu_w"].astype(np.float32)
        self._mu_b = d["mu_b"].astype(np.float32)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        x = obs.astype(np.float32)
        for w, b, act in self._layers:
            x = act(x @ w.T + b)
        return np.tanh(x @ self._mu_w.T + self._mu_b)


def _f32_msg(arr: np.ndarray) -> Float32MultiArray:
    msg = Float32MultiArray()
    dim = MultiArrayDimension()
    dim.label  = "action"
    dim.size   = len(arr)
    dim.stride = len(arr)
    msg.layout.dim = [dim]
    msg.data = arr.tolist()
    return msg


class PolicyNode(Node):

    # Training-env action scaling
    MAX_SURGE    = 2.5     # m/s
    MAX_YAW_RATE = 0.05    # rad/s

    # Inner-loop controller gains (tune if WAM-V under/overshoots)
    KP_SURGE     = 500.0   # N / (m/s error)
    KI_SURGE     = 100.0   # N / (m/s · s) — integrates out drag
    KP_YAW       = 1000.0  # N·m / (rad/s error)

    MAX_THRUST   = 1500.0  # N per thruster (VRX WAM-V max)
    CTRL_HZ      = 10.0    # inner-loop rate

    def __init__(self):
        super().__init__("asv_policy")

        self.declare_parameter("actor_weights",
            "/work/reference/trained_policy/actor_weights.npz")
        npz_path = self.get_parameter("actor_weights").value

        self.get_logger().info(f"Loading actor weights from {npz_path} ...")
        self._actor = SACActorNumpy(npz_path)
        self.get_logger().info("Actor loaded (numpy inference, no SB3 required).")

        # Publishers
        self._pub_action = self.create_publisher(
            Float32MultiArray, "/asv_bridge/action", 10)
        self._pub_left  = self.create_publisher(
            Float64, "/wamv/thrusters/left/thrust",  10)
        self._pub_right = self.create_publisher(
            Float64, "/wamv/thrusters/right/thrust", 10)

        # Outer-loop: policy inference triggered by new observation
        self.create_subscription(
            Float32MultiArray, "/asv_bridge/obs_normalized",
            self._obs_cb, 10)

        # Inner-loop sensors
        self.create_subscription(
            Imu,       "/wamv/sensors/imu/imu/data",    self._imu_cb, 10)
        self.create_subscription(
            NavSatFix, "/wamv/sensors/gps/gps/fix",     self._gps_cb, 10)

        # Commanded velocities (set by outer loop, held until next tick)
        self._surge_cmd   = 0.0   # m/s
        self._yaw_cmd     = 0.0   # rad/s

        # Measured state (updated by sensor callbacks)
        self._yaw_rate    = 0.0   # rad/s  (IMU)
        self._surge_meas  = 0.0   # m/s    (GPS-derived)
        self._heading     = 0.0   # rad    (IMU, for GPS projection)

        # GPS velocity estimation
        self._gps_ref  = None   # (lat0, lon0) — set on first fix
        self._gps_pos  = None   # (x, y) m  — previous position
        self._gps_time = None   # float seconds — previous timestamp

        # Surge integrator
        self._surge_int = 0.0
        self._int_limit = self.MAX_THRUST / max(self.KI_SURGE, 1e-6)

        # Inner-loop timer
        self.create_timer(1.0 / self.CTRL_HZ, self._ctrl_step)

        self._step = 0

    # ── Outer loop: policy inference ───────────────────────────────────

    def _obs_cb(self, msg: Float32MultiArray):
        obs = np.array(msg.data, dtype=np.float32)
        if obs.shape[0] != 20:
            return

        action = np.clip(self._actor.predict(obs), -1.0, 1.0).astype(np.float32)

        new_surge = float(action[0]) * self.MAX_SURGE
        new_yaw   = float(action[1]) * self.MAX_YAW_RATE

        # Reset integrator if surge command reverses direction
        if new_surge * self._surge_cmd < 0:
            self._surge_int = 0.0

        self._surge_cmd = new_surge
        self._yaw_cmd   = new_yaw
        self._step += 1

        self._pub_action.publish(_f32_msg(action))
        self.get_logger().info(
            f"step={self._step:4d}  "
            f"action=[{action[0]:+.3f}, {action[1]:+.3f}]  "
            f"cmd=[{new_surge:+.2f} m/s, {new_yaw:+.4f} rad/s]"
        )

    # ── Sensor callbacks ───────────────────────────────────────────────

    def _imu_cb(self, msg: Imu):
        self._yaw_rate = msg.angular_velocity.z
        q = msg.orientation
        self._heading = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def _gps_cb(self, msg: NavSatFix):
        lat, lon = msg.latitude, msg.longitude
        if self._gps_ref is None:
            self._gps_ref = (lat, lon)
        lat0, lon0 = self._gps_ref
        # ENU convention: x = East, y = North
        x = (lon - lon0) * math.pi / 180.0 * _R_EARTH * math.cos(math.radians(lat0))
        y = (lat - lat0) * math.pi / 180.0 * _R_EARTH

        now = self.get_clock().now().nanoseconds * 1e-9
        if self._gps_pos is not None and self._gps_time is not None:
            dt = now - self._gps_time
            if dt > 0.01:
                dx = x - self._gps_pos[0]
                dy = y - self._gps_pos[1]
                # Project world-frame velocity onto body-frame surge axis
                c, s = math.cos(self._heading), math.sin(self._heading)
                raw_surge = (dx * c + dy * s) / dt
                # Low-pass filter to reduce GPS noise (α = 0.3)
                self._surge_meas = 0.3 * raw_surge + 0.7 * self._surge_meas

        self._gps_pos  = (x, y)
        self._gps_time = now

    # ── Inner loop: 10 Hz velocity controller ─────────────────────────

    def _ctrl_step(self):
        dt = 1.0 / self.CTRL_HZ

        # Surge PI
        surge_err = self._surge_cmd - self._surge_meas
        self._surge_int = float(np.clip(
            self._surge_int + surge_err * dt,
            -self._int_limit, self._int_limit,
        ))
        f_surge = self.KP_SURGE * surge_err + self.KI_SURGE * self._surge_int

        # Yaw-rate P
        yaw_err = self._yaw_cmd - self._yaw_rate
        f_yaw   = self.KP_YAW * yaw_err

        # Differential thrust
        # left = surge_force - yaw_torque,  right = surge_force + yaw_torque
        left  = float(np.clip(f_surge - f_yaw, -self.MAX_THRUST, self.MAX_THRUST))
        right = float(np.clip(f_surge + f_yaw, -self.MAX_THRUST, self.MAX_THRUST))

        self._pub_left.publish(Float64(data=left))
        self._pub_right.publish(Float64(data=right))

        self._ctrl_count = getattr(self, "_ctrl_count", 0) + 1
        if self._ctrl_count % 50 == 0:   # every 5 s
            self.get_logger().info(
                f"ctrl: cmd=[{self._surge_cmd:+.2f} m/s, {self._yaw_cmd:+.4f} rad/s]  "
                f"meas=[{self._surge_meas:+.2f} m/s, {self._yaw_rate:+.4f} rad/s]  "
                f"hdg={math.degrees(self._heading):.1f}°  "
                f"L={left:.0f} R={right:.0f} N"
            )


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
