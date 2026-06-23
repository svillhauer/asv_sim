"""
ASV Bridge Node.

Kinematic mode (default):  EkfCore drives ASV position internally.
Gazebo mode (gazebo_mode:=True):  real WAM-V pose from GPS + IMU.

Publishes:
  /asv_bridge/obs_raw        Float32MultiArray  20-D raw observation
  /asv_bridge/obs_normalized Float32MultiArray  VecNormalize-applied observation

Subscribes to:
  /asv_bridge/action                 Float32MultiArray  from policy_node
  /wamv/sensors/gps/gps/fix          NavSatFix          (gazebo mode)
  /wamv/sensors/imu/imu/data         Imu                (gazebo mode)
"""

import math
import pickle
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import PoseArray, Pose as GmPose
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy

_LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
)

_R_EARTH = 6_371_000.0   # metres


def _make_f32_msg(arr: np.ndarray) -> Float32MultiArray:
    msg = Float32MultiArray()
    dim = MultiArrayDimension()
    dim.label  = "obs"
    dim.size   = len(arr)
    dim.stride = len(arr)
    msg.layout.dim = [dim]
    msg.data = arr.tolist()
    return msg


class BridgeNode(Node):

    def __init__(self):
        super().__init__("asv_bridge")

        self.declare_parameter("seed",        42)
        self.declare_parameter("vecnorm_pkl",
            "/work/reference/trained_policy/vec_normalize_20km_ciric_version1.pkl")
        self.declare_parameter("gazebo_mode", False)
        self.declare_parameter("tick_period",  1.0)   # seconds; use 300.0 in gazebo_mode

        seed         = self.get_parameter("seed").value
        pkl_path     = self.get_parameter("vecnorm_pkl").value
        self._gazebo = self.get_parameter("gazebo_mode").value
        tick_period  = self.get_parameter("tick_period").value

        from asv_bridge.ekf_core import EkfCore
        self._core = EkfCore()
        obs = self._core.reset(seed=seed)
        self.get_logger().info(f"EkfCore initialised  seed={seed}  obs[:4]={obs[:4]}")

        # VecNormalize — try .npz first (no SB3 needed), fall back to pkl
        self._vec_mean = None
        self._vec_var  = None
        self._vec_clip = 10.0
        self._vec_eps  = 1e-8
        npz_path = (pkl_path.replace(".pkl", "_arrays.npz")
                    if pkl_path.endswith(".pkl") else pkl_path + "_arrays.npz")
        try:
            if __import__("os").path.exists(npz_path):
                d = np.load(npz_path)
                self._vec_mean = d["obs_mean"].astype(np.float32)
                self._vec_var  = d["obs_var"].astype(np.float32)
                self._vec_clip = float(d["clip_obs"])
                self._vec_eps  = float(d["epsilon"])
                self.get_logger().info(
                    f"VecNormalize loaded from .npz  clip={self._vec_clip}  "
                    f"mean[:4]={self._vec_mean[:4]}"
                )
            else:
                with open(pkl_path, "rb") as f:
                    vn = pickle.load(f)
                self._vec_mean = vn.obs_rms.mean.astype(np.float32)
                self._vec_var  = vn.obs_rms.var.astype(np.float32)
                if hasattr(vn, "clip_obs"):
                    self._vec_clip = float(vn.clip_obs)
                self.get_logger().info(
                    f"VecNormalize loaded from .pkl  clip={self._vec_clip}  "
                    f"mean[:4]={self._vec_mean[:4]}"
                )
        except Exception as e:
            self.get_logger().warn(f"Could not load VecNormalize: {e}  "
                                   "obs_normalized will equal obs_raw.")

        self._pub_raw     = self.create_publisher(Float32MultiArray, "/asv_bridge/obs_raw",       10)
        self._pub_norm    = self.create_publisher(Float32MultiArray, "/asv_bridge/obs_normalized", 10)
        self._pub_gliders = self.create_publisher(PoseArray, "/asv_bridge/glider_poses", _LATCHED_QOS)

        self._latest_action = np.array([0.0, 0.0], dtype=np.float32)
        self.create_subscription(
            Float32MultiArray, "/asv_bridge/action", self._action_callback, 10)

        # Gazebo pose from GPS + IMU
        self._gz_pos        = np.zeros(2, dtype=np.float32)
        self._gz_yaw        = 0.0
        self._gps_ref       = None   # (lat0, lon0) set on first fix
        self._gps_ready     = False
        self._imu_ready     = False

        if self._gazebo:
            self.create_subscription(
                NavSatFix,
                "/wamv/sensors/gps/gps/fix",
                self._gps_callback,
                10,
            )
            self.create_subscription(
                Imu,
                "/wamv/sensors/imu/imu/data",
                self._imu_callback,
                10,
            )
            self.get_logger().info(
                "Gazebo mode ON — subscribing to GPS + IMU"
            )

        self._timer = self.create_timer(tick_period, self._tick)
        self.get_logger().info(f"Tick period: {tick_period} s")

        # Publish initial glider positions immediately so glider_viz_node can
        # spawn the models as soon as Gazebo is ready — don't wait for first tick.
        self._pub_gliders.publish(self._glider_pose_msg())

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _action_callback(self, msg):
        self._latest_action = np.clip(
            np.array(msg.data, dtype=np.float32), -1.0, 1.0
        )

    def _gps_callback(self, msg: NavSatFix):
        lat, lon = msg.latitude, msg.longitude
        if self._gps_ref is None:
            self._gps_ref = (lat, lon)
            self.get_logger().info(
                f"GPS reference set: lat={lat:.6f}  lon={lon:.6f}"
            )
        lat0, lon0 = self._gps_ref
        self._gz_pos[0] = (lat - lat0) * math.pi / 180.0 * _R_EARTH
        self._gz_pos[1] = (lon - lon0) * math.pi / 180.0 * _R_EARTH * math.cos(math.radians(lat0))
        self._gps_ready = True

    def _imu_callback(self, msg: Imu):
        q = msg.orientation
        self._gz_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self._imu_ready = True

    def _tick(self):
        action = self._latest_action.copy()

        if self._gazebo:
            if not (self._gps_ready and self._imu_ready):
                self.get_logger().warn(
                    f"Waiting for GPS ({self._gps_ready}) and IMU ({self._imu_ready}) …",
                    throttle_duration_sec=5.0,
                )
                return
            obs_raw = self._core.step_gazebo(self._gz_pos, self._gz_yaw, action)
            self.get_logger().info(
                f"step={self._core.step_count}  "
                f"gz_pos=[{self._gz_pos[0]:.2f}, {self._gz_pos[1]:.2f}]  "
                f"gz_yaw={self._gz_yaw:.3f}"
            )
        else:
            obs_raw = self._core.step(action)

        self._pub_raw.publish(_make_f32_msg(obs_raw))

        if self._vec_mean is not None:
            obs_norm = np.clip(
                (obs_raw - self._vec_mean) / np.sqrt(self._vec_var + self._vec_eps),
                -self._vec_clip, self._vec_clip,
            ).astype(np.float32)
        else:
            obs_norm = obs_raw.copy()

        self._pub_norm.publish(_make_f32_msg(obs_norm))
        self._pub_gliders.publish(self._glider_pose_msg())

    def _glider_pose_msg(self) -> PoseArray:
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        for pos in (self._core.g1_pos, self._core.g2_pos):
            p = GmPose()
            p.position.x = float(pos[0])
            p.position.y = float(pos[1])
            p.position.z = 0.0   # render at surface; true depth is ~100 m
            p.orientation.w = 1.0
            msg.poses.append(p)
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
