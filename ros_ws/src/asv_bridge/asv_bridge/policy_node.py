"""
Policy Node — step 3a (kinematic) and 3b (real physics).

Subscribes to /asv_bridge/obs_normalized (VecNorm-applied 20-D obs from bridge_node),
runs the trained SAC actor deterministically, and publishes:

  /asv_bridge/action           Float32MultiArray  [surge, yaw_rate] in [-1, 1]
  /wamv/thrusters/left/thrust  std_msgs/Float64   Newtons
  /wamv/thrusters/right/thrust std_msgs/Float64   Newtons

Inference uses a plain-numpy re-implementation of the SAC actor MLP so the
container does not need SB3 or a matching numpy version.
Run extract_actor.py on the host first to produce actor_weights.npz.

Launch:
  ros2 run asv_bridge policy_node --ros-args \
      -p actor_weights:=/work/reference/trained_policy/actor_weights.npz
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32MultiArray, MultiArrayDimension


# WAM-V thruster geometry
_THRUSTER_HALF_SPAN = 1.03
_THRUST_MAX_N       = 150.0


_ACT_FNS = {
    "ReLU":  lambda x: np.maximum(0.0, x),
    "Tanh":  np.tanh,
    "ELU":   lambda x: np.where(x > 0, x, np.exp(x) - 1.0),
    "GELU":  lambda x: x * 0.5 * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3))),
}


class SACActorNumpy:
    """
    Deterministic SAC actor using only numpy.
    Loads weights produced by extract_actor.py (no SB3 / no pickle).
    Forward pass: obs → MLP(ReLU) → tanh(mu) → action in [-1, 1].
    """

    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        n = int(d["n_layers"])
        acts = list(d["act_names"])
        self._layers = [
            (d[f"mlp_w{i}"].astype(np.float32),
             d[f"mlp_b{i}"].astype(np.float32),
             _ACT_FNS.get(acts[i] if i < len(acts) else "ReLU",
                          _ACT_FNS["ReLU"]))
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


def _differential_thrust(action, max_cmd=2.5, max_yaw_rate=0.05):
    surge    = float(action[0]) * max_cmd
    yaw_rate = float(action[1]) * max_yaw_rate
    k_surge  = _THRUST_MAX_N / max_cmd
    k_yaw    = _THRUST_MAX_N / (max_yaw_rate * _THRUSTER_HALF_SPAN * 2)
    left_N   = np.clip(surge * k_surge + yaw_rate * _THRUSTER_HALF_SPAN * k_yaw,
                       -_THRUST_MAX_N, _THRUST_MAX_N)
    right_N  = np.clip(surge * k_surge - yaw_rate * _THRUSTER_HALF_SPAN * k_yaw,
                       -_THRUST_MAX_N, _THRUST_MAX_N)
    return float(left_N), float(right_N)


class PolicyNode(Node):

    def __init__(self):
        super().__init__("asv_policy")

        self.declare_parameter(
            "actor_weights",
            "/work/reference/trained_policy/actor_weights.npz",
        )
        npz_path = self.get_parameter("actor_weights").value

        self.get_logger().info(f"Loading actor weights from {npz_path} ...")
        try:
            self._actor = SACActorNumpy(npz_path)
            self.get_logger().info("Actor loaded (numpy inference, no SB3 required).")
        except Exception as e:
            self.get_logger().error(f"Failed to load actor: {e}")
            self._actor = None

        self._obs_sub = self.create_subscription(
            Float32MultiArray,
            "/asv_bridge/obs_normalized",
            self._obs_callback,
            10,
        )
        self._action_pub = self.create_publisher(
            Float32MultiArray, "/asv_bridge/action", 10
        )
        self._left_pub  = self.create_publisher(Float64, "/wamv/thrusters/left/thrust",  10)
        self._right_pub = self.create_publisher(Float64, "/wamv/thrusters/right/thrust", 10)

        self._step = 0

    def _obs_callback(self, msg):
        if self._actor is None:
            return

        obs    = np.array(msg.data, dtype=np.float32)
        action = np.clip(self._actor.predict(obs), -1.0, 1.0).astype(np.float32)

        self._action_pub.publish(_f32_msg(action))

        left_N, right_N = _differential_thrust(action)
        self._left_pub.publish(Float64(data=left_N))
        self._right_pub.publish(Float64(data=right_N))

        self._step += 1
        if self._step % 10 == 0 or self._step <= 3:
            self.get_logger().info(
                f"step={self._step:4d}  "
                f"action=[{action[0]:+.3f}, {action[1]:+.3f}]  "
                f"thrust=[{left_N:+6.1f}, {right_N:+6.1f}] N"
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
