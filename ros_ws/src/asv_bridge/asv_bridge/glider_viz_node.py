"""
GliderVizNode — spawns and teleports two Slocum glider models in Gazebo.

Subscribes to:
  /asv_bridge/glider_poses  geometry_msgs/PoseArray  [glider1, glider2]

On first message: spawns both glider models in Gazebo.
On subsequent messages: teleports them to updated EKF positions.

Requires:
  - Gazebo running the open_ocean world (or world_name parameter)
  - /work/models/glider_slocum/model.sdf accessible in the container
  - UserCommands plugin loaded in the world (provides create + set_pose services)
"""

import subprocess
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy

_LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
)


class GliderVizNode(Node):

    def __init__(self):
        super().__init__("glider_viz")

        self.declare_parameter("world_name",   "open_ocean")
        self.declare_parameter("model_sdf",    "/work/models/glider_slocum/model.sdf")
        self.declare_parameter("glider1_name", "glider1")
        self.declare_parameter("glider2_name", "glider2")

        self._world  = self.get_parameter("world_name").value
        self._sdf    = self.get_parameter("model_sdf").value
        self._name1  = self.get_parameter("glider1_name").value
        self._name2  = self.get_parameter("glider2_name").value
        self._spawned = False

        self.create_subscription(
            PoseArray, "/asv_bridge/glider_poses", self._pose_cb, _LATCHED_QOS
        )
        self.get_logger().info(
            f"GliderViz ready — world={self._world}  sdf={self._sdf}"
        )

    def _pose_cb(self, msg: PoseArray):
        if len(msg.poses) < 2:
            return

        p1 = msg.poses[0]
        p2 = msg.poses[1]
        x1, y1, z1 = p1.position.x, p1.position.y, p1.position.z
        x2, y2, z2 = p2.position.x, p2.position.y, p2.position.z

        if not self._spawned:
            ok1 = self._spawn(self._name1, x1, y1, z1)
            ok2 = self._spawn(self._name2, x2, y2, z2)
            if ok1 and ok2:
                self._spawned = True
                self.get_logger().info(
                    f"Gliders spawned: "
                    f"{self._name1}=({x1:.0f}, {y1:.0f})  "
                    f"{self._name2}=({x2:.0f}, {y2:.0f})"
                )
        else:
            self._set_pose(self._name1, x1, y1, z1)
            self._set_pose(self._name2, x2, y2, z2)
            self.get_logger().debug(
                f"Gliders moved: "
                f"{self._name1}=({x1:.0f}, {y1:.0f})  "
                f"{self._name2}=({x2:.0f}, {y2:.0f})"
            )

    def _gz_service(self, service, reqtype, reptype, req_str):
        cmd = [
            "gz", "service",
            "-s", service,
            "--reqtype", reqtype,
            "--reptype", reptype,
            "--timeout", "5000",
            "--req", req_str,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                self.get_logger().warn(
                    f"gz service {service} returned non-zero: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
                return False
            return True
        except subprocess.TimeoutExpired:
            self.get_logger().warn(f"gz service {service} timed out (10 s)")
            return False
        except FileNotFoundError:
            self.get_logger().error(
                "'gz' binary not found — is Gazebo sourced in this shell?"
            )
            return False

    def _spawn(self, name, x, y, z):
        req = (
            f'sdf_filename: "{self._sdf}" '
            f'name: "{name}" '
            f'pose {{ position {{ x: {x:.3f} y: {y:.3f} z: {z:.3f} }} }}'
        )
        return self._gz_service(
            f"/world/{self._world}/create",
            "gz.msgs.EntityFactory",
            "gz.msgs.Boolean",
            req,
        )

    def _set_pose(self, name, x, y, z):
        req = (
            f'name: "{name}" '
            f'position {{ x: {x:.3f} y: {y:.3f} z: {z:.3f} }}'
        )
        self._gz_service(
            f"/world/{self._world}/set_pose",
            "gz.msgs.Pose",
            "gz.msgs.Boolean",
            req,
        )


def main(args=None):
    rclpy.init(args=args)
    node = GliderVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
