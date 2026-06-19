import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


def quat_translation_to_matrix(q, t):
    x, y, z, w = q.x, q.y, q.z, q.w
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = [t.x, t.y, t.z]
    return m


class OdomTfBridge(Node):
    """
    Republishes ZED's odom -> zed_camera_link estimate as odom -> base_link.

    The ZED node's own TF publishing is disabled (publish_tf=false) because its
    positional-tracking child frame is hardcoded to zed_camera_link, which
    conflicts with the static upper_plate_link -> zed_camera_link transform
    owned by our URDF (a tf2 frame can only have one parent at a time).
    Composing the two transforms through tf2's own buffer hits the same
    conflict internally (zed_camera_link would get two parents - base_link
    statically, odom dynamically), so the composition is done by hand here:
    odom_to_base = odom_to_cam * inverse(base_to_cam).
    """

    def __init__(self):
        super().__init__("odom_tf_bridge")
        self.declare_parameter("odom_topic", "/zed/zed_node/odom")
        self.declare_parameter("camera_frame", "zed_camera_link")
        self.declare_parameter("base_frame", "base_link")

        self.camera_frame = self.get_parameter("camera_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.base_to_cam = None  # cached static transform, looked up lazily

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value,
            self.odom_cb, qos_profile_sensor_data,
        )

    def odom_cb(self, msg: Odometry):
        if self.base_to_cam is None:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.base_frame, self.camera_frame, rclpy.time.Time()
                )
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(
                    f"Waiting for static {self.base_frame} -> {self.camera_frame}: {e}",
                    throttle_duration_sec=2.0,
                )
                return
            self.base_to_cam = quat_translation_to_matrix(
                tf.transform.rotation, tf.transform.translation
            )

        odom_to_cam = quat_translation_to_matrix(
            msg.pose.pose.orientation, msg.pose.pose.position
        )
        odom_to_base = odom_to_cam @ np.linalg.inv(self.base_to_cam)

        # Ground-vehicle constraint: keep only x, y, yaw.
        # The full 3D matrix correctly removes the camera's 10.1° mount pitch,
        # but residual Z drift and micro-roll/pitch from ZED live tracking must
        # not propagate — a wheeled vehicle on a flat floor has no Z or tilt DOF.
        x = float(odom_to_base[0, 3])
        y = float(odom_to_base[1, 3])
        yaw = np.arctan2(odom_to_base[1, 0], odom_to_base[0, 0])
        qz = float(np.sin(yaw / 2.0))
        qw = float(np.cos(yaw / 2.0))

        out = TransformStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = msg.header.frame_id
        out.child_frame_id = self.base_frame
        out.transform.translation.x = x
        out.transform.translation.y = y
        out.transform.translation.z = 0.0
        out.transform.rotation.x = 0.0
        out.transform.rotation.y = 0.0
        out.transform.rotation.z = qz
        out.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(out)


def main(args=None):
    rclpy.init(args=args)
    node = OdomTfBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
