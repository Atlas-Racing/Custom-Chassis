import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from zed_msgs.msg import PosTrackStatus

POSITION_SANITY_LIMIT_M = 100.0  # a QCar will never legitimately be this far from origin

STATUS_NAMES = {0: "OK", 1: "UNAVAILABLE/LOOP_CLOSED", 2: "SEARCHING", 3: "OFF"}


def yaw_pitch_roll_deg(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


class StateMonitor(Node):
    def __init__(self):
        super().__init__("state_monitor")
        self.declare_parameter("odom_topic", "/zed/zed_node/odom")
        self.declare_parameter("imu_topic", "/zed/zed_node/imu/data")
        self.declare_parameter("pose_topic", "/zed/zed_node/pose")
        self.declare_parameter("pose_status_topic", "/zed/zed_node/pose/status")
        self.declare_parameter("print_rate_hz", 2.0)

        self.latest = {"odom": None, "imu": None, "pose": None, "pose_status": None}

        self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value,
            lambda msg: self.latest.update(odom=msg), qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu, self.get_parameter("imu_topic").value,
            lambda msg: self.latest.update(imu=msg), qos_profile_sensor_data,
        )
        self.create_subscription(
            PoseStamped, self.get_parameter("pose_topic").value,
            lambda msg: self.latest.update(pose=msg), qos_profile_sensor_data,
        )
        self.create_subscription(
            PosTrackStatus, self.get_parameter("pose_status_topic").value,
            lambda msg: self.latest.update(pose_status=msg), qos_profile_sensor_data,
        )

        period = 1.0 / float(self.get_parameter("print_rate_hz").value)
        self.create_timer(period, self.print_summary)

    def print_summary(self):
        lines = ["=" * 64]
        odom = self.latest["odom"]
        if odom is not None:
            p = odom.pose.pose.position
            yaw, pitch, roll = yaw_pitch_roll_deg(odom.pose.pose.orientation)
            lin = odom.twist.twist.linear
            ang = odom.twist.twist.angular
            flag = " ** SUSPECT (>%.0fm) **" % POSITION_SANITY_LIMIT_M if max(
                abs(p.x), abs(p.y), abs(p.z)
            ) > POSITION_SANITY_LIMIT_M else ""
            lines.append(
                f"ODOM  [{odom.header.frame_id} -> {odom.child_frame_id}] "
                f"pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) m{flag}"
            )
            lines.append(
                f"      yaw={yaw:.1f} pitch={pitch:.1f} roll={roll:.1f} deg  "
                f"v_lin=({lin.x:.2f},{lin.y:.2f},{lin.z:.2f}) v_ang=({ang.x:.2f},{ang.y:.2f},{ang.z:.2f})"
            )
        else:
            lines.append("ODOM  <no message yet>")

        pose = self.latest["pose"]
        if pose is not None:
            p = pose.pose.position
            yaw, pitch, roll = yaw_pitch_roll_deg(pose.pose.orientation)
            flag = " ** SUSPECT (>%.0fm) **" % POSITION_SANITY_LIMIT_M if max(
                abs(p.x), abs(p.y), abs(p.z)
            ) > POSITION_SANITY_LIMIT_M else ""
            lines.append(
                f"POSE  [{pose.header.frame_id}] pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) m{flag} "
                f"yaw={yaw:.1f} pitch={pitch:.1f} roll={roll:.1f} deg"
            )
        else:
            lines.append("POSE  <no message yet>")

        status = self.latest["pose_status"]
        if status is not None:
            odo_s = STATUS_NAMES.get(status.odometry_status, status.odometry_status)
            mem_s = STATUS_NAMES.get(status.spatial_memory_status, status.spatial_memory_status)
            lines.append(f"POSE STATUS  odometry={odo_s}  spatial_memory={mem_s}")
        else:
            lines.append("POSE STATUS  <no message yet>")

        imu = self.latest["imu"]
        if imu is not None:
            yaw, pitch, roll = yaw_pitch_roll_deg(imu.orientation)
            av, la = imu.angular_velocity, imu.linear_acceleration
            lines.append(
                f"IMU   yaw={yaw:.1f} pitch={pitch:.1f} roll={roll:.1f} deg  "
                f"gyro=({av.x:.3f},{av.y:.3f},{av.z:.3f}) rad/s  "
                f"accel=({la.x:.2f},{la.y:.2f},{la.z:.2f}) m/s^2"
            )
        else:
            lines.append("IMU   <no message yet>")

        self.get_logger().info("\n".join(lines))


def main(args=None):
    rclpy.init(args=args)
    node = StateMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
