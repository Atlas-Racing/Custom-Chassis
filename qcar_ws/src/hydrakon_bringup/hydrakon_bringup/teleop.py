#!/usr/bin/env python3

import sys
import termios
import tty
import select

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped

HELP = """
Ackermann Teleop
----------------
  w / s  : increase / decrease speed
  a / d  : steer left / right
  space  : stop (zero speed, zero steering)
  q      : quit

Speed step  : 0.1   (shift = x5)  range [-1, 1]
Steer step  : 0.05  (shift = x5)  range [-1, 1]
"""


class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_node')

        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('max_steer', 1.0)
        self.declare_parameter('speed_step', 0.1)
        self.declare_parameter('steer_step', 0.05)
        self.declare_parameter('steering_offset', -0.250)

        self.max_speed = self.get_parameter('max_speed').value
        self.max_steer = self.get_parameter('max_steer').value
        self.speed_step = self.get_parameter('speed_step').value
        self.steer_step = self.get_parameter('steer_step').value
        self.steering_offset = self.get_parameter('steering_offset').value

        self.speed = 0.0
        self.steer = 0.0

        self.pub = self.create_publisher(AckermannDriveStamped, '/ackermann_cmd', 10)

        rate = self.get_parameter('publish_rate').value
        self.create_timer(1.0 / rate, self._publish)

        # Send one-shot neutral correction on startup to home the wheels
        # to physical zero before accepting any key input.
        self._publish_neutral()

    def _publish_neutral(self):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = 0.0
        msg.drive.steering_angle = self.steering_offset
        self.pub.publish(msg)

    def _publish(self):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = self.speed
        # Hardware steers right for positive steering_angle, so invert here to keep
        # 'a' (positive self.steer) producing a physical left turn.
        msg.drive.steering_angle = -self.steer
        self.pub.publish(msg)

    def _clamp(self, val, limit):
        return max(-limit, min(limit, val))

    def handle_key(self, key):
        if key == 'w':
            self.speed = self._clamp(self.speed + self.speed_step, self.max_speed)
        elif key == 's':
            self.speed = self._clamp(self.speed - self.speed_step, self.max_speed)
        elif key == 'a':
            self.steer = self._clamp(self.steer + self.steer_step, self.max_steer)
        elif key == 'd':
            self.steer = self._clamp(self.steer - self.steer_step, self.max_steer)
        elif key == 'W':
            self.speed = self._clamp(self.speed + self.speed_step * 5, self.max_speed)
        elif key == 'S':
            self.speed = self._clamp(self.speed - self.speed_step * 5, self.max_speed)
        elif key == 'A':
            self.steer = self._clamp(self.steer + self.steer_step * 5, self.max_steer)
        elif key == 'D':
            self.steer = self._clamp(self.steer - self.steer_step * 5, self.max_steer)
        elif key == ' ':
            self.speed = 0.0
            self.steer = 0.0

        self.get_logger().info(
            f'speed={self.speed:+.2f}  steer={self.steer:+.3f}',
            throttle_duration_sec=0.1,
        )


def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    settings = termios.tcgetattr(sys.stdin)

    print(HELP)

    try:
        while rclpy.ok():
            key = get_key(settings)
            if key == 'q':
                break
            if key:
                node.handle_key(key)
            rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        # Publish a stop command before exiting
        stop = AckermannDriveStamped()
        stop.header.stamp = node.get_clock().now().to_msg()
        stop.drive.speed = 0.0
        stop.drive.steering_angle = 0.0
        node.pub.publish(stop)

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
