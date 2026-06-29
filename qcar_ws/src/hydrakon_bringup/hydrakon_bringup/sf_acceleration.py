#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PointStamped
import math
import numpy as np

class AccelerationControllerNode(Node):
    def __init__(self):
        super().__init__('accel_controller_node')
        self.get_logger().info("Acceleration PP Started")

        self.sprint_distance = 80.0
        self.stop_distance = 100.0
        
        self.min_lookahead = 10.0
        self.lookahead_gain = 0.45
        self.max_lookahead = 28.0
        
        self.roi_width = 3.0       # Ignore cones outside 3 meters in width
        self.vision_horizon = 30.0 # only consider cones within 30 meters ahead
        self.track_width = 3.0 

        self.target_throttle = 1.0 
        self.launch_base_throttle = 0.40 #startthrottle 
        self.throttle_gain = 0.08        # Add 8% throttle per 1 m/s of speed
        
        self.max_steer = 0.3     
        self.steering_gain = 1.5
        self.wheelbase = 2.8

        self.start_pose = None
        self.dist_traveled = 0.0
        self.current_speed = 0.0
        self.mission_complete = False
        self.braking_active = False

        self.marker_sub = self.create_subscription(MarkerArray, '/camera/cone_markers', self.marker_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.target_pub = self.create_publisher(PointStamped, '/pure_pursuit/target', 10)

    def odom_callback(self, msg):
        # Calculate current speed from odometry
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed = math.hypot(vx, vy)

        pose = msg.pose.pose.position
        
        if self.start_pose is None:
            self.start_pose = pose

        # calculate distance travelled 
        self.dist_traveled = math.hypot(pose.x - self.start_pose.x, pose.y - self.start_pose.y)

        # braking logic
        if not self.braking_active and self.dist_traveled >= self.sprint_distance:
            self.braking_active = True
            self.get_logger().warn(f"Braking.")

        if self.braking_active and (self.dist_traveled >= self.stop_distance or (self.current_speed < 0.1 and self.dist_traveled > self.sprint_distance)):
            if not self.mission_complete:
                self.get_logger().info("Vehicle Stopped Successfully.")
                self.mission_complete = True
                self.stop_car()

    def marker_callback(self, msg):
        if self.mission_complete:
            self.stop_car()
            return

        blues, yellows = [], []
        
        for m in msg.markers:
            x, y = m.pose.position.x, m.pose.position.y

            if m.color.r > 0.8 and 0.2 < m.color.g < 0.8 and m.color.b < 0.3:
                if 0.0 < x < 5.0:
                    if not self.braking_active:
                        self.braking_active = True
                        self.get_logger().warn(f"OPTICAL FAIL-SAFE: Orange cone detected at {x:.2f}m. BRAKING!")
            
            if 0.0 < x < self.vision_horizon and abs(y) < self.roi_width:
                pos = np.array([x, y])
                if m.color.b > 0.8 and m.color.r < 0.2:
                    blues.append(pos)
                elif m.color.r > 0.8 and m.color.g > 0.8:
                    yellows.append(pos)

        # Dynamic Lookahead
        Ld = self.min_lookahead + (self.lookahead_gain * self.current_speed)
        Ld = min(Ld, self.max_lookahead)
        target = None

        # Find closest cones to the lookahead distance
        def get_closest(cones):
            if not cones: return None
            return min(cones, key=lambda p: abs(p[0] - Ld))

        best_blue = get_closest(blues)
        best_yellow = get_closest(yellows)

        if best_blue is not None and best_yellow is not None:
            target = (best_blue + best_yellow) / 2.0
        elif best_blue is not None:
            target = best_blue + np.array([0.0, -self.track_width / 2.0])
        elif best_yellow is not None:
            target = best_yellow + np.array([0.0, self.track_width / 2.0])

        self.drive_to_target(target, Ld)

    def drive_to_target(self, target, Ld):
        twist = Twist()

        # If we passed 75m or hit the optical fail-safe, slam the brakes
        if self.braking_active:
            twist.linear.x = -1.0  
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            return

        # pure pursuit, i am taking the formula and info from (https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/PurePursuit.html)
        if target is not None:
            x, y = float(target[0]), float(target[1])
            dist = math.hypot(x, y)
            L_d_actual = max(self.min_lookahead, dist)

            steering_angle = math.atan((2.0 * self.wheelbase * y) / (L_d_actual * L_d_actual))
            steering_angle *= self.steering_gain
            steering_angle = max(-self.max_steer, min(self.max_steer, steering_angle))
            
            twist.angular.z = -steering_angle
            
            t_msg = PointStamped()
            t_msg.header.stamp = self.get_clock().now().to_msg()
            t_msg.header.frame_id = "base_link"
            t_msg.point.x, t_msg.point.y = x, y
            self.target_pub.publish(t_msg)
        else:
            twist.angular.z = 0.0

        # Ramping
        # Calculating throttle based on current speed
        dynamic_throttle = self.launch_base_throttle + (self.current_speed * self.throttle_gain)
        
        # Apply the throttle, but never exceed target_throttle
        twist.linear.x = min(self.target_throttle, dynamic_throttle)
        
        self.cmd_vel_pub.publish(twist)

    def stop_car(self):
        t = Twist()
        t.linear.x = 0.0
        t.angular.z = 0.0
        self.cmd_vel_pub.publish(t)

def main(args=None):
    rclpy.init(args=args)
    node = AccelerationControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_car()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()