#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import FollowPath, ComputePathThroughPoses
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from action_msgs.msg import GoalStatus
from tf2_ros import Buffer, TransformListener
import csv
import math

class LapManager(Node):
    def __init__(self):
        super().__init__('lap_manager')

        self.declare_parameter("path_file", "my_track_path.csv")
        self.declare_parameter("waypoint_spacing", 1.5)
        self.declare_parameter("laps", 3)
        self.declare_parameter("lap_start_radius", 10.0)   #from start before finish pointish
        self.declare_parameter("lap_finish_radius", 4.0)   #return within this distance then lap completee

        self.path_file = self.get_parameter("path_file").value
        self.spacing = self.get_parameter("waypoint_spacing").value
        self.target_laps = self.get_parameter("laps").value
        self.lap_start_radius = self.get_parameter("lap_start_radius").value
        self.lap_finish_radius = self.get_parameter("lap_finish_radius").value

        self.current_lap = 0
        self.amcl_received = False
        self.nav_engaged = False
        self.current_pose = None

        #position based lap detection state
        self.start_x = None
        self.start_y = None
        self.left_start_zone = False
        self.lap_detected = False
        self.intentional_cancel = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._follow_client  = ActionClient(self, FollowPath,              'follow_path')
        self._planner_client = ActionClient(self, ComputePathThroughPoses, 'compute_path_through_poses')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.amcl_callback, 10)

        self.waypoints = self.load_and_sample_path(self.path_file, self.spacing)

        if not self.waypoints:
            self.get_logger().error("No waypoints loaded! Aborting.")
            return

        self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints. Waiting for odom...")

        self.timer = self.create_timer(1.0, self.check_start_condition)
        self.goal_handle = None

    def amcl_callback(self, msg):
        self.current_pose = msg.pose.pose

        if not self.amcl_received:
            self.get_logger().info("Odom received. Starting pose tracking.")
            self.amcl_received = True
            self.start_x = msg.pose.pose.position.x
            self.start_y = msg.pose.pose.position.y
            self.get_logger().info(f"Start/finish positioned at ({self.start_x:.2f}, {self.start_y:.2f})")
            return

        #detect laps based on position
        if not self.nav_engaged or self.lap_detected:
            return

        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        dist = math.hypot(px - self.start_x, py - self.start_y)

        if not self.left_start_zone:
            if dist > self.lap_start_radius:
                self.left_start_zone = True
                self.get_logger().info(f"left start point at (dist={dist:.1f}m), Finish point marked")
        else:
            if dist < self.lap_finish_radius:
                self.lap_detected = True
                self.handle_lap_complete()

    def handle_lap_complete(self):
        self.current_lap += 1
        self.get_logger().info(f"Lap {self.current_lap} / {self.target_laps} complete!!")

        if self.current_lap >= self.target_laps:
            self.get_logger().info("All laps doneee!!")
            self.stop_vehicle()
            if self.goal_handle is not None:
                self.intentional_cancel = True
                self.goal_handle.cancel_goal_async()
            return

        #cancel current goal to  start the next lap
        self.left_start_zone = False
        self.lap_detected = False
        if self.goal_handle is not None:
            self.intentional_cancel = True
            cancel_future = self.goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self._on_cancel_for_next_lap)
        else:
            self._send_next_lap_goal()

    def _on_cancel_for_next_lap(self, future):
        self.goal_handle = None
        self.intentional_cancel = False
        self._send_next_lap_goal()

    def _send_next_lap_goal(self):
        self.get_logger().info(f"Starting Lap {self.current_lap + 1} / {self.target_laps}")
        self._send_goal()

    def check_start_condition(self):
        if self.amcl_received and not self.nav_engaged:
            self.get_logger().info("Odom ready, waiting 15s for nav stack to activate...")
            self.nav_engaged = True
            self.timer.cancel()
            self.start_timer = self.create_timer(15.0, self.delayed_start)
        elif not self.amcl_received:
            self.get_logger().info("Waiting for /odom", throttle_duration_sec=5.0)

    def delayed_start(self):
        self.start_timer.cancel()
        self.get_logger().info(f"Starting Lap 1 / {self.target_laps}")
        self._send_goal()

    def load_and_sample_path(self, filename, spacing):
        points = []
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        points.append((float(row[0]), float(row[1])))
        except Exception as e:
            self.get_logger().error(f"Failed to read path file: {e}")
            return []

        if not points:
            return []

        sampled = []
        last_point = points[0]
        sampled.append(points[0])

        for p in points[1:]:
            dist = math.hypot(p[0] - last_point[0], p[1] - last_point[1])
            if dist >= spacing:
                sampled.append(p)
                last_point = p

        if math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1]) > 2.0:
            sampled.append(points[-1])

        poses = []
        for i, p in enumerate(sampled):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]

            if i < len(sampled) - 1:
                dx = sampled[i+1][0] - p[0]
                dy = sampled[i+1][1] - p[1]
                yaw = math.atan2(dy, dx)
            else:
                dx = sampled[0][0] - p[0]
                dy = sampled[0][1] - p[1]
                yaw = math.atan2(dy, dx)

            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            poses.append(pose)

        return poses

    def _get_map_pose(self):
        """Look up the robot's current pose in the map frame via TF."""
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            t = tf.transform.translation
            # Return a simple namespace with .position.x / .position.y
            class _Pos:
                pass
            pos = _Pos(); pos.x = t.x; pos.y = t.y
            pose = _Pos(); pose.position = pos
            return pose
        except Exception:
            return None

    def get_cycled_path(self):
        """Find nearest point and cycle the path so that point is at the start"""
        map_pose = self._get_map_pose() or self.current_pose
        if not map_pose or not self.waypoints:
            return self.waypoints

        best_idx = 0
        min_dist = float('inf')

        for i, p in enumerate(self.waypoints):
            dist = math.hypot(p.pose.position.x - map_pose.position.x,
                              p.pose.position.y - map_pose.position.y)
            if dist < min_dist:
                min_dist = dist
                best_idx = i

        start_idx = best_idx

        end_idx = (best_idx - 60) % len(self.waypoints)

        cycled = []
        curr = start_idx
        while curr != end_idx:
            cycled.append(self.waypoints[curr])
            curr = (curr + 1) % len(self.waypoints)
        cycled.append(self.waypoints[end_idx])

        return cycled

    def stop_vehicle(self):
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = 0.0
        for _ in range(5):
            self.cmd_vel_pub.publish(stop_cmd)

    def _send_goal(self):
        cycled = self.get_cycled_path()
        now = self.get_clock().now().to_msg()
        for p in cycled:
            p.header.stamp = now
        self._pending_waypoints = cycled

        if not self._planner_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('SMAC planner not available, falling back to CSV path')
            self._send_follow_path(self._csv_path_from_waypoints(cycled))
            return

        step = 10
        smac_goals = cycled[step::step]  # skip nearest waypoint — it's too close for Dubins
        if not smac_goals:
            smac_goals = [cycled[-1]]
        elif smac_goals[-1] is not cycled[-1]:
            smac_goals.append(cycled[-1])

        goal_msg = ComputePathThroughPoses.Goal()
        goal_msg.goals = smac_goals
        goal_msg.planner_id = 'GridBased'
        goal_msg.use_start = False

        self.get_logger().info(f'Requesting SMAC path through {len(smac_goals)} guide waypoints...')
        self._compute_future = self._planner_client.send_goal_async(goal_msg)
        self._compute_future.add_done_callback(self._compute_goal_response_callback)

    def _compute_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('SMAC planning goal rejected, falling back to CSV path')
            self._send_follow_path(self._csv_path_from_waypoints(self._pending_waypoints))
            return
        self._compute_result_future = goal_handle.get_result_async()
        self._compute_result_future.add_done_callback(self._compute_result_callback)

    def _compute_result_callback(self, future):
        result = future.result()
        if result.status == GoalStatus.STATUS_SUCCEEDED and len(result.result.path.poses) > 0:
            self.get_logger().info(
                f'SMAC planned {len(result.result.path.poses)} poses '
                f'in {result.result.planning_time.sec}.'
                f'{result.result.planning_time.nanosec // 1_000_000:03d}s'
            )
            self._send_follow_path(result.result.path)
        else:
            self.get_logger().warn('SMAC planning failed, falling back to CSV path')
            self._send_follow_path(self._csv_path_from_waypoints(self._pending_waypoints))

    def _csv_path_from_waypoints(self, waypoints):
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        path.poses = waypoints
        return path

    def _send_follow_path(self, path):
        if not self._follow_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('FollowPath action server not available')
            return

        goal_msg = FollowPath.Goal()
        goal_msg.path = path
        goal_msg.controller_id = 'FollowPath'
        goal_msg.goal_checker_id = 'goal_checker'
        goal_msg.progress_checker_id = 'progress_checker'

        self.get_logger().info(f'Sending {len(path.poses)} poses to RPP controller')
        self._send_goal_future = self._follow_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by controller')
            return

        self.get_logger().info('Controller accepted path — racing')
        self.goal_handle = goal_handle
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        pass

    def get_result_callback(self, future):
        if self.intentional_cancel:
            return

        result = future.result()
        status = result.status

        if status == GoalStatus.STATUS_CANCELED:
            return

        if self.current_lap >= self.target_laps:
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Path completed, resending goal')
            self._send_goal()
        else:
            self._retry_timer = self.create_timer(3.0, self._retry_send_goal)

    def _retry_send_goal(self):
        self._retry_timer.cancel()
        if self.current_lap >= self.target_laps:
            return
        self._send_goal()

def main(args=None):
    rclpy.init(args=args)
    node = LapManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
