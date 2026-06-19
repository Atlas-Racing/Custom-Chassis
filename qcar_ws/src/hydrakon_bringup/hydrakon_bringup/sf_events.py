#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

import numpy as np
import math
import threading
import time

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Twist, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs


# ==============================================================================
# SKIDPAD PLANNER NODE
# ==============================================================================

class SkidpadPlanner(Node):

    def __init__(self):
        super().__init__('skidpad_planner')

        self.get_logger().info("Skidpad Planner Started (With Rolling Window)")

        self.car_x = 0.0
        self.car_y = 0.0
        self.car_yaw = 0.0

        self.path_generated = False
        self.current_index = 0

        self.global_path = None
        self.global_cones = []

        self.create_subscription(Odometry, '/odom', self.pose_callback, 10)
        self.path_pub = self.create_publisher(Path, '/received_global_plan', 10)
        self.full_path_pub = self.create_publisher(Path, '/skidpad_full_path', 10)
        self.cone_pub = self.create_publisher(MarkerArray, '/skidpad_ghost_cones', 10)

        self.create_timer(0.1, self.publish_loop)

    def pose_callback(self, msg):

        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        self.car_yaw = math.atan2(siny_cosp, cosy_cosp)

        if not self.path_generated:
            self.global_path, self.global_cones = self.generate_skidpad(self.car_x, self.car_y, self.car_yaw)
            self.path_generated = True
            self.get_logger().info("Generated skidpad map")

    def generate_skidpad(self, car_x, car_y, car_yaw):

        density = 0.20
        center_distance = 18.25

        inner_radius = 15.25 / 2.0
        outer_radius = 21.25 / 2.0
        driving_radius = (inner_radius + outer_radius) / 2.0

        entry_length = 15.0
        exit_length = 20.0

        entry_pts = int(entry_length / density)
        entry = np.stack((np.zeros(entry_pts), np.linspace(-entry_length, 0.0, entry_pts, endpoint=False)), axis=1)

        right_pts = int((2.0 * np.pi * driving_radius * 2.0) / density)
        r_angles = np.linspace(np.pi, -3.0 * np.pi, right_pts, endpoint=False)
        right_circle = np.stack((center_distance / 2.0 + driving_radius * np.cos(r_angles), driving_radius * np.sin(r_angles)), axis=1)

        left_pts = int((2.0 * np.pi * driving_radius * 2.0) / density)
        l_angles = np.linspace(0.0, 4.0 * np.pi, left_pts, endpoint=False)
        left_circle = np.stack((-center_distance / 2.0 + driving_radius * np.cos(l_angles), driving_radius * np.sin(l_angles)), axis=1)

        exit_pts = int(exit_length / density)
        exit_path = np.stack((np.zeros(exit_pts), np.linspace(0.0, exit_length, exit_pts)), axis=1)

        base_path = np.vstack((entry, right_circle, left_circle, exit_path))

        base_cones = []

        for a in np.linspace(-3*np.pi/4, 3*np.pi/4, 13):
            base_cones.append({'x': center_distance/2.0 + outer_radius*np.cos(a), 'y': outer_radius*np.sin(a), 'color': 'blue'})
        for a in np.linspace(0, 2*np.pi, 16, endpoint=False):
            base_cones.append({'x': center_distance/2.0 + inner_radius*np.cos(a), 'y': inner_radius*np.sin(a), 'color': 'yellow'})
        for a in np.linspace(np.pi/4, 7*np.pi/4, 13):
            base_cones.append({'x': -center_distance/2.0 + outer_radius*np.cos(a), 'y': outer_radius*np.sin(a), 'color': 'yellow'})
        for a in np.linspace(0, 2*np.pi, 16, endpoint=False):
            base_cones.append({'x': -center_distance/2.0 + inner_radius*np.cos(a), 'y': inner_radius*np.sin(a), 'color': 'blue'})
        for y in np.linspace(-15.0, -2.0, 6):
            base_cones.append({'x': -1.5, 'y': y, 'color': 'yellow'})
            base_cones.append({'x': 1.5, 'y': y, 'color': 'blue'})
        for y in np.linspace(2.0, 25.0, 8):
            base_cones.append({'x': -1.5, 'y': y, 'color': 'yellow'})
            base_cones.append({'x': 1.5, 'y': y, 'color': 'blue'})

        rot_angle = car_yaw - (np.pi / 2.0)
        rot_matrix = np.array([
            [np.cos(rot_angle), -np.sin(rot_angle)],
            [np.sin(rot_angle),  np.cos(rot_angle)]
        ])

        rotated_path = np.dot(base_path, rot_matrix.T)
        offset = np.array([car_x, car_y]) - rotated_path[0]
        final_path = rotated_path + offset

        final_cones = []
        for c in base_cones:
            p = np.array([c['x'], c['y']])
            rp = np.dot(p, rot_matrix.T) + offset
            final_cones.append({'x': rp[0], 'y': rp[1], 'color': c['color']})

        return final_path, final_cones

    def publish_loop(self):

        if not self.path_generated:
            return

        # for rviz, the full path
        full_msg = Path()
        full_msg.header.frame_id = "odom"
        full_msg.header.stamp = self.get_clock().now().to_msg()

        for p in self.global_path:
            pose = PoseStamped()
            pose.pose.position.x = float(p[0])
            pose.pose.position.y = float(p[1])
            full_msg.poses.append(pose)

        self.full_path_pub.publish(full_msg)

        # for controller
        car = np.array([self.car_x, self.car_y])
        path_length = len(self.global_path)

        if self.current_index < path_length - 1:
            # limit our vision to smaller track segment
            search_window = 75
            end_search = min(path_length, self.current_index + search_window)
            local_segment = self.global_path[self.current_index : end_search]

            # Find the closest point in this segment
            local_distances = np.linalg.norm(local_segment - car, axis=1)
            local_closest_idx = np.argmin(local_distances)

            # Update our progress along the track
            self.current_index += local_closest_idx

            # only send the next 10 meters to the controller
            lookahead_points = 50
            publish_end = min(path_length, self.current_index + lookahead_points)
            segment = self.global_path[self.current_index : publish_end]

            path_msg = Path()
            path_msg.header.frame_id = "odom"
            path_msg.header.stamp = self.get_clock().now().to_msg()

            for point in segment:
                pose = PoseStamped()
                pose.pose.position.x = float(point[0])
                pose.pose.position.y = float(point[1])
                path_msg.poses.append(pose)

            self.path_pub.publish(path_msg)

        # for rviz, can remove
        marker_array = MarkerArray()
        for i, c in enumerate(self.global_cones):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = float(c['x'])
            marker.pose.position.y = float(c['y'])
            marker.pose.position.z = 0.15

            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.3

            color = ColorRGBA()
            color.a = 1.0

            if c['color'] == 'blue':
                color.r = 0.0; color.g = 0.0; color.b = 1.0
            else:
                color.r = 1.0; color.g = 1.0; color.b = 0.0

            marker.color = color
            marker_array.markers.append(marker)

        self.cone_pub.publish(marker_array)


# ==============================================================================
# PURE PURSUIT CONTROLLER NODE
# ==============================================================================

class PIDController:
    # just to handle the throttle control
    def __init__(self, kp, ki, kd, min_val, max_val):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.min_val, self.max_val = min_val, max_val
        self.prev_error, self.integral, self.last_time = 0.0, 0.0, None

    def update(self, error, current_time):
        if self.last_time is None:
            self.last_time = current_time
            return 0.0
        dt = current_time - self.last_time
        if dt <= 0.0: return 0.0

        p = self.kp * error
        self.integral += error * dt
        i = self.ki * self.integral
        d = self.kd * (error - self.prev_error) / dt
        self.prev_error, self.last_time = error, current_time
        return max(self.min_val, min(self.max_val, p + i + d))


def get_yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit')

        self.steering_polarity = -1.0

        self.path = None
        self.car_pos = np.array([0.0, 0.0])
        self.car_yaw = 0.0
        self.current_speed = 0.0
        self.local_cones = []

        self.is_finished = False

        # can tune speed and lookahead stuff over here. also the virtual offset
        self.virtual_offset = 2.5
        self.cone_tube_radius = 4.0  # blinder so that car dont go boom boom

        self.prev_steering = 0.0
        self.steering_filter = 0.5  # speed of steering, 1 is fast, 0.1 is very slow

        self.wheelbase = 2.8
        self.max_steer = 0.60  # limit on steering

        self.max_speed = 7.0
        self.min_speed = 4.0

        self.min_lookahead = 3.0
        self.max_lookahead = 6.5
        self.lookahead_speed_factor = 0.5  # increases lookahead with speed

        self.speed_pid = PIDController(kp=1.0, ki=0.01, kd=0.0, min_val=-1.0, max_val=1.0)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(Path, '/received_global_plan', self.path_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(MarkerArray, '/camera/cone_markers', self.marker_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.target_pub = self.create_publisher(PointStamped, '/pure_pursuit/target', 10)
        self.local_path_pub = self.create_publisher(Path, '/pure_pursuit/local_track', 10)

        self.create_timer(0.05, self.control_loop)

    def path_callback(self, msg):
        self.path = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])

    def odom_callback(self, msg):
        self.current_speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self.car_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        self.car_yaw = get_yaw_from_quaternion(msg.pose.pose.orientation)

    def marker_callback(self, msg):
        if not msg.markers:
            self.local_cones = []
            return

        cones = []
        target_frame = "base_link"
        try:
            trans = self.tf_buffer.lookup_transform(target_frame, msg.markers[0].header.frame_id, rclpy.time.Time())
            for marker in msg.markers:
                if marker.action == 3: continue
                p_in = PointStamped()
                p_in.point = marker.pose.position
                p = tf2_geometry_msgs.do_transform_point(p_in, trans)

                color = 'blue' if marker.color.b > 0.8 else 'yellow' if marker.color.r > 0.8 and marker.color.g > 0.8 else 'unknown'

                if 0.0 < p.point.x < 12.0 and abs(p.point.y) < 6.0:
                    cones.append({'x': p.point.x, 'y': p.point.y, 'color': color})
            self.local_cones = cones
        except Exception:
            return

    def generate_smooth_local_track(self):
        if not self.local_cones:
            return None

        global_bl = []
        if self.path is not None and len(self.path) > 0:
            for p in self.path:
                dx = p[0] - self.car_pos[0]
                dy = p[1] - self.car_pos[1]
                lx = math.cos(-self.car_yaw) * dx - math.sin(-self.car_yaw) * dy
                ly = math.sin(-self.car_yaw) * dx + math.cos(-self.car_yaw) * dy
                global_bl.append([lx, ly])
        global_bl = np.array(global_bl)

        valid_cones = []
        if len(global_bl) > 0:
            for c in self.local_cones:
                c_pt = np.array([c['x'], c['y']])
                dists = np.linalg.norm(global_bl - c_pt, axis=1)
                if np.min(dists) <= self.cone_tube_radius:
                    valid_cones.append(c)
        else:
            valid_cones = self.local_cones

        blue = [c for c in valid_cones if c['color'] == 'blue']
        yellow = [c for c in valid_cones if c['color'] == 'yellow']

        if not blue and not yellow:
            return None

        raw_midpoints = []
        paired_yellow = set()

        for i, b_cone in enumerate(blue):
            closest_j = -1
            min_dist = float('inf')
            for j, y_cone in enumerate(yellow):
                if j in paired_yellow: continue
                d = math.hypot(b_cone['x'] - y_cone['x'], b_cone['y'] - y_cone['y'])
                if d < min_dist:
                    min_dist = d
                    closest_j = j
            if closest_j != -1 and 1.5 <= min_dist <= 4.5:
                paired_yellow.add(closest_j)
                mid_x = (b_cone['x'] + yellow[closest_j]['x']) / 2.0
                mid_y = (b_cone['y'] + yellow[closest_j]['y']) / 2.0
                raw_midpoints.append({'x': mid_x, 'y': mid_y})

        for b_cone in blue:
            if not any(math.hypot(b_cone['x'] - m['x'], (b_cone['y'] + self.virtual_offset) - m['y']) < 1.0 for m in raw_midpoints):
                raw_midpoints.append({'x': b_cone['x'], 'y': b_cone['y'] + self.virtual_offset})

        for y_cone in yellow:
            if not any(math.hypot(y_cone['x'] - m['x'], (y_cone['y'] - self.virtual_offset) - m['y']) < 1.0 for m in raw_midpoints):
                raw_midpoints.append({'x': y_cone['x'], 'y': y_cone['y'] - self.virtual_offset})

        if not raw_midpoints:
            return None

        raw_midpoints.sort(key=lambda m: m['x'])

        traced_path = [[0.0, 0.0]]
        for m in raw_midpoints:
            current_pt = traced_path[-1]
            if m['x'] <= current_pt[0]: continue
            d = math.hypot(m['x'] - current_pt[0], m['y'] - current_pt[1])
            if d > 4.5: continue

            if len(traced_path) > 1:
                prev_pt = traced_path[-2]
                vec1 = np.array([current_pt[0] - prev_pt[0], current_pt[1] - prev_pt[1]])
                vec2 = np.array([m['x'] - current_pt[0], m['y'] - current_pt[1]])
                v1_n, v2_n = np.linalg.norm(vec1), np.linalg.norm(vec2)
                if v1_n > 0 and v2_n > 0:
                    dot = np.clip(np.dot(vec1, vec2) / (v1_n * v2_n), -1.0, 1.0)
                    angle_diff = math.acos(dot)
                    if angle_diff > 0.8:
                        continue

            traced_path.append([m['x'], m['y']])

        if len(traced_path) < 2: return None
        return np.array(traced_path)

    def control_loop(self):
        current_time = self.get_clock().now().nanoseconds / 1e9

        if self.is_finished:
            cmd = Twist()
            cmd.linear.x = -1.0  # Brakes applied
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            return

        if self.path is not None and len(self.path) > 0:
            dist_to_end = math.hypot(self.path[-1][0] - self.car_pos[0], self.path[-1][1] - self.car_pos[1])
            if len(self.path) < 45 and dist_to_end < 4.0:
                self.is_finished = True
                self.get_logger().info("Finish Line Reached! Braking")
                return

        control_lookahead = max(self.min_lookahead, min(self.max_lookahead, self.current_speed * self.lookahead_speed_factor))

        local_path = self.generate_smooth_local_track()
        target_x, target_y = None, None

        if local_path is not None:
            self.publish_local_path_rviz(local_path)

            for i in range(1, len(local_path)):
                dist = math.hypot(local_path[i][0], local_path[i][1])
                if dist >= control_lookahead:
                    target_x, target_y = local_path[i][0], local_path[i][1]
                    break

            if target_x is None:
                target_x, target_y = local_path[-1][0], local_path[-1][1]

        elif self.path is not None and len(self.path) > 0:
            distances = np.linalg.norm(self.path - self.car_pos, axis=1)
            closest_idx = np.argmin(distances)
            target_global = self.path[-1]

            for i in range(closest_idx, len(self.path)):
                if np.linalg.norm(self.path[i] - self.car_pos) >= control_lookahead:
                    target_global = self.path[i]
                    break

            dx, dy = target_global[0] - self.car_pos[0], target_global[1] - self.car_pos[1]
            target_x = math.cos(-self.car_yaw) * dx - math.sin(-self.car_yaw) * dy
            target_y = math.sin(-self.car_yaw) * dx + math.cos(-self.car_yaw) * dy
        else:
            return

        ld = max(0.1, math.hypot(target_x, target_y))
        raw_steering = math.atan((2.0 * self.wheelbase * (target_y / ld)) / ld)
        raw_steering = max(-self.max_steer, min(self.max_steer, raw_steering))

        final_steering = (self.steering_filter * raw_steering) + ((1.0 - self.steering_filter) * self.prev_steering)
        self.prev_steering = final_steering

        steer_ratio = (abs(final_steering) / self.max_steer) ** 2
        target_speed = self.max_speed - steer_ratio * (self.max_speed - self.min_speed)
        throttle = self.speed_pid.update(target_speed - self.current_speed, current_time)

        cmd = Twist()
        cmd.linear.x = float(throttle)
        cmd.angular.z = float(self.steering_polarity * final_steering)
        self.cmd_pub.publish(cmd)

        self.publish_debug_target(target_x, target_y)

    def publish_debug_target(self, local_x, local_y):
        gx = self.car_pos[0] + math.cos(self.car_yaw) * local_x - math.sin(self.car_yaw) * local_y
        gy = self.car_pos[1] + math.sin(self.car_yaw) * local_x + math.cos(self.car_yaw) * local_y

        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.point.x, msg.point.y, msg.point.z = float(gx), float(gy), 0.5
        self.target_pub.publish(msg)

    def publish_local_path_rviz(self, local_path):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        for pt in local_path:
            gx = self.car_pos[0] + math.cos(self.car_yaw) * pt[0] - math.sin(self.car_yaw) * pt[1]
            gy = self.car_pos[1] + math.sin(self.car_yaw) * pt[0] + math.cos(self.car_yaw) * pt[1]
            pose = PoseStamped()
            pose.pose.position.x, pose.pose.position.y = float(gx), float(gy)
            msg.poses.append(pose)
        self.local_path_pub.publish(msg)



def main(args=None):
    rclpy.init(args=args)

    executor = MultiThreadedExecutor()

    planner_node = SkidpadPlanner()
    executor.add_node(planner_node)

    planner_node.get_logger().info(
        "SkidpadPlanner is running. PurePursuit will start in 5 seconds..."
    )

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    time.sleep(5.0)

    pursuit_node = PurePursuit()
    executor.add_node(pursuit_node)
    pursuit_node.get_logger().info("PurePursuit controller is now active.")

    try:
        spin_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        planner_node.destroy_node()
        pursuit_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
