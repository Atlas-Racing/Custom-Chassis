#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import math

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, PointStamped, PoseStamped
from visualization_msgs.msg import MarkerArray

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs

class PIDController:
    #just to handle the throttle control
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

# get yaw from quaternion
def get_yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w*q.z + q.x*q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
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

        #can tune speed and lookahead stuff over here. also the virtual offset
        self.virtual_offset = 2.5
        self.cone_tube_radius = 4.0 #blinder so that car dont go boom boom, i.e. dont see cones its not concerned with

        self.prev_steering = 0.0
        self.steering_filter = 0.5 #speed of steering, 1 is fast steering, 0.1 is vslow steering

        self.wheelbase = 2.8
        self.max_steer = 0.60 #limit on steering

        self.max_speed = 7.0
        self.min_speed = 4.0

        self.min_lookahead = 3.0
        self.max_lookahead = 6.5
        self.lookahead_speed_factor = 0.5 #increases loohahead with speed

        self.speed_pid = PIDController(kp=1.0, ki=0.01, kd=0.0, min_val=-1.0, max_val=1.0) #tune kp for throttle control

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

                #checking how far the cone is, if its too far or behind us, we dont care about it. also only care about cones 12m in front of us and within 6m to the sides, to reduce noise and focus on relevant cones
                if 0.0 < p.point.x < 12.0 and abs(p.point.y) < 6.0:
                    cones.append({'x': p.point.x, 'y': p.point.y, 'color': color})
            self.local_cones = cones
        except Exception:
            return

    def generate_smooth_local_track(self):
        if not self.local_cones:
            return None

        #just transform global path to local frame, so we can do some math with the cones and the path together. also makes it easier to filter cones based on distance from path
        global_bl = []
        if self.path is not None and len(self.path) > 0:
            for p in self.path:
                dx = p[0] - self.car_pos[0]
                dy = p[1] - self.car_pos[1]
                lx = math.cos(-self.car_yaw) * dx - math.sin(-self.car_yaw) * dy #path aligned to car frame lx and ly
                ly = math.sin(-self.car_yaw) * dx + math.cos(-self.car_yaw) * dy
                global_bl.append([lx, ly])
        global_bl = np.array(global_bl)

        #only allow cones which are within the blinding radius so that car not get distracted.
        valid_cones = []
        if len(global_bl) > 0:
            for c in self.local_cones:
                c_pt = np.array([c['x'], c['y']])
                dists = np.linalg.norm(global_bl - c_pt, axis=1)
                if np.min(dists) <= self.cone_tube_radius:
                    valid_cones.append(c)
        else:
            valid_cones = self.local_cones

        #clean list of blue and yellow cones
        blue = [c for c in valid_cones if c['color'] == 'blue']
        yellow = [c for c in valid_cones if c['color'] == 'yellow']

        if not blue and not yellow:
            return None

        #finding the midpoints
        raw_midpoints = []
        paired_yellow = set()

        #check if pair can be made
        for i, b_cone in enumerate(blue):
            closest_j = -1
            min_dist = float('inf')
            for j, y_cone in enumerate(yellow): #if paired, no one else can take her from me
                if j in paired_yellow: continue
                d = math.hypot(b_cone['x'] - y_cone['x'], b_cone['y'] - y_cone['y'])
                if d < min_dist:
                    min_dist = d
                    closest_j = j
            #if the distance bw two cones is within a range, they both can be pairs and live happily ever after
            if closest_j != -1 and 1.5 <= min_dist <= 4.5:
                paired_yellow.add(closest_j)
                mid_x = (b_cone['x'] + yellow[closest_j]['x']) / 2.0 #getting midpoints
                mid_y = (b_cone['y'] + yellow[closest_j]['y']) / 2.0
                raw_midpoints.append({'x': mid_x, 'y': mid_y})

        #for the lonely cones without a wife, we have imaginary midpoints tuned with virtual_offset
        for b_cone in blue:
            if not any(math.hypot(b_cone['x'] - m['x'], (b_cone['y'] + self.virtual_offset) - m['y']) < 1.0 for m in raw_midpoints):
                raw_midpoints.append({'x': b_cone['x'], 'y': b_cone['y'] + self.virtual_offset})

        for y_cone in yellow:
            if not any(math.hypot(y_cone['x'] - m['x'], (y_cone['y'] - self.virtual_offset) - m['y']) < 1.0 for m in raw_midpoints):
                raw_midpoints.append({'x': y_cone['x'], 'y': y_cone['y'] - self.virtual_offset})

        if not raw_midpoints:
            return None

        raw_midpoints.sort(key=lambda m: m['x']) #sorting midpoints from the wild dots array

        #connecting the dots
        traced_path = [[0.0, 0.0]]
        for m in raw_midpoints:
            current_pt = traced_path[-1]
            if m['x'] <= current_pt[0]: continue #only look forward, no going back
            d = math.hypot(m['x'] - current_pt[0], m['y'] - current_pt[1])
            if d > 4.5: continue   #if the next dot is very far, probably not a real midpoint, so skip it. also helps to reduce noise and crazy paths when we have very few cones visible

            # drawing two lines vec1 from last point to current point, and vec2 from current point to next midpoint. if the angle between them is very sharp, probably not a real midpoint or we have a bad jump in midpoints, so skip it.
            if len(traced_path) > 1:
                prev_pt = traced_path[-2]
                vec1 = np.array([current_pt[0] - prev_pt[0], current_pt[1] - prev_pt[1]])
                vec2 = np.array([m['x'] - current_pt[0], m['y'] - current_pt[1]])
                v1_n, v2_n = np.linalg.norm(vec1), np.linalg.norm(vec2)
                if v1_n > 0 and v2_n > 0: #connecting the two lines, if sharp angle then ignore this midpoint. v2 is like BC
                    dot = np.clip(np.dot(vec1, vec2) / (v1_n * v2_n), -1.0, 1.0)
                    angle_diff = math.acos(dot) #CRAZY MATH
                    if angle_diff > 0.8:
                        continue

            traced_path.append([m['x'], m['y']]) #if the dot is good and passed all the checks, add it to the traced path

        if len(traced_path) < 2: return None
        return np.array(traced_path)

    def control_loop(self):
        current_time = self.get_clock().now().nanoseconds / 1e9

        if self.is_finished:
            cmd = Twist()
            cmd.linear.x = -1.0 # Brakes applied
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
        else: return

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
    node = PurePursuit()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
