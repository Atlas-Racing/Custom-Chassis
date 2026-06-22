#!/usr/bin/env python3

import math
import os
import subprocess
import time

import cv2
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


class PurePursuitNode(Node):

    STATE_START_ZONE   = 0
    STATE_RACING       = 1
    STATE_FINISHED     = 2
    STATE_LAP_COOLDOWN = 3

    def __init__(self):
        super().__init__('pure_pursuit_node')
        self._init_parameters()
        self._init_state_variables()
        self._init_interfaces()
        self._publish_neutral()
        self.get_logger().info('Pure Pursuit Node started.')

    # ── Parameters ─────────────────────────────────────────────────────────────

    def _init_parameters(self):
        g = lambda n: self.get_parameter(n).value

        # Geometry
        self.declare_parameter('wheelbase',          0.257)
        self.declare_parameter('max_steering_angle', 0.7)
        self.declare_parameter('steering_gain',      2.0)
        self.declare_parameter('steering_offset',    -0.300)
        self.declare_parameter('steer_smooth',       0.5)   # EMA alpha: 0=no smooth, higher=smoother

        # Vision
        self.declare_parameter('vision_horizon',     5.0)
        self.declare_parameter('max_cone_y',         2.0)
        self.declare_parameter('track_half_width',   1.3)   # boundary-to-center distance for blue (m)
        self.declare_parameter('yellow_offset',      0.3)   # smaller → car follows yellow boundary more closely
        self.declare_parameter('min_lookahead',      0.3)
        # < 0.5 shifts midpoint toward yellow (right), away from blue (left).
        # Compensates for car body width — 0.45 is a good starting point.
        self.declare_parameter('centerline_bias',    0.0)

        # Speed
        self.declare_parameter('target_speed',      0.15)
        self.declare_parameter('min_speed',         0.08)
        self.declare_parameter('single_side_speed', 0.18)  # fixed speed in single-side boundary-following mode
        self.declare_parameter('long_kp',           1.0)   # longitudinal P gain
        self.declare_parameter('long_kd',           1.0)   # longitudinal D gain (no I)

        # Blue cone repulsion — pushes steer right when a blue cone is too close
        self.declare_parameter('blue_repulsion_radius', 0.8)   # m
        self.declare_parameter('blue_repulsion_gain',   0.6)

        # Right-turn understeer compensation — multiplies raw_steer when turning right
        self.declare_parameter('right_steer_multiplier', 1.4)

        # Stop-and-steer for right turns (chassis hardware limitation)
        # When required right steer exceeds this threshold, stop and hold max right steering.
        self.declare_parameter('right_steer_threshold', 0.55)  # rad — only fires on sharp right turns

        # Race
        self.declare_parameter('target_laps',        5)
        self.declare_parameter('save_dir',           os.getcwd())

        self.wheelbase        = g('wheelbase')
        self.max_steer        = g('max_steering_angle')
        self.steering_gain    = g('steering_gain')
        self.steering_offset  = g('steering_offset')
        self.steer_smooth     = g('steer_smooth')
        self.vision_horizon   = g('vision_horizon')
        self.max_cone_y       = g('max_cone_y')
        self.track_half_width       = g('track_half_width')
        self.yellow_offset          = g('yellow_offset')
        self.min_lookahead          = g('min_lookahead')
        self.centerline_bias        = g('centerline_bias')
        self.target_speed           = g('target_speed')
        self.min_speed              = g('min_speed')
        self.single_side_speed      = g('single_side_speed')
        self.long_kp                = g('long_kp')
        self.long_kd                = g('long_kd')
        self.blue_repulsion_radius   = g('blue_repulsion_radius')
        self.blue_repulsion_gain     = g('blue_repulsion_gain')
        self.right_steer_multiplier  = g('right_steer_multiplier')
        self.right_steer_threshold   = g('right_steer_threshold')
        self.target_laps            = g('target_laps')
        self.save_dir         = g('save_dir')

    # ── State ──────────────────────────────────────────────────────────────────

    def _init_state_variables(self):
        self.state               = self.STATE_START_ZONE
        self.start_time          = self.get_clock().now()
        self.cooldown_start_time = None
        self.lap_count           = 0
        self.path_points         = []

        self._speed        = 0.0
        self._cmd_speed    = 0.0
        self._steer_ema    = 0.0
        self._single_side  = False
        self._blue_cones   = []
        self._yellow_cones = []
        self._orange_cones = []

        self._prev_control_time = None
        self._prev_speed_error  = 0.0

    # ── Interfaces ─────────────────────────────────────────────────────────────

    def _init_interfaces(self):
        self.marker_sub = self.create_subscription(
            MarkerArray, '/hydrakon_camera/cone_markers', self.marker_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/zed/zed_node/odom', self.odom_callback, 10
        )
        self.drive_pub  = self.create_publisher(AckermannDriveStamped, '/ackermann_cmd', 10)
        self.target_pub = self.create_publisher(PointStamped, '/pure_pursuit/target', 10)

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def odom_callback(self, msg: Odometry):
        self._speed = msg.twist.twist.linear.x
        if self.state != self.STATE_FINISHED:
            self.path_points.append((msg.pose.pose.position.x, msg.pose.pose.position.y))

    def marker_callback(self, msg: MarkerArray):
        if self.state == self.STATE_FINISHED:
            return

        blue, yellow, orange = [], [], []

        for m in msg.markers:
            if m.action in (Marker.DELETE, Marker.DELETEALL):
                continue
            x, y = m.pose.position.x, m.pose.position.y
            if x < 0.0 or x > self.vision_horizon:
                continue
            if abs(y) > self.max_cone_y:
                continue
            r, gc, b = m.color.r, m.color.g, m.color.b
            if b > 0.9 and r < 0.1:
                blue.append(np.array([x, y]))
            elif r > 0.9 and gc > 0.9:
                yellow.append(np.array([x, y]))
            elif r > 0.9 and 0.4 < gc < 0.6:
                orange.append(np.array([x, y]))

        self._blue_cones   = blue
        self._yellow_cones = yellow
        self._orange_cones = orange
        self._update_state_machine(orange)
        if self.state == self.STATE_FINISHED:
            return

        target = self._get_target(blue, yellow)
        if target is None:
            self.get_logger().warn('No valid cones — stopping.', throttle_duration_sec=2.0)
            self.stop()
            self._draw_bev(blue, yellow, orange, None)
            return

        self._draw_bev(blue, yellow, orange, target)
        self._publish_target_debug(target)
        self._execute_control(target)

    # ── State machine ──────────────────────────────────────────────────────────

    def _update_state_machine(self, orange_cones):
        now     = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds / 1e9

        if self.state == self.STATE_START_ZONE:
            if len(orange_cones) == 0 and elapsed > 5.0:
                self.get_logger().info('Left Start Zone. RACING.')
                self.state = self.STATE_RACING

        elif self.state == self.STATE_RACING:
            close = [p for p in orange_cones if p[0] < 5.0]
            if len(close) >= 2:
                self._handle_lap_completion()

        elif self.state == self.STATE_LAP_COOLDOWN:
            cooldown = (now - self.cooldown_start_time).nanoseconds / 1e9
            if cooldown > 5.0:
                self.get_logger().info('Cooldown done. Resuming.')
                self.state = self.STATE_RACING

    def _handle_lap_completion(self):
        self.lap_count += 1
        self.get_logger().info(f'LAP {self.lap_count} COMPLETED!')
        if self.lap_count >= self.target_laps:
            self.stop()
            self.state = self.STATE_FINISHED
            time.sleep(2.0)
            self._save_mission_data()
        else:
            self.state = self.STATE_LAP_COOLDOWN
            self.cooldown_start_time = self.get_clock().now()

    def _save_mission_data(self):
        map_path = os.path.join(self.save_dir, 'my_track_map')
        try:
            subprocess.run(
                ['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', map_path],
                check=True,
            )
            self.get_logger().info(f'Map saved to {map_path}')
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f'Map save failed: {e}')

        path_file = os.path.join(self.save_dir, 'my_track_path.csv')
        try:
            with open(path_file, 'w') as f:
                f.write('x,y\n')
                for px, py in self.path_points:
                    f.write(f'{px},{py}\n')
            self.get_logger().info(f'Path saved to {path_file}')
        except Exception as e:
            self.get_logger().error(f'Path save failed: {e}')

    # ── Target selection ───────────────────────────────────────────────────────

    def _get_target(self, blue, yellow):
        """
        Both sides visible → midpoint of nearest blue and nearest yellow cone.
        Single side visible → follow that boundary's curvature, offset to center.
        """
        blue_ahead   = sorted([c for c in blue   if c[0] > 0.0], key=lambda p: p[0])
        yellow_ahead = sorted([c for c in yellow if c[0] > 0.0], key=lambda p: p[0])

        if blue_ahead and yellow_ahead:
            self._single_side = False
            nearest_blue   = min(blue_ahead,   key=lambda p: np.linalg.norm(p))
            nearest_yellow = min(yellow_ahead, key=lambda p: np.linalg.norm(p))
            return self.centerline_bias * nearest_blue + (1.0 - self.centerline_bias) * nearest_yellow

        # Single side: follow the boundary curvature with a small inward offset.
        # Left turn  → only yellow (right boundary) visible.
        # Right turn → only blue  (left  boundary) visible.
        self._single_side = True
        if yellow_ahead:
            if len(yellow_ahead) == 1:
                # Only one yellow cone — steer left to scan for more of the boundary.
                cone = yellow_ahead[0]
                return np.array([cone[0], cone[1] + self.track_half_width])
            return self._boundary_target(yellow_ahead, side='right')
        if blue_ahead:
            return self._boundary_target(blue_ahead, side='left')

        return None

    def _boundary_target(self, cones, side):
        """
        Fit a polynomial through single-side boundary cones.
        Evaluate 1 m ahead on the curve, then offset laterally toward track centre.
        Pure lateral offset (no perpendicular normal) keeps the target always ahead
        of the car even on tight turns where a normal would point backward.
        """
        xs = np.array([p[0] for p in cones], dtype=float)
        ys = np.array([p[1] for p in cones], dtype=float)

        deg = min(2, len(cones) - 1)
        try:
            coeffs = np.polyfit(xs, ys, deg)
        except np.linalg.LinAlgError:
            p  = cones[0]
            dy = -self.track_half_width if side == 'left' else self.track_half_width
            return p + np.array([0.0, dy])

        # For blue (right turn), look 1.5 m ahead to better capture cone curvature;
        # yellow (left turn) stays at 1.0 m.
        lookahead_x = 1.5 if side == 'left' else 1.0
        eval_x  = float(np.clip(lookahead_x, xs[0], xs[-1]))
        curve_y = float(np.polyval(coeffs, eval_x))

        # Lateral offset toward track centre — always keeps target at positive x
        if side == 'left':
            target_y = curve_y - self.track_half_width   # right of blue boundary
        else:
            target_y = curve_y + self.yellow_offset      # left of yellow boundary — stay close

        return np.array([eval_x, target_y])

    # ── Control ────────────────────────────────────────────────────────────────

    def _execute_control(self, target):
        x, y = float(target[0]), float(target[1])
        dist = math.hypot(x, y)

        now = self.get_clock().now()
        dt = 0.033  # ~30 Hz fallback
        if self._prev_control_time is not None:
            dt = max((now - self._prev_control_time).nanoseconds / 1e9, 1e-4)
        self._prev_control_time = now

        # Pure pursuit steering: δ = atan(2L sinα / ld)
        ld    = max(self.min_lookahead, dist)
        alpha = math.atan2(y, x)
        raw_steer = math.atan2(
            2.0 * self.wheelbase * math.sin(alpha), ld
        ) * self.steering_gain
        # Compensate for right-turn hardware understeer
        if raw_steer < 0.0:
            raw_steer *= self.right_steer_multiplier
        raw_steer = float(np.clip(raw_steer, -self.max_steer, self.max_steer))

        # Blue cone repulsion: push steer right (negative) when any blue cone is close
        for cone in self._blue_cones:
            dist = float(np.linalg.norm(cone))
            if 0.05 < dist < self.blue_repulsion_radius:
                strength = (1.0 - dist / self.blue_repulsion_radius) ** 2
                raw_steer -= self.blue_repulsion_gain * strength
        raw_steer = float(np.clip(raw_steer, -self.max_steer, self.max_steer))

        # EMA smoothing
        self._steer_ema = self.steer_smooth * self._steer_ema + (1.0 - self.steer_smooth) * raw_steer

        drive = AckermannDriveStamped()
        drive.header.stamp    = self.get_clock().now().to_msg()
        drive.header.frame_id = 'base_link'

        # Stop-and-steer: chassis cannot produce enough right steering while moving.
        # Only applies when both cone colours are visible — single-side boundary
        # following already steers toward the curve and must not be interrupted.
        if not self._single_side and self._steer_ema < -self.right_steer_threshold:
            self._cmd_speed           = 0.0
            self._prev_speed_error    = 0.0
            drive.drive.speed          = 0.0
            drive.drive.steering_angle = self.max_steer   # positive = right on hardware
        else:
            if self._single_side:
                tgt = float(self.single_side_speed)
            else:
                steering_ratio = abs(self._steer_ema) / self.max_steer
                tgt = max(0.0, self.min_speed + (self.target_speed - self.min_speed) * (1.0 - steering_ratio))

            error   = tgt - self._speed
            d_error = (error - self._prev_speed_error) / dt
            self._cmd_speed        = float(np.clip(
                self.long_kp * error + self.long_kd * d_error, 0.0, tgt
            ))
            self._prev_speed_error = error
            drive.drive.speed          = self._cmd_speed
            drive.drive.steering_angle = -float(self._steer_ema)

        self.get_logger().info(
            f'[CMD] spd={drive.drive.speed:.3f} steer={drive.drive.steering_angle:.3f} | '
            f'target=({x:.2f},{y:.2f}) single_side={self._single_side} '
            f'n_blue={len(self._blue_cones)} n_yellow={len(self._yellow_cones)} '
            f'cmd_spd={self._cmd_speed:.3f}',
            throttle_duration_sec=0.5,
        )
        self.drive_pub.publish(drive)

    def _draw_bev(self, blue, yellow, orange, target):
        img_h, img_w = 600, 400
        scale = 70          # pixels per metre
        cx = img_w // 2
        cy = img_h - 60

        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        def w2i(wx, wy):
            return (int(cx - wy * scale), int(cy - wx * scale))

        # Distance grid
        for d in range(1, 9):
            row = cy - d * scale
            if 0 <= row < img_h:
                cv2.line(img, (0, row), (img_w - 1, row), (30, 30, 30), 1)
                cv2.putText(img, f'{d}m', (3, row - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (55, 55, 55), 1)

        # Pairing line between nearest blue and nearest yellow
        if blue and yellow and not self._single_side:
            nb = min(blue,   key=lambda p: float(np.linalg.norm(p)))
            ny = min(yellow, key=lambda p: float(np.linalg.norm(p)))
            cv2.line(img, w2i(*nb), w2i(*ny), (0, 180, 0), 1)

        for c in orange:
            cv2.circle(img, w2i(c[0], c[1]), 6, (0, 127, 255), -1)
        for c in yellow:
            cv2.circle(img, w2i(c[0], c[1]), 6, (0, 220, 220), -1)
        for c in blue:
            cv2.circle(img, w2i(c[0], c[1]), 6, (255, 80,  0),  -1)

        # Target point + line from car
        if target is not None:
            tp = w2i(float(target[0]), float(target[1]))
            cv2.line(img, (cx, cy), tp, (0, 200, 0), 1)
            cv2.circle(img, tp, 7, (0, 255, 0), -1)

        # Car footprint
        cv2.rectangle(img, (cx - 10, cy - 18), (cx + 10, cy + 18), (200, 200, 200), 2)
        cv2.arrowedLine(img, (cx, cy), (cx, cy - 26), (200, 200, 200), 1, tipLength=0.4)

        # HUD
        cv2.putText(img,
                    f'spd={self._speed:.2f} cmd={self._cmd_speed:.2f} steer={self._steer_ema:.3f}',
                    (5, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        mode = 'SINGLE-SIDE' if self._single_side else 'BOTH-SIDES'
        cv2.putText(img, mode, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 0), 1)

        cv2.imshow('BeV — Cone Pairing', img)
        cv2.waitKey(1)

    def _publish_neutral(self):
        msg = AckermannDriveStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed     = 0.0
        msg.drive.steering_angle = self.steering_offset
        self.drive_pub.publish(msg)

    def _publish_target_debug(self, point: np.ndarray):
        t = PointStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.point.x = float(point[0])
        t.point.y = float(point[1])
        self.target_pub.publish(t)

    def stop(self):
        drive = AckermannDriveStamped()
        drive.header.stamp    = self.get_clock().now().to_msg()
        drive.header.frame_id = 'base_link'
        drive.drive.speed     = 0.0
        drive.drive.steering_angle = 0.0
        self.drive_pub.publish(drive)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
