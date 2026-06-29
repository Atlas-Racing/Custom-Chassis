#!/usr/bin/env python3

import math
import os
import subprocess
import time

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
        self.declare_parameter('steering_gain',      2.5)
        self.declare_parameter('steering_offset',    -0.300)
        self.declare_parameter('steer_smooth',       0.3)   # EMA alpha: 0=no smooth, higher=smoother

        # Vision
        self.declare_parameter('vision_horizon',     5.0)
        self.declare_parameter('max_cone_y',         2.0)
        self.declare_parameter('track_half_width',   1.3)   # boundary-to-center distance for blue (m)
        self.declare_parameter('yellow_offset',      0.3)   # smaller → car follows yellow boundary more closely
        self.declare_parameter('min_lookahead',      0.3)
        # < 0.5 shifts midpoint toward yellow (right), away from blue (left).
        # Compensates for car body width — 0.45 is a good starting point.
        self.declare_parameter('centerline_bias',    0.45)

        # Speed
        self.declare_parameter('target_speed',       0.13)
        self.declare_parameter('min_speed',          0.10)
        self.declare_parameter('single_side_speed',  0.11)  # fixed speed in single-side boundary-following mode (yellow turns)
        self.declare_parameter('blue_turn_speed',    0.10)  # slower for blue (right) turns — compensates for hardware understeer
        self.declare_parameter('cone_loss_timeout',  0.4)   # seconds to keep last cmd before stopping when cones disappear
        self.declare_parameter('long_kp',            0.8)    # longitudinal P gain
        self.declare_parameter('long_kd',            0.2)  # longitudinal D gain — keep small to avoid derivative kick
        self.declare_parameter('long_ki',            0.5)   # longitudinal I gain — overcomes pull-out torque
        self.declare_parameter('long_i_max',         0.1)   # anti-windup clamp on integral accumulator (m/s·s)

        # Blue cone repulsion — pushes steer right when a blue cone is too close
        self.declare_parameter('blue_repulsion_radius', 0.8)   # m
        self.declare_parameter('blue_repulsion_gain',   0.6)

        # Right-turn understeer compensation — multiplies raw_steer when turning right
        self.declare_parameter('right_steer_multiplier', 1.4)


        # PID damping on lateral error (y-component of target in car frame)
        # kd damps oscillation; ki corrects persistent drift; i_max prevents windup
        self.declare_parameter('steer_kd',    0.08)   # derivative gain — raise to damp faster
        self.declare_parameter('steer_ki',    0.0)    # integral gain — start at 0, tune last
        self.declare_parameter('steer_i_max', 0.3)    # anti-windup clamp (rad)

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
        self.blue_turn_speed        = g('blue_turn_speed')
        self.cone_loss_timeout      = g('cone_loss_timeout')
        self.long_kp                = g('long_kp')
        self.long_kd                = g('long_kd')
        self.long_ki                = g('long_ki')
        self.long_i_max             = g('long_i_max')
        self.blue_repulsion_radius   = g('blue_repulsion_radius')
        self.blue_repulsion_gain     = g('blue_repulsion_gain')
        self.right_steer_multiplier  = g('right_steer_multiplier')
        self.steer_kd    = g('steer_kd')
        self.steer_ki    = g('steer_ki')
        self.steer_i_max = g('steer_i_max')
        self.target_laps            = g('target_laps')
        self.save_dir         = g('save_dir')

    # ── State ──────────────────────────────────────────────────────────────────

    def _init_state_variables(self):
        self.state               = self.STATE_START_ZONE
        self.start_time          = self.get_clock().now()
        self.cooldown_start_time = None
        self.lap_count           = 0
        self.path_points         = []

        self._speed       = 0.0
        self._steer_ema   = 0.0
        self._single_side = False
        self._blue_side   = False
        self._blue_cones  = []
        self._yellow_cones = []
        self._last_cone_time = self.get_clock().now()

        # Lateral PID state
        self._prev_lateral_error = 0.0
        self._lateral_integral   = 0.0
        self._prev_control_time  = None

        # Longitudinal PID state
        self._prev_speed_error = 0.0
        self._speed_integral   = 0.0
        self._cmd_speed        = 0.0

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
        self._update_state_machine(orange)
        if self.state == self.STATE_FINISHED:
            return

        target = self._get_target(blue, yellow)
        if target is None:
            elapsed = (self.get_clock().now() - self._last_cone_time).nanoseconds / 1e9
            if elapsed > self.cone_loss_timeout:
                self.get_logger().warn('No valid cones — stopping.', throttle_duration_sec=2.0)
                self.stop()
            # else: keep last published command — brief dropout, don't brake
            return

        self._last_cone_time = self.get_clock().now()
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
            # Use only the nearest cone on each side — ignore everything further.
            best_blue, best_yellow = blue_ahead[0], yellow_ahead[0]

            dist_blue   = float(np.linalg.norm(best_blue))
            dist_yellow = float(np.linalg.norm(best_yellow))

            # Reject the gate if the x-mismatch is too large OR if one cone is
            # radially much farther than the other (near-blue + far-yellow false pair).
            x_mismatch_ok = abs(best_blue[0] - best_yellow[0]) <= 1.5
            dist_ratio_ok = dist_yellow < dist_blue * 2.0 and dist_blue < dist_yellow * 2.0

            if x_mismatch_ok and dist_ratio_ok:
                self._single_side = False
                target = self.centerline_bias * best_blue + (1.0 - self.centerline_bias) * best_yellow
                nudge_y = 0.0
                if dist_blue < dist_yellow / 1.5:
                    nudge_y = -0.3
                elif dist_yellow < dist_blue / 1.5:
                    nudge_y = 0.3
                return target + np.array([0.0, nudge_y])

            # Gate rejected: fall through to single-side using the closer boundary
            if dist_blue < dist_yellow:
                yellow_ahead = []
            else:
                blue_ahead = []

        # Single side: follow the boundary curvature, offset to track centre.
        # Left turn  → only yellow (right boundary) visible.
        # Right turn → only blue  (left  boundary) visible.
        self._single_side = True
        if yellow_ahead:
            self._blue_side = False
            if len(yellow_ahead) == 1:
                cone = yellow_ahead[0]
                return np.array([cone[0], cone[1] + self.track_half_width - self.yellow_offset])
            return self._boundary_target(yellow_ahead[:3], side='right')
        if blue_ahead:
            self._blue_side = True
            if len(blue_ahead) == 1:
                cone = blue_ahead[0]
                return np.array([cone[0], cone[1] - self.track_half_width])
            return self._boundary_target(blue_ahead[:3], side='left')

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
            dy = -self.track_half_width if side == 'left' else (self.track_half_width - self.yellow_offset)
            return p + np.array([0.0, dy])

        # For blue (right turn), look 1.5 m ahead to better capture cone curvature;
        # yellow (left turn) stays at 1.0 m.
        lookahead_x = 1.5 if side == 'left' else 1.0
        eval_x  = float(np.clip(lookahead_x, xs[0], xs[-1]))
        curve_y = float(np.polyval(coeffs, eval_x))

        # Lateral offset toward track centre
        if side == 'left':
            target_y = curve_y - self.track_half_width                      # right of blue boundary
        else:
            target_y = curve_y + self.track_half_width - self.yellow_offset  # left of yellow boundary, pulled back from blue

        return np.array([eval_x, target_y])

    # ── Control ────────────────────────────────────────────────────────────────

    def _execute_control(self, target):
        x, y = float(target[0]), float(target[1])
        dist = math.hypot(x, y)

        # Compute dt — used by both lateral and longitudinal PID
        now = self.get_clock().now()
        dt  = 0.033  # ~30 Hz fallback
        if self._prev_control_time is not None:
            dt = max((now - self._prev_control_time).nanoseconds / 1e9, 1e-4)
        self._prev_control_time = now

        # Lateral PID: derivative + integral on target y in car frame
        if dt < 1.0:  # skip on first tick (no prev_control_time)
            d_term = self.steer_kd * (y - self._prev_lateral_error) / dt
            self._lateral_integral = float(np.clip(
                self._lateral_integral + y * dt,
                -self.steer_i_max, self.steer_i_max,
            ))
            i_term = self.steer_ki * self._lateral_integral
        else:
            d_term = 0.0
            i_term = 0.0
        self._prev_lateral_error = y

        # Pure pursuit steering: δ = atan(2L sinα / ld)
        ld    = max(self.min_lookahead, dist)
        alpha = math.atan2(y, x)
        raw_steer = math.atan2(
            2.0 * self.wheelbase * math.sin(alpha), ld
        ) * self.steering_gain + d_term + i_term
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

        # Target speed — governed by alpha (angle to target), not EMA steering.
        # Alpha is a leading indicator: it reacts to upcoming turn sharpness immediately,
        # before steering has built up, so the car slows before it enters the turn.
        alpha_ratio = min(abs(alpha) / (math.pi / 2), 1.0)  # 0 = straight, 1 = 90 deg
        if self._single_side and self._blue_side:
            spd_min, spd_max = self.blue_turn_speed, self.single_side_speed
        elif self._single_side:
            spd_min, spd_max = self.min_speed, self.single_side_speed
        else:
            spd_min, spd_max = self.min_speed, self.target_speed
        # Quadratic drop: speed falls hard even at moderate turn angles
        tgt = spd_min + (spd_max - spd_min) * (1.0 - alpha_ratio) ** 2

        # Longitudinal PID on speed error
        spd_error = tgt - self._speed
        spd_d     = (spd_error - self._prev_speed_error) / dt
        self._speed_integral = float(np.clip(
            self._speed_integral + spd_error * dt,
            -self.long_i_max, self.long_i_max,
        ))
        self._cmd_speed = float(np.clip(
            self.long_kp * spd_error + self.long_kd * spd_d + self.long_ki * self._speed_integral,
            self.min_speed, tgt,
        ))
        self._prev_speed_error = spd_error

        drive.drive.speed          = self._cmd_speed
        drive.drive.steering_angle = -float(self._steer_ema)

        self.get_logger().info(
            f'[CMD] spd={drive.drive.speed:.3f} tgt={tgt:.3f} actual={self._speed:.3f} | '
            f'steer={drive.drive.steering_angle:.3f} target=({x:.2f},{y:.2f}) '
            f'd={d_term:.3f} i={i_term:.3f} single_side={self._single_side} blue_side={self._blue_side} '
            f'n_blue={len(self._blue_cones)} n_yellow={len(self._yellow_cones)}',
            throttle_duration_sec=0.5,
        )
        self.drive_pub.publish(drive)

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
        self._speed_integral   = 0.0
        self._prev_speed_error = 0.0
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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
