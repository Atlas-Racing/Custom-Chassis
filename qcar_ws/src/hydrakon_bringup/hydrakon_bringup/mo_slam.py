#!/usr/bin/env python3
"""
MO-SLAM: Merged Observation SLAM Node
---------------------------------------
Camera-only landmark SLAM for the qcar platform.

Pose is tracked purely from ZED odometry. Landmark positions are maintained
as running averages — no covariance matrix, no matrix inversions.

Subscriptions:
    /hydrakon_camera/cone_markers (MarkerArray) — camera cone detections in base_link frame
    /zed/zed_node/odom (Odometry) — ZED positional tracking for pose prediction

Publications:
    /map (OccupancyGrid) — inflated cone obstacles for Nav2
    /map/landmarks (MarkerArray) — colored cone landmarks for RViz
    /tf: map → odom (identity for now)
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import TransformStamped

import tf2_ros
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
import tf2_geometry_msgs


# ═════════════════════════════════════════════════════════════════════════
# LANDMARK METADATA
# ═════════════════════════════════════════════════════════════════════════
class Landmark:
    """Metadata for a confirmed landmark."""
    def __init__(self, landmark_id, color_id):
        self.id = landmark_id
        self.color_id = color_id  # 0=unknown, 1=yellow, 2=blue, 3=orange
        self.obs_count = 0

    def update_color(self, color_id):
        if color_id != 0 and self.color_id == 0:
            self.color_id = color_id


# ═════════════════════════════════════════════════════════════════════════
# MO-SLAM NODE
# ═════════════════════════════════════════════════════════════════════════
class MOSlamNode(Node):
    def __init__(self):
        super().__init__('mo_slam_node')

        # ── Data Association ────────────────────────────────────────────
        self.declare_parameter('association_threshold', 0.12)
        self.declare_parameter('new_landmark_min_obs', 1)
        self.declare_parameter('color_gate_enabled', True)

        # ── Occupancy Grid ──────────────────────────────────────────────
        self.declare_parameter('grid_width', 20.0)
        self.declare_parameter('grid_height', 20.0)
        self.declare_parameter('grid_resolution', 0.05)
        self.declare_parameter('cone_inflation_radius', 0.04)
        self.declare_parameter('grid_publish_rate', 2.0)
        self.declare_parameter('landmark_publish_rate', 10.0)

        # ── Landmark Marker Dimensions ──────────────────────────────────
        self.declare_parameter('landmark_diameter', 0.04)
        self.declare_parameter('landmark_height', 0.06)
        self.declare_parameter('max_detection_range', 2.5)

        # ── Read All Parameters ─────────────────────────────────────────
        self.assoc_thresh = self.get_parameter('association_threshold').value
        self.min_obs      = self.get_parameter('new_landmark_min_obs').value
        self.color_gate   = self.get_parameter('color_gate_enabled').value

        self.grid_width       = self.get_parameter('grid_width').value
        self.grid_height      = self.get_parameter('grid_height').value
        self.grid_res         = self.get_parameter('grid_resolution').value
        self.cone_inflate     = self.get_parameter('cone_inflation_radius').value
        grid_rate             = self.get_parameter('grid_publish_rate').value
        landmark_rate         = self.get_parameter('landmark_publish_rate').value
        self.landmark_diameter = self.get_parameter('landmark_diameter').value
        self.landmark_height   = self.get_parameter('landmark_height').value
        self.max_range         = self.get_parameter('max_detection_range').value

        # ── Landmark Merge ──────────────────────────────────────────────
        self.declare_parameter('merge_distance', 0.15)
        self.declare_parameter('merge_trigger_count', 20)
        self.merge_dist    = self.get_parameter('merge_distance').value
        self.merge_trigger = self.get_parameter('merge_trigger_count').value
        self.merge_requested = False
        self.create_timer(10.0, self._periodic_merge)

        # ── TF ──────────────────────────────────────────────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ── Pose + Landmark State ───────────────────────────────────────
        # mu: [x, y, theta, lm0_x, lm0_y, lm1_x, lm1_y, ...]
        # No covariance matrix — landmark positions updated by running average.
        self.mu = np.zeros(3)
        self.landmarks = []
        self.next_landmark_id = 0
        self.pending = []

        # ── Odometry Tracking ──────────────────────────────────────────
        self.prev_odom_x     = None
        self.prev_odom_y     = None
        self.prev_odom_theta = None
        self.odom_initialized = False

        self.cam_msg_count    = 0
        self.cam_process_every = 1
        self.pending_updates  = []

        # ── Publishers ─────────────────────────────────────────────────
        grid_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.grid_pub     = self.create_publisher(OccupancyGrid, '/map', grid_qos)
        self.landmark_pub = self.create_publisher(MarkerArray, '/map/landmarks', 10)

        # ── Subscribers ────────────────────────────────────────────────
        self.create_subscription(MarkerArray, '/hydrakon_camera/cone_markers', self.camera_callback, 10)
        self.create_subscription(Odometry, '/zed/zed_node/odom', self.odom_callback, 10)

        # ── Timers ─────────────────────────────────────────────────────
        self.create_timer(1.0 / grid_rate,    self.publish_occupancy_grid)
        self.create_timer(1.0 / landmark_rate, self.publish_landmarks)
        self.create_timer(0.05,               self.publish_tf)

        self.get_logger().info(
            f'MO-SLAM started (mean-only) | '
            f'assoc={self.assoc_thresh}m | min_obs={self.min_obs} | '
            f'range={self.max_range}m'
        )

    # ═════════════════════════════════════════════════════════════════════
    # TF HELPERS
    # ═════════════════════════════════════════════════════════════════════
    def get_transform(self, target_frame, source_frame):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time()
            )
        except Exception:
            return None

    @staticmethod
    def _quat_to_yaw(x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    # ═════════════════════════════════════════════════════════════════════
    # POSE PREDICTION (from odometry — no covariance propagation)
    # ═════════════════════════════════════════════════════════════════════
    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = self._quat_to_yaw(q.x, q.y, q.z, q.w)

        if not self.odom_initialized:
            self.prev_odom_x     = x
            self.prev_odom_y     = y
            self.prev_odom_theta = theta
            self.mu[0] = x
            self.mu[1] = y
            self.mu[2] = theta
            self.odom_initialized = True
            return

        dx     = x - self.prev_odom_x
        dy     = y - self.prev_odom_y
        dtheta = self._normalize_angle(theta - self.prev_odom_theta)

        self.prev_odom_x     = x
        self.prev_odom_y     = y
        self.prev_odom_theta = theta

        if math.hypot(dx, dy) < 1e-6 and abs(dtheta) < 1e-6:
            return

        self.mu[0] += dx
        self.mu[1] += dy
        self.mu[2]  = self._normalize_angle(self.mu[2] + dtheta)

    # ═════════════════════════════════════════════════════════════════════
    # CAMERA PIPELINE (transform → color → landmark update)
    # ═════════════════════════════════════════════════════════════════════
    def camera_callback(self, msg):
        self.cam_msg_count += 1
        if self.cam_msg_count % self.cam_process_every != 0:
            return

        if not self.odom_initialized or not msg.markers:
            return

        source_frame = msg.markers[0].header.frame_id
        transform    = self.get_transform("odom", source_frame)
        if not transform:
            return

        for marker in msg.markers:
            if marker.action == Marker.DELETEALL:
                continue

            if math.hypot(marker.pose.position.x, marker.pose.position.y) > self.max_range:
                continue

            p_stamped         = tf2_geometry_msgs.PointStamped()
            p_stamped.point.x = marker.pose.position.x
            p_stamped.point.y = marker.pose.position.y
            p_stamped.point.z = marker.pose.position.z

            try:
                p_odom = tf2_geometry_msgs.do_transform_point(p_stamped, transform)

                r, g, b  = marker.color.r, marker.color.g, marker.color.b
                color_id = 0
                if r > 0.9 and g > 0.9:
                    color_id = 1  # Yellow
                elif b > 0.9:
                    color_id = 2  # Blue
                elif r > 0.9 and g > 0.4:
                    color_id = 3  # Orange

                self._process_observation(p_odom.point.x, p_odom.point.y, color_id)
            except Exception:
                continue

        self._apply_updates()

        if self.landmarks:
            self.publish_landmarks()

    # ═════════════════════════════════════════════════════════════════════
    # DATA ASSOCIATION
    # ═════════════════════════════════════════════════════════════════════
    def _process_observation(self, obs_x, obs_y, color_id):
        best_idx  = -1
        best_dist = self.assoc_thresh

        for j, lm in enumerate(self.landmarks):
            if self.color_gate and color_id != 0 and lm.color_id != 0:
                if color_id != lm.color_id:
                    continue
            lm_idx = 3 + 2 * j
            dist   = math.hypot(obs_x - self.mu[lm_idx], obs_y - self.mu[lm_idx + 1])
            if dist < best_dist:
                best_dist = dist
                best_idx  = j

        if best_idx >= 0:
            self.pending_updates.append((best_idx, obs_x, obs_y, color_id))
        elif not self._try_promote_pending(obs_x, obs_y, color_id):
            self._add_pending(obs_x, obs_y, color_id)

    # ═════════════════════════════════════════════════════════════════════
    # RUNNING-AVERAGE LANDMARK UPDATE (replaces EKF batch — no matrices)
    # ═════════════════════════════════════════════════════════════════════
    def _apply_updates(self):
        for (lm_index, obs_x, obs_y, color_id) in self.pending_updates:
            lm_idx = 3 + 2 * lm_index
            n      = self.landmarks[lm_index].obs_count + 1
            # Cumulative running average: new_mean = old_mean + (obs - old_mean) / n
            self.mu[lm_idx]     += (obs_x - self.mu[lm_idx])     / n
            self.mu[lm_idx + 1] += (obs_y - self.mu[lm_idx + 1]) / n
            self.landmarks[lm_index].obs_count = n
            self.landmarks[lm_index].update_color(color_id)

        self.pending_updates.clear()

        if self.merge_requested:
            self._merge_landmarks()
            self.merge_requested = False

    # ═════════════════════════════════════════════════════════════════════
    # LANDMARK MANAGEMENT
    # ═════════════════════════════════════════════════════════════════════
    def _add_pending(self, x, y, color_id):
        for p in self.pending:
            color_ok = (color_id == 0 or p['color_id'] == 0 or color_id == p['color_id'])
            if math.hypot(p['x'] - x, p['y'] - y) < self.assoc_thresh and color_ok:
                n        = p['obs_count']
                p['x']   = (p['x'] * n + x) / (n + 1)
                p['y']   = (p['y'] * n + y) / (n + 1)
                p['obs_count'] += 1
                if color_id != 0:
                    p['color_id'] = color_id
                return

        self.pending.append({'x': x, 'y': y, 'color_id': color_id, 'obs_count': 1})

    def _try_promote_pending(self, x, y, color_id):
        for i, p in enumerate(self.pending):
            color_ok = (color_id == 0 or p['color_id'] == 0 or color_id == p['color_id'])
            if math.hypot(p['x'] - x, p['y'] - y) < self.assoc_thresh and color_ok:
                p['obs_count'] += 1
                n       = p['obs_count']
                p['x'] += (x - p['x']) / n
                p['y'] += (y - p['y']) / n
                if color_id != 0:
                    p['color_id'] = color_id

                if p['obs_count'] >= self.min_obs:
                    # Don't promote if too close to an existing landmark
                    for j in range(len(self.landmarks)):
                        lm_idx = 3 + 2 * j
                        if math.hypot(self.mu[lm_idx] - p['x'], self.mu[lm_idx + 1] - p['y']) < self.assoc_thresh:
                            self.pending.pop(i)
                            return True
                    self._add_landmark(p['x'], p['y'], p['color_id'])
                    self.pending.pop(i)
                    return True
                return False
        return False

    def _add_landmark(self, x, y, color_id):
        lm           = Landmark(self.next_landmark_id, color_id)
        lm.obs_count = 1
        self.next_landmark_id += 1
        self.landmarks.append(lm)
        self.mu = np.append(self.mu, [x, y])

        self.get_logger().info(
            f'Landmark #{lm.id} at ({x:.2f}, {y:.2f}) '
            f'color={color_id} | total: {len(self.landmarks)}'
        )

    def _periodic_merge(self):
        if len(self.landmarks) >= self.merge_trigger:
            self.merge_requested = True

    def _merge_landmarks(self):
        if len(self.landmarks) < 2:
            return

        remove_indices = set()

        for i in range(len(self.landmarks)):
            if i in remove_indices:
                continue
            for j in range(i + 1, len(self.landmarks)):
                if j in remove_indices:
                    continue

                idx_i = 3 + 2 * i
                idx_j = 3 + 2 * j
                dist  = math.hypot(
                    self.mu[idx_i]     - self.mu[idx_j],
                    self.mu[idx_i + 1] - self.mu[idx_j + 1],
                )

                if dist > 0.3:
                    continue

                ci = self.landmarks[i].color_id
                cj = self.landmarks[j].color_id
                if ci != 0 and cj != 0 and ci != cj:
                    continue

                obs_i = self.landmarks[i].obs_count
                obs_j = self.landmarks[j].obs_count

                if obs_i > obs_j * 2:
                    remove_indices.add(j)
                elif obs_j > obs_i * 2:
                    remove_indices.add(i)
                    break
                elif dist < self.merge_dist:
                    if obs_i >= obs_j:
                        remove_indices.add(j)
                    else:
                        remove_indices.add(i)
                        break

        for idx in sorted(remove_indices, reverse=True):
            state_idx = 3 + 2 * idx
            self.mu = np.delete(self.mu, [state_idx, state_idx + 1])
            self.landmarks.pop(idx)

        if remove_indices:
            self.get_logger().info(
                f'Cleanup: removed {len(remove_indices)} duplicates | '
                f'Remaining: {len(self.landmarks)} landmarks'
            )

    # ═════════════════════════════════════════════════════════════════════
    # PUBLISHERS
    # ═════════════════════════════════════════════════════════════════════
    def publish_tf(self):
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id  = 'odom'
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    def publish_landmarks(self):
        marker_array = MarkerArray()
        clear        = Marker()
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        color_map = {
            0: (1.0, 1.0, 1.0),
            1: (1.0, 1.0, 0.0),
            2: (0.0, 0.0, 1.0),
            3: (1.0, 0.5, 0.0),
        }

        now = self.get_clock().now().to_msg()

        for j, lm in enumerate(self.landmarks):
            lm_idx = 3 + 2 * j
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp    = now
            m.ns              = 'slam_landmarks'
            m.id              = lm.id
            m.type            = Marker.CYLINDER
            m.action          = Marker.ADD
            m.pose.position.x = float(self.mu[lm_idx])
            m.pose.position.y = float(self.mu[lm_idx + 1])
            m.pose.position.z = 0.15
            m.pose.orientation.w = 1.0
            m.scale.x         = self.landmark_diameter
            m.scale.y         = self.landmark_diameter
            m.scale.z         = self.landmark_height

            rgb       = color_map.get(lm.color_id, (1.0, 1.0, 1.0))
            m.color.r = rgb[0]
            m.color.g = rgb[1]
            m.color.b = rgb[2]
            m.color.a = 1.0
            marker_array.markers.append(m)

        self.landmark_pub.publish(marker_array)

    def publish_occupancy_grid(self):
        if not self.landmarks:
            return

        grid_cols     = int(self.grid_width  / self.grid_res)
        grid_rows     = int(self.grid_height / self.grid_res)
        origin_x      = self.mu[0] - self.grid_width  / 2.0
        origin_y      = self.mu[1] - self.grid_height / 2.0
        inflate_cells = int(math.ceil(self.cone_inflate / self.grid_res))

        grid = np.zeros((grid_rows, grid_cols), dtype=np.int8)

        for j in range(len(self.landmarks)):
            lm_idx = 3 + 2 * j
            col    = int((self.mu[lm_idx]     - origin_x) / self.grid_res)
            row    = int((self.mu[lm_idx + 1] - origin_y) / self.grid_res)

            for dr in range(-inflate_cells, inflate_cells + 1):
                for dc in range(-inflate_cells, inflate_cells + 1):
                    r = row + dr
                    c = col + dc
                    if 0 <= r < grid_rows and 0 <= c < grid_cols:
                        if math.hypot(dr, dc) * self.grid_res <= self.cone_inflate:
                            grid[r, c] = 100

        msg                        = OccupancyGrid()
        msg.header.stamp           = self.get_clock().now().to_msg()
        msg.header.frame_id        = 'map'
        msg.info                   = MapMetaData()
        msg.info.resolution        = float(self.grid_res)
        msg.info.width             = grid_cols
        msg.info.height            = grid_rows
        msg.info.origin.position.x = origin_x
        msg.info.origin.position.y = origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data                   = grid.flatten().tolist()
        self.grid_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MOSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
