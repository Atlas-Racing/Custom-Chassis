#!/usr/bin/env python3
"""
MO-SLAM: Merged Observation SLAM Node
---------------------------------------
Camera-only EKF SLAM for the qcar platform.

Directly consumes camera cone detections, performs:
1. Transform to odom frame
2. EKF SLAM with persistent landmarks (joint state vector)
3. Occupancy grid + landmark marker publishing

Subscriptions:
    /hydrakon_camera/cone_markers (MarkerArray) — camera cone detections in base_link frame
    /zed/zed_node/odom (Odometry) — ZED positional tracking for EKF prediction

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
    """Metadata for a confirmed EKF landmark."""
    def __init__(self, landmark_id, color_id):
        self.id = landmark_id
        self.color_id = color_id  # 0=unknown, 1=yellow, 2=blue, 3=orange
        self.obs_count = 0

    def update_color(self, color_id):
        """Simple majority: overwrite unknown with known color."""
        if color_id != 0 and self.color_id == 0:
            self.color_id = color_id
        elif color_id != 0 and self.color_id != 0:
            # Only overwrite if we've seen this color more — for now, keep first
            pass


# ═════════════════════════════════════════════════════════════════════════
# MO-SLAM NODE
# ═════════════════════════════════════════════════════════════════════════
class MOSlamNode(Node):
    def __init__(self):
        super().__init__('mo_slam_node')

        # ── EKF Motion Model Noise ──────────────────────────────────────
        self.declare_parameter('alpha1', 0.001)
        self.declare_parameter('alpha2', 0.0005)
        self.declare_parameter('alpha3', 0.001)
        self.declare_parameter('alpha4', 0.0005)

        # ── Data Association ────────────────────────────────────────────
        self.declare_parameter('association_threshold', 0.05)
        self.declare_parameter('new_landmark_min_obs', 3)
        self.declare_parameter('color_gate_enabled', True)

        # ── Observation Noise ───────────────────────────────────────────
        self.declare_parameter('camera_obs_noise', 0.2)

        # ── Occupancy Grid ──────────────────────────────────────────────
        self.declare_parameter('grid_width', 200.0)
        self.declare_parameter('grid_height', 200.0)
        self.declare_parameter('grid_resolution', 0.1)
        self.declare_parameter('cone_inflation_radius', 0.04)
        self.declare_parameter('grid_publish_rate', 2.0)
        self.declare_parameter('landmark_publish_rate', 10.0)

        # ── Landmark Marker Dimensions ──────────────────────────────────
        self.declare_parameter('landmark_diameter', 0.04)
        self.declare_parameter('landmark_height', 0.06)

        # ── Read All Parameters ─────────────────────────────────────────
        self.alpha1 = self.get_parameter('alpha1').value
        self.alpha2 = self.get_parameter('alpha2').value
        self.alpha3 = self.get_parameter('alpha3').value
        self.alpha4 = self.get_parameter('alpha4').value

        self.assoc_thresh = self.get_parameter('association_threshold').value
        self.min_obs = self.get_parameter('new_landmark_min_obs').value
        self.color_gate = self.get_parameter('color_gate_enabled').value

        self.camera_noise = self.get_parameter('camera_obs_noise').value

        self.grid_width = self.get_parameter('grid_width').value
        self.grid_height = self.get_parameter('grid_height').value
        self.grid_res = self.get_parameter('grid_resolution').value
        self.cone_inflate = self.get_parameter('cone_inflation_radius').value
        grid_rate = self.get_parameter('grid_publish_rate').value
        landmark_rate = self.get_parameter('landmark_publish_rate').value
        self.landmark_diameter = self.get_parameter('landmark_diameter').value
        self.landmark_height = self.get_parameter('landmark_height').value

        # ── Landmark Merge ──────────────────────────────────────────────
        self.declare_parameter('merge_distance', 1.2)
        self.declare_parameter('merge_trigger_count', 200)
        self.merge_dist = self.get_parameter('merge_distance').value
        self.merge_trigger = self.get_parameter('merge_trigger_count').value
        self.merge_requested = False
        self.create_timer(10.0, self._periodic_merge)

        # ── TF ──────────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ── EKF State ──────────────────────────────────────────────────
        # State: [x, y, theta, lm1_x, lm1_y, lm2_x, lm2_y, ...]
        self.mu = np.zeros(3)
        self.sigma = np.diag([0.001, 0.001, 0.001])
        self.landmarks = []
        self.next_landmark_id = 0
        self.pending = []

        # ── Odometry Tracking ──────────────────────────────────────────
        self.prev_odom_x = None
        self.prev_odom_y = None
        self.prev_odom_theta = None
        self.odom_initialized = False

        self.cam_msg_count = 0
        self.cam_process_every = 3
        self.pending_updates = []

        # ── Publishers ─────────────────────────────────────────────────
        grid_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.grid_pub = self.create_publisher(OccupancyGrid, '/map', grid_qos)
        self.landmark_pub = self.create_publisher(MarkerArray, '/map/landmarks', 10)

        # ── Subscribers ────────────────────────────────────────────────
        self.create_subscription(MarkerArray, '/hydrakon_camera/cone_markers', self.camera_callback, 10)
        self.create_subscription(Odometry, '/zed/zed_node/odom', self.odom_callback, 10)

        # ── Timers ─────────────────────────────────────────────────────
        self.create_timer(1.0 / grid_rate, self.publish_occupancy_grid)
        self.create_timer(1.0 / landmark_rate, self.publish_landmarks)
        self.create_timer(0.05, self.publish_tf)

        self.get_logger().info(
            f'MO-SLAM started | '
            f'camera_noise={self.camera_noise} | '
            f'assoc={self.assoc_thresh}m | min_obs={self.min_obs}'
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
    # EKF PREDICTION (from odometry)
    # ═════════════════════════════════════════════════════════════════════
    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = self._quat_to_yaw(q.x, q.y, q.z, q.w)

        if not self.odom_initialized:
            self.prev_odom_x = x
            self.prev_odom_y = y
            self.prev_odom_theta = theta
            self.mu[0] = x
            self.mu[1] = y
            self.mu[2] = theta
            self.odom_initialized = True
            return

        dx = x - self.prev_odom_x
        dy = y - self.prev_odom_y
        dtheta = self._normalize_angle(theta - self.prev_odom_theta)

        self.prev_odom_x = x
        self.prev_odom_y = y
        self.prev_odom_theta = theta

        d_trans = math.hypot(dx, dy)
        if d_trans < 1e-6 and abs(dtheta) < 1e-6:
            return

        # Apply motion to vehicle state
        self.mu[0] += dx
        self.mu[1] += dy
        self.mu[2] = self._normalize_angle(self.mu[2] + dtheta)

        # Process noise
        var_trans = self.alpha3 * d_trans**2 + self.alpha4 * dtheta**2
        var_rot = self.alpha1 * dtheta**2 + self.alpha2 * d_trans**2
        var_trans = max(var_trans, 1e-8)
        var_rot = max(var_rot, 1e-8)

        R = np.diag([var_trans, var_trans, var_rot])

        N = self.sigma.shape[0]
        G = np.eye(N)
        G[0, 2] = -dy
        G[1, 2] = dx

        self.sigma = G @ self.sigma @ G.T
        self.sigma[:3, :3] += R

    # ═════════════════════════════════════════════════════════════════════
    # CAMERA PIPELINE (transform → color → EKF update)
    # ═════════════════════════════════════════════════════════════════════
    def camera_callback(self, msg):
        self.cam_msg_count += 1
        if self.cam_msg_count % self.cam_process_every != 0:
            return

        if not self.odom_initialized:
            return
        if not msg.markers:
            return

        source_frame = msg.markers[0].header.frame_id
        transform = self.get_transform("odom", source_frame)
        if not transform:
            return

        R_cam = np.eye(2) * (self.camera_noise ** 2)

        for marker in msg.markers:
            if marker.action == Marker.DELETEALL:
                continue

            p_stamped = tf2_geometry_msgs.PointStamped()
            p_stamped.point.x = marker.pose.position.x
            p_stamped.point.y = marker.pose.position.y
            p_stamped.point.z = marker.pose.position.z

            try:
                p_odom = tf2_geometry_msgs.do_transform_point(p_stamped, transform)

                # Extract color from marker RGBA
                r, g, b = marker.color.r, marker.color.g, marker.color.b
                color_id = 0
                if r > 0.9 and g > 0.9:
                    color_id = 1  # Yellow
                elif b > 0.9:
                    color_id = 2  # Blue
                elif r > 0.9 and g > 0.4:
                    color_id = 3  # Orange

                self._process_observation(
                    p_odom.point.x, p_odom.point.y,
                    R_cam, color_id=color_id
                )
            except Exception:
                continue

        self._flush_ekf_updates()

    # ═════════════════════════════════════════════════════════════════════
    # UNIFIED OBSERVATION PROCESSING
    # ═════════════════════════════════════════════════════════════════════
    def _process_observation(self, obs_x, obs_y, R, color_id):
        """
        Single entry point for all observations.
        Associates with existing landmarks or adds to pending.
        """
        # ── Data Association (Euclidean + color gate) ──
        best_idx = -1
        best_dist = self.assoc_thresh

        for j, lm in enumerate(self.landmarks):
            # Color gate
            if self.color_gate and color_id != 0 and lm.color_id != 0:
                if color_id != lm.color_id:
                    continue

            lm_idx = 3 + 2 * j
            lm_x = self.mu[lm_idx]
            lm_y = self.mu[lm_idx + 1]
            dist = math.hypot(obs_x - lm_x, obs_y - lm_y)

            if dist < best_dist:
                best_dist = dist
                best_idx = j

        if best_idx >= 0:
            self.pending_updates.append((best_idx, obs_x, obs_y, R, color_id))
        else:
            promoted = self._try_promote_pending(obs_x, obs_y, color_id, R)
            if not promoted:
                self._add_pending(obs_x, obs_y, color_id, R)

    # ═════════════════════════════════════════════════════════════════════
    # EKF BATCH UPDATE
    # ═════════════════════════════════════════════════════════════════════
    def _flush_ekf_updates(self):
        if not self.pending_updates:
            return

        n_obs = len(self.pending_updates)
        N = self.sigma.shape[0]

        H = np.zeros((2 * n_obs, N))
        innov = np.zeros(2 * n_obs)
        R_block = np.zeros((2 * n_obs, 2 * n_obs))

        for k, (lm_index, obs_x, obs_y, obs_cov, color_id) in enumerate(self.pending_updates):
            lm_idx = 3 + 2 * lm_index
            lm_x = self.mu[lm_idx]
            lm_y = self.mu[lm_idx + 1]

            row = 2 * k
            H[row, lm_idx] = 1.0
            H[row + 1, lm_idx + 1] = 1.0
            innov[row] = obs_x - lm_x
            innov[row + 1] = obs_y - lm_y
            R_block[row:row + 2, row:row + 2] = obs_cov

            self.landmarks[lm_index].obs_count += 1
            self.landmarks[lm_index].update_color(color_id)

        S = H @ self.sigma @ H.T + R_block
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.pending_updates.clear()
            return

        K = self.sigma @ H.T @ S_inv
        self.mu = self.mu + K @ innov
        self.mu[2] = self._normalize_angle(self.mu[2])

        I_KH = np.eye(N) - K @ H
        self.sigma = I_KH @ self.sigma @ I_KH.T + K @ R_block @ K.T
        self.sigma = (self.sigma + self.sigma.T) / 2.0

        self.pending_updates.clear()

        if self.merge_requested:
            self._merge_landmarks()
            self.merge_requested = False

    # ═════════════════════════════════════════════════════════════════════
    # LANDMARK MANAGEMENT
    # ═════════════════════════════════════════════════════════════════════
    def _add_pending(self, x, y, color_id, cov):
        """Add observation to pending list or update existing pending entry."""
        for p in self.pending:
            dist = math.hypot(p['x'] - x, p['y'] - y)
            color_ok = (color_id == 0 or p['color_id'] == 0 or
                        color_id == p['color_id'])
            if dist < self.assoc_thresh and color_ok:
                n = p['obs_count']
                p['x'] = (p['x'] * n + x) / (n + 1)
                p['y'] = (p['y'] * n + y) / (n + 1)
                p['obs_count'] += 1
                if color_id != 0:
                    p['color_id'] = color_id
                return

        self.pending.append({
            'x': x, 'y': y,
            'color_id': color_id,
            'obs_count': 1,
            'cov': cov.copy(),
        })

    def _try_promote_pending(self, x, y, color_id, cov):
        """Check if a pending entry is ready for promotion to landmark."""
        for i, p in enumerate(self.pending):
            dist = math.hypot(p['x'] - x, p['y'] - y)
            color_ok = (color_id == 0 or p['color_id'] == 0 or
                        color_id == p['color_id'])
            if dist < self.assoc_thresh and color_ok:
                p['obs_count'] += 1
                n = p['obs_count']
                p['x'] = (p['x'] * (n - 1) + x) / n
                p['y'] = (p['y'] * (n - 1) + y) / n
                if color_id != 0:
                    p['color_id'] = color_id

                if p['obs_count'] >= self.min_obs:
                    # Check no existing landmark is too close
                    for j in range(len(self.landmarks)):
                        lm_idx = 3 + 2 * j
                        lm_dist = math.hypot(
                            self.mu[lm_idx] - p['x'],
                            self.mu[lm_idx + 1] - p['y']
                        )
                        if lm_dist < self.assoc_thresh:
                            self.pending.pop(i)
                            return True

                    self._add_landmark(p['x'], p['y'], p['color_id'], p['cov'])
                    self.pending.pop(i)
                    return True
                return False
        return False

    def _add_landmark(self, x, y, color_id, obs_cov):
        """Promote a pending observation to a full EKF landmark."""
        lm = Landmark(self.next_landmark_id, color_id)
        lm.obs_count = 1
        self.next_landmark_id += 1
        self.landmarks.append(lm)

        # Extend state vector
        self.mu = np.append(self.mu, [x, y])

        # Extend covariance matrix
        N = self.sigma.shape[0]
        sigma_new = np.zeros((N + 2, N + 2))
        sigma_new[:N, :N] = self.sigma
        sigma_new[N, N] = self.sigma[0, 0] + obs_cov[0, 0]
        sigma_new[N + 1, N + 1] = self.sigma[1, 1] + obs_cov[1, 1]
        sigma_new[N, :3] = self.sigma[0, :3]
        sigma_new[N + 1, :3] = self.sigma[1, :3]
        sigma_new[:3, N] = self.sigma[:3, 0]
        sigma_new[:3, N + 1] = self.sigma[:3, 1]
        self.sigma = sigma_new

        self.get_logger().info(
            f'Landmark #{lm.id} at ({x:.1f}, {y:.1f}) '
            f'color={color_id} | total: {len(self.landmarks)} | '
            f'state_dim: {len(self.mu)}'
        )

    def _periodic_merge(self):
        if len(self.landmarks) >= self.merge_trigger:
            self.merge_requested = True

    def _merge_landmarks(self):
        """Remove low-observation landmarks near high-observation ones."""
        if len(self.landmarks) < 2:
            return

        removed_count = 0
        remove_indices = set()

        for i in range(len(self.landmarks)):
            if i in remove_indices:
                continue
            for j in range(i + 1, len(self.landmarks)):
                if j in remove_indices:
                    continue

                idx_i = 3 + 2 * i
                idx_j = 3 + 2 * j
                dist = math.hypot(
                    self.mu[idx_i] - self.mu[idx_j],
                    self.mu[idx_i + 1] - self.mu[idx_j + 1]
                )

                if dist > 2.0:
                    continue

                # Color gate
                ci = self.landmarks[i].color_id
                cj = self.landmarks[j].color_id
                if ci != 0 and cj != 0 and ci != cj:
                    continue

                obs_i = self.landmarks[i].obs_count
                obs_j = self.landmarks[j].obs_count

                # Remove the one with significantly fewer observations
                if obs_i > obs_j * 2:
                    remove_indices.add(j)
                elif obs_j > obs_i * 2:
                    remove_indices.add(i)
                    break  # i is marked, move to next i
                elif dist < self.merge_dist:
                    # Similar obs counts but very close — merge into higher one
                    if obs_i >= obs_j:
                        remove_indices.add(j)
                    else:
                        remove_indices.add(i)
                        break

        # Remove in reverse order to keep indices valid
        for idx in sorted(remove_indices, reverse=True):
            state_idx = 3 + 2 * idx
            self.mu = np.delete(self.mu, [state_idx, state_idx + 1])
            self.sigma = np.delete(self.sigma, [state_idx, state_idx + 1], axis=0)
            self.sigma = np.delete(self.sigma, [state_idx, state_idx + 1], axis=1)
            self.landmarks.pop(idx)

        removed_count = len(remove_indices)
        if removed_count > 0:
            self.get_logger().info(
                f'Cleanup: removed {removed_count} duplicates | '
                f'Remaining: {len(self.landmarks)} landmarks'
            )

    # ═════════════════════════════════════════════════════════════════════
    # PUBLISHERS
    # ═════════════════════════════════════════════════════════════════════
    def publish_tf(self):
        """Publish identity map → odom transform."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    def publish_landmarks(self):
        """Publish confirmed landmarks as colored cylinders."""
        marker_array = MarkerArray()
        clear = Marker()
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
            mx = self.mu[lm_idx]
            my = self.mu[lm_idx + 1]

            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = now
            m.ns = 'slam_landmarks'
            m.id = lm.id
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(mx)
            m.pose.position.y = float(my)
            m.pose.position.z = 0.15
            m.pose.orientation.w = 1.0
            m.scale.x = self.landmark_diameter
            m.scale.y = self.landmark_diameter
            m.scale.z = self.landmark_height

            rgb = color_map.get(lm.color_id, (1.0, 1.0, 1.0))
            m.color.r = rgb[0]
            m.color.g = rgb[1]
            m.color.b = rgb[2]
            m.color.a = 1.0
            marker_array.markers.append(m)

        self.landmark_pub.publish(marker_array)

    def publish_occupancy_grid(self):
        """Publish inflated cone obstacles."""
        if not self.landmarks:
            return

        grid_cols = int(self.grid_width / self.grid_res)
        grid_rows = int(self.grid_height / self.grid_res)
        origin_x = -self.grid_width / 2.0
        origin_y = -self.grid_height / 2.0
        inflate_cells = int(math.ceil(self.cone_inflate / self.grid_res))

        grid = np.zeros((grid_rows, grid_cols), dtype=np.int8)

        for j in range(len(self.landmarks)):
            lm_idx = 3 + 2 * j
            mx = self.mu[lm_idx]
            my = self.mu[lm_idx + 1]

            col = int((mx - origin_x) / self.grid_res)
            row = int((my - origin_y) / self.grid_res)

            for dr in range(-inflate_cells, inflate_cells + 1):
                for dc in range(-inflate_cells, inflate_cells + 1):
                    r = row + dr
                    c = col + dc
                    if 0 <= r < grid_rows and 0 <= c < grid_cols:
                        if math.hypot(dr, dc) * self.grid_res <= self.cone_inflate:
                            grid[r, c] = 100

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info = MapMetaData()
        msg.info.resolution = float(self.grid_res)
        msg.info.width = grid_cols
        msg.info.height = grid_rows
        msg.info.origin.position.x = origin_x
        msg.info.origin.position.y = origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten().tolist()
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
