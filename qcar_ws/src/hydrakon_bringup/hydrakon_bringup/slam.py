#!/usr/bin/env python3
"""
EKF Landmark SLAM Node (v3)
-----------------------------
Simplified: everything operates in the odom frame.
Since CARLA odom doesn't drift, odom ≈ map.
Publishes an identity map→odom TF so the full TF tree stays connected.

Once this produces a clean map, we add proper map→odom correction.

Subscriptions:
    /odom (nav_msgs/Odometry) — vehicle motion for prediction step
    /fusion/observations (hydrakon_msgs/ConeObservationArray) — fused cone observations in odom frame

Publications:
    /map (nav_msgs/OccupancyGrid) — inflated cone map
    /slam/landmarks (visualization_msgs/MarkerArray) — cone map for rviz
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
from tf2_ros import TransformBroadcaster

from hydrakon_msgs.msg import ConeObservationArray


class Landmark:
    """Metadata for a landmark."""
    def __init__(self, landmark_id, color_id, color_probs):
        self.id = landmark_id
        self.color_id = color_id
        self.color_probs = np.array(color_probs, dtype=np.float64)
        self.obs_count = 0

    def update_color(self, color_id, color_probs):
        incoming = np.array(color_probs, dtype=np.float64)
        self.color_probs *= incoming
        total = np.sum(self.color_probs)
        if total > 0:
            self.color_probs /= total
        self.color_id = int(np.argmax(self.color_probs))


class EKFSLAMNode(Node):
    def __init__(self):
        super().__init__('ekf_slam_node')

        # ── Parameters ──────────────────────────────────────────────────
        # Motion model noise (low — CARLA odom is clean)
        self.declare_parameter('alpha1', 0.001)
        self.declare_parameter('alpha2', 0.0005)
        self.declare_parameter('alpha3', 0.001)
        self.declare_parameter('alpha4', 0.0005)

        # Data association
        self.declare_parameter('association_threshold', 2.0)
        self.declare_parameter('new_landmark_min_obs', 5)
        self.declare_parameter('color_gate_enabled', True)

        # Observation noise — fixed, overrides tight fusion covariances
        self.declare_parameter('obs_noise_std', 0.5)

        # Occupancy grid
        self.declare_parameter('grid_width', 200.0)
        self.declare_parameter('grid_height', 200.0)
        self.declare_parameter('grid_resolution', 0.1)
        self.declare_parameter('cone_inflation_radius', 0.3)
        self.declare_parameter('grid_publish_rate', 2.0)

        # Read parameters
        self.alpha1 = self.get_parameter('alpha1').value
        self.alpha2 = self.get_parameter('alpha2').value
        self.alpha3 = self.get_parameter('alpha3').value
        self.alpha4 = self.get_parameter('alpha4').value

        self.assoc_thresh = self.get_parameter('association_threshold').value
        self.min_obs = self.get_parameter('new_landmark_min_obs').value
        self.color_gate = self.get_parameter('color_gate_enabled').value
        self.obs_noise_std = self.get_parameter('obs_noise_std').value

        self.grid_width = self.get_parameter('grid_width').value
        self.grid_height = self.get_parameter('grid_height').value
        self.grid_res = self.get_parameter('grid_resolution').value
        self.cone_inflate = self.get_parameter('cone_inflation_radius').value
        grid_rate = self.get_parameter('grid_publish_rate').value

        # ── EKF State ──────────────────────────────────────────────────
        # State: [x, y, theta, lm1_x, lm1_y, lm2_x, lm2_y, ...]
        # Everything in odom frame
        self.mu = np.zeros(3)
        self.sigma = np.diag([0.001, 0.001, 0.001])

        self.landmarks = []
        self.next_landmark_id = 0
        self.pending = []

        # ── Odometry tracking ──────────────────────────────────────────
        self.prev_odom_x = None
        self.prev_odom_y = None
        self.prev_odom_theta = None
        self.odom_initialized = False

        # ── Publishers ─────────────────────────────────────────────────
        grid_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.grid_pub = self.create_publisher(OccupancyGrid, '/map', grid_qos)
        self.landmark_pub = self.create_publisher(MarkerArray, '/slam/landmarks', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ── Subscribers ────────────────────────────────────────────────
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(
            ConeObservationArray, '/fusion/observations',
            self.observation_callback, 10
        )

        # ── Timers ─────────────────────────────────────────────────────
        self.create_timer(1.0 / grid_rate, self.publish_occupancy_grid)
        self.create_timer(0.1, self.publish_landmarks)
        self.create_timer(0.05, self.publish_tf)

        self.get_logger().info('EKF SLAM v3 started (odom frame, identity map→odom)')

    # ═════════════════════════════════════════════════════════════════════
    # PREDICTION STEP
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

        # Relative motion
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
    # UPDATE STEP
    # ═════════════════════════════════════════════════════════════════════
    def observation_callback(self, msg: ConeObservationArray):
        if not self.odom_initialized:
            return

        # Fixed observation noise
        R = np.eye(2) * (self.obs_noise_std ** 2)

        for cone in msg.cones:
            obs_x = cone.x  # Already in odom frame
            obs_y = cone.y
            color_id = cone.color_id
            color_probs = list(cone.color_probs)
            obs_count = cone.observation_count

            if obs_count < 2:
                continue

            # ── Data Association (Euclidean + color gate) ──
            best_idx = -1
            best_dist = self.assoc_thresh

            for j in range(len(self.landmarks)):
                lm = self.landmarks[j]

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
                self._ekf_update(best_idx, obs_x, obs_y, R,
                                 color_id, color_probs)
            else:
                promoted = self._try_promote_pending(
                    obs_x, obs_y, color_id, color_probs, R
                )
                if not promoted:
                    self._update_pending(
                        obs_x, obs_y, color_id, color_probs, R
                    )

    def _ekf_update(self, lm_index, obs_x, obs_y, obs_cov,
                    color_id, color_probs):
        lm_idx = 3 + 2 * lm_index
        lm_x = self.mu[lm_idx]
        lm_y = self.mu[lm_idx + 1]

        innov = np.array([obs_x - lm_x, obs_y - lm_y])

        N = self.sigma.shape[0]
        H = np.zeros((2, N))
        H[0, lm_idx] = 1.0
        H[1, lm_idx + 1] = 1.0

        S = H @ self.sigma @ H.T + obs_cov

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        K = self.sigma @ H.T @ S_inv

        self.mu = self.mu + K @ innov
        self.mu[2] = self._normalize_angle(self.mu[2])

        # Joseph form
        I_KH = np.eye(N) - K @ H
        self.sigma = I_KH @ self.sigma @ I_KH.T + K @ obs_cov @ K.T

        # Symmetrize
        self.sigma = (self.sigma + self.sigma.T) / 2.0

        self.landmarks[lm_index].obs_count += 1
        self.landmarks[lm_index].update_color(color_id, color_probs)

    # ═════════════════════════════════════════════════════════════════════
    # LANDMARK MANAGEMENT
    # ═════════════════════════════════════════════════════════════════════
    def _update_pending(self, x, y, color_id, color_probs, cov):
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
                    p['color_probs'] = color_probs
                return

        self.pending.append({
            'x': x, 'y': y,
            'color_id': color_id,
            'color_probs': color_probs,
            'obs_count': 1,
            'cov': cov,
        })

    def _try_promote_pending(self, x, y, color_id, color_probs, cov):
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
                    p['color_probs'] = color_probs

                if p['obs_count'] >= self.min_obs:
                    # Check it doesn't duplicate an existing landmark
                    for j in range(len(self.landmarks)):
                        lm_idx = 3 + 2 * j
                        lm_dist = math.hypot(
                            self.mu[lm_idx] - p['x'],
                            self.mu[lm_idx + 1] - p['y']
                        )
                        if lm_dist < self.assoc_thresh:
                            self.pending.pop(i)
                            return True

                    self._add_landmark(
                        p['x'], p['y'],
                        p['color_id'], p['color_probs'], p['cov']
                    )
                    self.pending.pop(i)
                    return True
                return False
        return False

    def _add_landmark(self, x, y, color_id, color_probs, obs_cov):
        lm = Landmark(self.next_landmark_id, color_id, color_probs)
        lm.obs_count = 1
        self.next_landmark_id += 1
        self.landmarks.append(lm)

        self.mu = np.append(self.mu, [x, y])

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
            f'New landmark #{lm.id} at ({x:.1f}, {y:.1f}) '
            f'color={color_id} | total: {len(self.landmarks)} | '
            f'state dim: {len(self.mu)}'
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
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    def publish_landmarks(self):
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

        for j, lm in enumerate(self.landmarks):
            lm_idx = 3 + 2 * j
            mx = self.mu[lm_idx]
            my = self.mu[lm_idx + 1]

            m = Marker()
            m.header.frame_id = 'odom'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'slam_landmarks'
            m.id = lm.id
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(mx)
            m.pose.position.y = float(my)
            m.pose.position.z = 0.15
            m.pose.orientation.w = 1.0
            m.scale.x = 0.3
            m.scale.y = 0.3
            m.scale.z = 0.3
            rgb = color_map.get(lm.color_id, (1.0, 1.0, 1.0))
            m.color.r = rgb[0]
            m.color.g = rgb[1]
            m.color.b = rgb[2]
            m.color.a = 1.0
            marker_array.markers.append(m)

            # Covariance ellipse
            cov_xx = self.sigma[lm_idx, lm_idx]
            cov_yy = self.sigma[lm_idx + 1, lm_idx + 1]

            e = Marker()
            e.header.frame_id = 'odom'
            e.header.stamp = self.get_clock().now().to_msg()
            e.ns = 'slam_covariance'
            e.id = lm.id + 10000
            e.type = Marker.CYLINDER
            e.action = Marker.ADD
            e.pose.position.x = float(mx)
            e.pose.position.y = float(my)
            e.pose.position.z = 0.01
            e.pose.orientation.w = 1.0
            e.scale.x = float(4.0 * math.sqrt(max(cov_xx, 1e-6)))
            e.scale.y = float(4.0 * math.sqrt(max(cov_yy, 1e-6)))
            e.scale.z = 0.01
            e.color.r = rgb[0]
            e.color.g = rgb[1]
            e.color.b = rgb[2]
            e.color.a = 0.3
            marker_array.markers.append(e)

        self.landmark_pub.publish(marker_array)

    def publish_occupancy_grid(self):
        if len(self.landmarks) == 0:
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
                    if dr * dr + dc * dc <= inflate_cells * inflate_cells:
                        r, c = row + dr, col + dc
                        if 0 <= r < grid_rows and 0 <= c < grid_cols:
                            grid[r][c] = 100

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.info = MapMetaData()
        msg.info.resolution = self.grid_res
        msg.info.width = grid_cols
        msg.info.height = grid_rows
        msg.info.origin.position.x = origin_x
        msg.info.origin.position.y = origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten().tolist()

        self.grid_pub.publish(msg)

    # ═════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═════════════════════════════════════════════════════════════════════
    @staticmethod
    def _normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def _quat_to_yaw(qx, qy, qz, qw):
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny, cosy)


def main(args=None):
    rclpy.init(args=args)
    node = EKFSLAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

