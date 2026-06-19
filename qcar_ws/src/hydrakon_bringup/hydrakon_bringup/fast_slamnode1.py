"""
FastSLAM Node for HydrakonSimV2
Subscribes to /fusion/cone_markers and /odom
Runs FastSLAM to estimate vehicle pose + cone map
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
import math
import numpy as np

# --- TUNED CONSTANTS ---
N_PARTICLE = 100
DT = 0.02  # 50Hz odom rate
WHEELBASE = 1.535
MAX_STEER = np.deg2rad(21.0)
RESAMPLE_NEFF_RATIO = 0.5
EPSILON = 1e-6
# M_DIST_TH = 2.0
M_DIST_TH = 2.5

# Noise Matrices
# Low noise since we have ground truth odom from CARLA

Q_MEAS = np.diag([0.7, np.deg2rad(8.0)])**2
R_MOTION = np.diag([0.05, np.deg2rad(1.0)])**2  # motion noise (kept for reference)

def pi_2_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class Particle:
    """A 'ghost car' representing one possible location and its own map of the track."""
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.w   = 1.0 / N_PARTICLE
        self.x   = x
        self.y   = y
        self.yaw = yaw
        self.map = {'blue': [], 'yellow': [], 'orange': [], 'unknown': []}
        self.cov = {'blue': [], 'yellow': [], 'orange': [], 'unknown': []}
        self.counts = {'blue': [], 'yellow': [], 'orange': [], 'unknown': []}
        self.prev_odom_x = None
        self.prev_odom_y = None
        self.prev_odom_yaw = None

def motion_model(state, u):
    """Kept for reference but no longer used in predict_particles."""
    x, y, yaw = state
    v     = u[0]
    delta = np.clip(u[1], -MAX_STEER, MAX_STEER)
    beta  = math.atan(0.5 * math.tan(delta))
    new_x   = x   + v * math.cos(yaw + beta) * DT
    new_y   = y   + v * math.sin(yaw + beta) * DT
    new_yaw = yaw + (v / WHEELBASE) * math.cos(beta) * math.tan(delta) * DT
    return np.array([new_x, new_y, pi_2_pi(new_yaw)])


def predict_particles(particles, dx, dy, dyaw):

    for p in particles:

        noisy_dx = dx + np.random.randn() * 0.01
        noisy_dy = dy + np.random.randn() * 0.01
        noisy_dyaw = dyaw + np.random.randn() * np.deg2rad(0.3)

        p.x += noisy_dx
        p.y += noisy_dy
        p.yaw = pi_2_pi(p.yaw + noisy_dyaw)

    return particles


def predict_measurement(particle, lm_pos):
    """
    Predicts measurement (range, bearing) and Jacobian Hf
    for a given particle pose and landmark position.
    """
    dx = lm_pos[0] - particle.x
    dy = lm_pos[1] - particle.y
    d2 = dx**2 + dy**2
    d  = math.sqrt(d2) if d2 > EPSILON else math.sqrt(EPSILON)

    z_pred = np.array([
        d,
        pi_2_pi(math.atan2(dy, dx) - particle.yaw)
    ]).reshape(2, 1)

    Hf = np.array([
        [ dx / d,   dy / d  ],
        [-dy / d2,  dx / d2 ],
    ])
    return z_pred, Hf


def mahalanobis_distance(particle, obs_r, obs_b, lm_idx, color):
    """
    Mahalanobis distance between observation and landmark in measurement space.
    Accounts for landmark uncertainty via innovation covariance S.
    """
    lm_pos = particle.map[color][lm_idx]
    Pf     = particle.cov[color][lm_idx]

    z_pred, Hf = predict_measurement(particle, lm_pos)

    dz    = np.array([obs_r, obs_b]).reshape(2, 1) - z_pred
    dz[1] = pi_2_pi(float(dz[1]))

    S = Hf @ Pf @ Hf.T + Q_MEAS

    try:
        dist = float(dz.T @ np.linalg.inv(S) @ dz)
        return math.sqrt(max(dist, 0.0))
    except np.linalg.LinAlgError:
        return float('inf')


def init_landmark(particle, z, color):
    """
    First observation of a cone — convert polar (r, b) to global (x, y)
    using particle pose. Initial covariance via Jacobian Gz.
    """
    r, b = z
    phi  = pi_2_pi(particle.yaw + b)
    lx   = particle.x + r * math.cos(phi)
    ly   = particle.y + r * math.sin(phi)

    # Jacobian of polar→cartesian transform — projects Q_MEAS into map frame
    Gz = np.array([
        [math.cos(phi), -r * math.sin(phi)],
        [math.sin(phi),  r * math.cos(phi)]
    ])
    Pf = Gz @ Q_MEAS @ Gz.T

    particle.map[color].append([lx, ly])
    particle.cov[color].append(Pf)
    particle.counts[color].append(1)


def ekf_landmark_update(particle, z_obs, color, lm_idx):
    """
    EKF update for a known landmark.
    Returns Gaussian likelihood weight for this particle.
    """
    lm_pos     = particle.map[color][lm_idx]
    Pf         = particle.cov[color][lm_idx]
    z_pred, Hf = predict_measurement(particle, lm_pos)

    dz = np.array(z_obs).reshape(2, 1) - z_pred
    dz[1, 0] = pi_2_pi(float(dz[1, 0]))

    S = Hf @ Pf @ Hf.T + Q_MEAS + np.eye(2) * EPSILON

    try:
        K      = Pf @ Hf.T @ np.linalg.inv(S)
        K      = np.clip(K, -0.5, 0.5)
        new_lm = np.array(lm_pos).reshape(2, 1) + K @ dz

        particle.map[color][lm_idx] = new_lm.flatten().tolist()
        if lm_idx >= len(particle.counts[color]):
            particle.counts[color].append(1)
        else:
            particle.counts[color][lm_idx] += 1

        # Joseph form — same as teammate's EKF SLAM, more numerically stable
        I_KH = np.eye(2) - K @ Hf
        particle.cov[color][lm_idx] = I_KH @ Pf @ I_KH.T + K @ Q_MEAS @ K.T

        MIN_COV = 0.05
        particle.cov[color][lm_idx][0,0] = max(
            particle.cov[color][lm_idx][0,0],
            MIN_COV
            )

        particle.cov[color][lm_idx][1,1] = max(
            particle.cov[color][lm_idx][1,1],
            MIN_COV
            )

        # Gaussian likelihood weight
        num = np.exp(-0.5 * float(dz.T @ np.linalg.solve(S, dz)))
        den = 2.0 * math.pi * math.sqrt(np.linalg.det(S))
        return num / (den + EPSILON)

    except np.linalg.LinAlgError:
        return 1e-9



def associate_landmark(particle, obs_r, obs_b, color):
    """
    Nearest-neighbour data association with Mahalanobis gating.
    Color-separated — only compares against landmarks of same color.
    Returns lm_idx >= 0 for match, -1 for new landmark.
    """
    if not particle.map[color]:
        return -1

    best_idx  = -1
    best_dist = float('inf')
    second_best = float('inf')

    for i in range(len(particle.map[color])):
        dist = mahalanobis_distance(particle, obs_r, obs_b, i, color)

        if dist < best_dist:
            second_best = best_dist
            best_dist = dist
            best_idx = i
        elif dist < second_best:
            second_best = dist

    # ambiguity check
    # gate = 2.0 if obs_r < 5.0 else 1.2
    gate = 1.8 if obs_r <6.0 else 1.2

    if best_dist > gate:
        return -1

    return best_idx


def update_with_observations(particles, observations):
    """
    Main update loop — for each particle x each observation:
    associate → init new landmark OR EKF update existing.
    """
    for particle in particles:
        for obs in observations:
            color    = obs['color']
            obs_r    = obs['r']
            obs_b    = obs['b']

            # Range gate — matches teammate's obs_noise_std=0.5 approach
            if obs_r > 8.0:
                continue

            lm_idx = associate_landmark(particle, obs_r, obs_b, color)

            if lm_idx == -1:
                init_landmark(particle, [obs_r, obs_b], color)
            else:
                w = ekf_landmark_update(particle, [obs_r, obs_b], color, lm_idx)
                particle.w *= w
                particle.w  = max(particle.w, 1e-300)

    # Normalise weights
    total_w = sum(p.w for p in particles)
    if total_w > 1e-12:
        for p in particles:
            p.w /= total_w
    else:
        for p in particles:
            p.w = 1.0 / N_PARTICLE

    return particles


def low_variance_resampling(particles):
    """Low-variance systematic resampling."""
    weights = np.array([p.w for p in particles])
    if np.sum(weights) < 1e-12:
        return particles

    weights /= np.sum(weights)

    n_eff = 1.0 / np.sum(np.square(weights))
    if n_eff > N_PARTICLE * RESAMPLE_NEFF_RATIO:
        return particles

    new_particles = []
    r = np.random.uniform(0, 1.0 / N_PARTICLE)
    c = weights[0]
    i = 0

    for m in range(N_PARTICLE):
        u = r + m / N_PARTICLE
        while u > c and i < N_PARTICLE - 1:
            i += 1
            c += weights[i]

        p_old = particles[i]
        p_new = Particle(p_old.x, p_old.y, p_old.yaw)
        p_new.w = 1.0 / N_PARTICLE
        # copy landmarks
        p_new.map = {
            k: [list(lm) for lm in v]
            for k, v in p_old.map.items()
        }

        # copy covariance
        p_new.cov = {
            k: [np.copy(cv) for cv in v]
            for k, v in p_old.cov.items()
        }

        # IMPORTANT: copy landmark observation counts
        p_new.counts = {
            k: list(cnts) for k, cnts in p_old.counts.items()
        }

        new_particles.append(p_new)

    return new_particles


class FastSLAMNode(Node):
    def __init__(self):
        super().__init__('fast_slam_node')

        self.declare_parameter('frame_id', 'base_footprint')
        self.declare_parameter('odom_frame_id', 'odom')
        self.frame_id      = self.get_parameter('frame_id').value
        self.odom_frame_id = self.get_parameter('odom_frame_id').value

        # Particle set
        self.particles = [Particle() for _ in range(N_PARTICLE)]
        self.u = np.array([0.0, 0.0])

        # Ground truth pose from odom — used directly in predict_particles
        self.current_x   = 0.0
        self.current_y   = 0.0
        self.current_yaw = 0.0

        # Counters
        self._odom_count = 0
        self._cone_count = 0

        # Subscribers
        self.create_subscription(
            MarkerArray, '/fusion/cone_markers',
            self.cone_callback, 10)
        self.create_subscription(
            Odometry, '/odom',
            self.odom_callback, 10)

        # Publishers
        self.pose_pub      = self.create_publisher(PoseStamped, '/slam/pose',      10)
        self.particles_pub = self.create_publisher(PoseArray,   '/slam/particles', 10)
        self.map_pub       = self.create_publisher(MarkerArray, '/slam/cone_map',  10)

        self.create_timer(5.0, self._status_log)
        self.get_logger().info("FastSLAM Node Started")

    def _status_log(self):
        self.get_logger().info(
            f"Status — odom: {self._odom_count}, cones: {self._cone_count}, "
            f"pos=({self.current_x:.2f},{self.current_y:.2f}) "
            f"yaw={math.degrees(self.current_yaw):.1f}deg"
        )

    # def odom_callback(self, msg: Odometry):
    #     self._odom_count += 1

    #     x = msg.pose.pose.position.x
    #     y = msg.pose.pose.position.y
    #     q = msg.pose.pose.orientation
    #     yaw = math.atan2(
    #         2.0 * (q.w * q.z + q.x * q.y),
    #         1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    #     )

    #     # Initialise particles at actual car position on first message
    #     if self._odom_count == 1:
    #         self.particles   = [Particle(x, y, yaw) for _ in range(N_PARTICLE)]
    #         self.current_x   = x
    #         self.current_y   = y
    #         self.current_yaw = yaw
    #         self.u = np.array([0.0, 0.0])
    #         self.get_logger().info(
    #             f"Particles initialised at ({x:.2f}, {y:.2f}, {math.degrees(yaw):.1f}deg)"
    #         )
    #         return

    #     # Store ground truth pose — used directly in predict_particles
    #     self.current_x   = x
    #     self.current_y   = y
    #     self.current_yaw = yaw

    #     if self._odom_count % 100 == 0:
    #         self.get_logger().info(
    #             f"/odom #{self._odom_count} — "
    #             f"pos=({x:.2f},{y:.2f}) yaw={math.degrees(yaw):.1f}deg"
    #         )

    def odom_callback(self, msg: Odometry):
        self._odom_count += 1

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        if self._odom_count == 1:
            self.particles   = [Particle(x, y, yaw) for _ in range(N_PARTICLE)]
            self.current_x   = x
            self.current_y   = y
            self.current_yaw = yaw
            self.get_logger().info(
                f"Particles initialised at ({x:.2f}, {y:.2f}, {math.degrees(yaw):.1f}deg)"
            )
            return

        # Calculate displacement from previous odom reading
        dx = x - self.current_x
        dy = y - self.current_y
        dyaw = pi_2_pi(yaw - self.current_yaw)

        self.current_x   = x
        self.current_y   = y
        self.current_yaw = yaw

        # Update particles immediately on every odom message
        # so yaw is always current when cone_callback fires
        self.particles = predict_particles(
            self.particles, dx, dy, dyaw
        )

        if self._odom_count % 100 == 0:
            self.get_logger().info(
                f"/odom #{self._odom_count} — "
                f"pos=({x:.2f},{y:.2f}) yaw={math.degrees(yaw):.1f}deg"
            )

    # def cone_callback(self, msg: MarkerArray):
    #     self._cone_count += 1
    #     total_markers = len(msg.markers)
    #     observations  = self._markers_to_obs(msg)

    #     if self._cone_count == 1:
    #         self.get_logger().info(
    #             f"First /fusion/cone_markers — "
    #             f"{total_markers} markers, {len(observations)} valid"
    #         )

    #     if not observations:
    #         if self._cone_count % 50 == 0:
    #             self.get_logger().warn(
    #                 f"cone_callback #{self._cone_count} — 0 valid obs, skipping"
    #             )
    #         return

    #     if self._cone_count % 50 == 0:
    #         best = max(self.particles, key=lambda p: p.w)
    #         total_cones = sum(len(v) for v in best.map.values())
    #         self.get_logger().info(
    #             f"SLAM update #{self._cone_count} — "
    #             f"{len(observations)} obs, "
    #             f"map has {total_cones} cones, "
    #             f"pose=({best.x:.2f}, {best.y:.2f}, {math.degrees(best.yaw):.1f}deg)"
    #         )

    #     # Use ground truth pose directly — no motion model integration needed
    #     self.particles = predict_particles(
    #         self.particles, self.u,
    #         self.current_x, self.current_y, self.current_yaw
    #     )

    #     self.particles = update_with_observations(self.particles, observations)
    #     self.particles = low_variance_resampling(self.particles)

    #     stamp = self.get_clock().now().to_msg()
    #     self._publish_pose(stamp)
    #     self._publish_map(stamp)

    def cone_callback(self, msg: MarkerArray):
        self._cone_count += 1
        total_markers = len(msg.markers)
        observations  = self._markers_to_obs(msg)

        if self._cone_count == 1:
            self.get_logger().info(
                f"First /fusion/cone_markers — "
                f"{total_markers} markers, {len(observations)} valid"
            )

        if not observations:
            if self._cone_count % 50 == 0:
                self.get_logger().warn(
                    f"cone_callback #{self._cone_count} — 0 valid obs, skipping"
                )
            return

        if self._cone_count % 50 == 0:
            best = max(self.particles, key=lambda p: p.w)
            total_cones = sum(len(v) for v in best.map.values())
            self.get_logger().info(
                f"SLAM update #{self._cone_count} — "
                f"{len(observations)} obs, "
                f"map has {total_cones} cones, "
                f"pose=({best.x:.2f}, {best.y:.2f}, {math.degrees(best.yaw):.1f}deg)"
            )

        # predict_particles now runs in odom_callback at 50Hz
        # so particles already have current pose when we get here
        self.particles = update_with_observations(self.particles, observations)
        self.particles = low_variance_resampling(self.particles)

        stamp = self.get_clock().now().to_msg()
        self._publish_pose(stamp)
        self._publish_map(stamp)


    def _markers_to_obs(self, msg: MarkerArray):
        obs = []
        skipped_sentinel = 0
        skipped_range    = 0

        for marker in msg.markers:
            if marker.action == 3:
                skipped_sentinel += 1
                continue

            x = marker.pose.position.x
            y = marker.pose.position.y
            r = math.hypot(x, y)

            if r < 0.3 or r > 20.0:
                skipped_range += 1
                continue

            b     = math.atan2(y, x)
            color = self._decode_color(marker)
            obs.append({'r': r, 'b': b, 'color': color})

        if not obs and self._cone_count % 20 == 0:
            self.get_logger().warn(
                f"_markers_to_obs — 0 valid obs. "
                f"Skipped {skipped_sentinel} sentinels, {skipped_range} out-of-range"
            )
        return obs

    def _decode_color(self, marker: Marker) -> str:
        r, g, b = marker.color.r, marker.color.g, marker.color.b
        if r > 0.9 and g > 0.9 and b < 0.2:
            return 'yellow'
        if b > 0.9 and r < 0.2:
            return 'blue'
        if r > 0.9 and 0.3 < g < 0.7:
            return 'orange'
        return 'yellow'

    def _publish_pose(self, stamp):
        total_w = sum(p.w for p in self.particles)
        if total_w < 1e-12:
            self.get_logger().warn("Particle weights near zero — filter may have collapsed")
            return

        x_est   = sum(p.w * p.x for p in self.particles)
        y_est   = sum(p.w * p.y for p in self.particles)
        sin_sum = sum(p.w * math.sin(p.yaw) for p in self.particles)
        cos_sum = sum(p.w * math.cos(p.yaw) for p in self.particles)
        yaw_est = math.atan2(sin_sum, cos_sum)

        msg = PoseStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = self.odom_frame_id
        msg.pose.position.x = x_est
        msg.pose.position.y = y_est
        msg.pose.orientation.z = math.sin(yaw_est / 2)
        msg.pose.orientation.w = math.cos(yaw_est / 2)
        self.pose_pub.publish(msg)

        pa = PoseArray()
        pa.header.stamp    = stamp
        pa.header.frame_id = self.odom_frame_id
        for p in self.particles:
            pose = Pose()
            pose.position.x = p.x
            pose.position.y = p.y
            pose.orientation.z = math.sin(p.yaw / 2)
            pose.orientation.w = math.cos(p.yaw / 2)
            pa.poses.append(pose)
        self.particles_pub.publish(pa)

    def _publish_map(self, stamp):
        best = max(self.particles, key=lambda p: p.w)
        ma   = MarkerArray()

        del_m = Marker()
        del_m.action = 3
        ma.markers.append(del_m)

        color_rgb = {
            'yellow': (1.0, 1.0, 0.0),
            'blue':   (0.0, 0.0, 1.0),
            'orange': (1.0, 0.5, 0.0),
        }

        marker_id = 0
        for color, positions in best.map.items():
            for pos in positions:
                m = Marker()
                m.header.frame_id = self.odom_frame_id
                m.header.stamp    = stamp
                m.ns     = f'slam_{color}'
                m.id     = marker_id
                m.type   = Marker.CYLINDER
                m.action = Marker.ADD
                m.pose.position.x = float(pos[0])
                m.pose.position.y = float(pos[1])
                m.pose.position.z = 0.15
                m.scale.x = m.scale.y = 0.3
                m.scale.z = 0.5
                rgb = color_rgb[color]
                m.color.r, m.color.g, m.color.b, m.color.a = rgb[0], rgb[1], rgb[2], 0.8
                ma.markers.append(m)
                marker_id += 1

        self.map_pub.publish(ma)

        if self._cone_count % 50 == 0:
            self.get_logger().info(f"Map published — {marker_id} cones")


def main(args=None):
    rclpy.init(args=args)
    node = FastSLAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
