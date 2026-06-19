import ctypes
import site
import sys

VENV_SITE_PACKAGES = (
    "/home/atlasracing/atlasracing-perception/venv/lib/python3.10/site-packages"
)
CUSPARSELT_SO = (
    f"{VENV_SITE_PACKAGES}/nvidia/cusparselt/lib/libcusparseLt.so.0"
)

# torch's dlopen-by-soname lookup for libcusparseLt can't find it since it only
# ships inside the venv's pip nvidia-cusparselt package, not on the system
# linker path. Preloading it here (RTLD_LOCAL; RTLD_GLOBAL collides with cv2's
# own bundled libs and crashes with "free(): invalid pointer") makes it
# resident before torch's import triggers its own dlopen.
ctypes.CDLL(CUSPARSELT_SO, mode=ctypes.RTLD_LOCAL)

# Must come before any other import in this process: torch was built against
# a newer numpy C-API than the system's numpy (kept old on purpose so
# cv_bridge's compiled bindings work). Prepending the venv dir and importing
# numpy here forces the *venv's* numpy (1.26.4, still 1.x so cv_bridge is
# still happy) to be the one cached in sys.modules - whoever imports numpy
# later (cv_bridge included) just reuses it. Without this, torch.from_numpy()
# hard-fails with "RuntimeError: Numpy is not available".
sys.path.insert(0, VENV_SITE_PACKAGES)
# addsitedir (not a plain path insert) also processes the venv's .pth files,
# which is what registers torchvision's legacy .egg install on sys.path -
# without it "import torchvision" works but its package metadata doesn't,
# and ultralytics' version check fails with PackageNotFoundError.
site.addsitedir(VENV_SITE_PACKAGES)
import numpy as np  # noqa: E402

import cv2  # noqa: E402
import rclpy  # noqa: E402
from cv_bridge import CvBridge  # noqa: E402
from message_filters import ApproximateTimeSynchronizer, Subscriber  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from std_msgs.msg import ColorRGBA  # noqa: E402
from tf2_ros import Buffer, TransformListener  # noqa: E402
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException  # noqa: E402
from vision_msgs.msg import (  # noqa: E402
    BoundingBox3D,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)
from visualization_msgs.msg import Marker, MarkerArray  # noqa: E402
from ultralytics import YOLO  # noqa: E402

CLASS_COLORS = {
    0: ColorRGBA(r=0.6, g=0.6, b=0.6, a=1.0),  # unknown_cone
    1: ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),  # yellow_cone
    2: ColorRGBA(r=0.0, g=0.3, b=1.0, a=1.0),  # blue_cone
    3: ColorRGBA(r=1.0, g=0.45, b=0.0, a=1.0),  # orange_cone
    4: ColorRGBA(r=1.0, g=0.45, b=0.0, a=1.0),  # large_orange_cone
}
CLASS_CONE_SIZE = {
    0: (0.15, 0.15, 0.3),
    1: (0.15, 0.15, 0.3),
    2: (0.15, 0.15, 0.3),
    3: (0.15, 0.15, 0.3),
    4: (0.22, 0.22, 0.5),
}


def quat_translation_to_matrix(q, t):
    x, y, z, w = q.x, q.y, q.z, q.w
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = [t.x, t.y, t.z]
    return m


class ConeLocator(Node):
    def __init__(self):
        super().__init__("cone_locator")

        self.declare_parameter("rgb_topic", "/zed/zed_node/rgb/color/rect/image")
        self.declare_parameter("depth_topic", "/zed/zed_node/depth/depth_registered")
        self.declare_parameter(
            "camera_info_topic", "/zed/zed_node/rgb/color/rect/camera_info"
        )
        self.declare_parameter(
            "engine_path",
            "/home/atlasracing/atlasracing-perception/yolo_models/"
            "YOLOv26_1280x1280/yolo26m_1280/weights/best.engine",
        )
        self.declare_parameter("conf_threshold", 0.4)
        self.declare_parameter("publish_debug_image", True)
        # Compensates for depth being registered to the left camera while RGB
        # comes from the right camera. Shift all cone y-positions right (negative)
        # by the stereo baseline (~0.12 m for ZED X) until blue cones read correctly.
        self.declare_parameter("lateral_correction", 0.0)

        rgb_topic = self.get_parameter("rgb_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        engine_path = self.get_parameter("engine_path").value
        self.conf_threshold = self.get_parameter("conf_threshold").value
        self.publish_debug_image = self.get_parameter("publish_debug_image").value
        self.lateral_correction = self.get_parameter("lateral_correction").value

        self.get_logger().info(f"Loading YOLO engine: {engine_path}")
        self.model = YOLO(engine_path, task="detect")
        self.class_names = self.model.names
        self.get_logger().info(f"Loaded classes: {self.class_names}")

        self.bridge = CvBridge()
        self.intrinsics = None  # (fx, fy, cx, cy)

        # Cone markers are published directly in base_link (not the camera's
        # optical frame) so their ground-relative height can be set
        # explicitly instead of trusting a single noisy depth sample. This
        # is the static optical-frame -> base_link transform, looked up
        # once and cached, used to move each detection's position there.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cam_to_base = None

        self.detections_pub = self.create_publisher(
            Detection3DArray, "/hydrakon_camera/cone_detections", 10
        )
        self.markers_pub = self.create_publisher(
            MarkerArray, "/hydrakon_camera/cone_markers", 10
        )
        if self.publish_debug_image:
            self.debug_image_pub = self.create_publisher(
                Image, "/hydrakon_camera/debug_image", 5
            )

        self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_cb, qos_profile_sensor_data
        )

        self.rgb_sub = Subscriber(
            self, Image, rgb_topic, qos_profile=qos_profile_sensor_data
        )
        self.depth_sub = Subscriber(
            self, Image, depth_topic, qos_profile=qos_profile_sensor_data
        )
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=5, slop=0.05
        )
        self.sync.registerCallback(self.image_cb)

    def camera_info_cb(self, msg: CameraInfo):
        k = msg.k
        self.intrinsics = (k[0], k[4], k[2], k[5])

    def _get_cam_to_base(self, optical_frame):
        if self.cam_to_base is None:
            try:
                tf = self.tf_buffer.lookup_transform(
                    "base_link", optical_frame, rclpy.time.Time()
                )
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(
                    f"Waiting for static base_link -> {optical_frame}: {e}",
                    throttle_duration_sec=2.0,
                )
                return None
            self.cam_to_base = quat_translation_to_matrix(
                tf.transform.rotation, tf.transform.translation
            )
        return self.cam_to_base

    def image_cb(self, rgb_msg: Image, depth_msg: Image):
        if self.intrinsics is None:
            return

        fx, fy, cx_i, cy_i = self.intrinsics
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        results = self.model.predict(rgb, conf=self.conf_threshold, verbose=False)[0]

        detection_array = Detection3DArray()
        detection_array.header = rgb_msg.header
        marker_array = MarkerArray()
        debug_img = rgb.copy() if self.publish_debug_image else None
        cam_to_base = self._get_cam_to_base(rgb_msg.header.frame_id)

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.class_names.get(cls_id, str(cls_id))
            px, py = int((x1 + x2) / 2), int((y1 + y2) / 2)

            z = self._sample_depth(depth, px, py)
            if z is None:
                if debug_img is not None:
                    self._draw_box(debug_img, x1, y1, x2, y2, cls_name, conf)
                continue

            x = (px - cx_i) * z / fx
            y = (py - cy_i) * z / fy

            detection = Detection3D()
            detection.header = rgb_msg.header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = cls_name
            hyp.hypothesis.score = conf
            hyp.pose.pose.position.x = x
            hyp.pose.pose.position.y = y
            hyp.pose.pose.position.z = z
            detection.results.append(hyp)

            size = CLASS_CONE_SIZE.get(cls_id, (0.15, 0.15, 0.3))
            bbox = BoundingBox3D()
            bbox.center.position.x = x
            bbox.center.position.y = y
            bbox.center.position.z = z
            bbox.size.x, bbox.size.y, bbox.size.z = size
            detection.bbox = bbox
            detection.id = cls_name
            detection_array.detections.append(detection)

            # Real-world bbox footprint, back-projected the same way as the
            # detection's centre point, so the marker reflects the actual
            # detected box rather than a fixed nominal cone size.
            width = abs(x2 - x1) * z / fx
            height = abs(y2 - y1) * z / fy
            marker = self._make_marker(
                rgb_msg.header, i, cls_id, x, y, z, width, height, cam_to_base
            )
            if marker is not None:
                marker_array.markers.append(marker)

            if debug_img is not None:
                self._draw_box(debug_img, x1, y1, x2, y2, cls_name, conf, z)

        self.detections_pub.publish(detection_array)
        self.markers_pub.publish(marker_array)
        if debug_img is not None:
            out_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            out_msg.header = rgb_msg.header
            self.debug_image_pub.publish(out_msg)

    @staticmethod
    def _sample_depth(depth, px, py, patch=2):
        h, w = depth.shape[:2]
        x0, x1 = max(0, px - patch), min(w, px + patch + 1)
        y0, y1 = max(0, py - patch), min(h, py + patch + 1)
        patch_vals = depth[y0:y1, x0:x1].astype(np.float32).flatten()
        valid = patch_vals[np.isfinite(patch_vals) & (patch_vals > 0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _make_marker(self, header, idx, cls_id, x, y, z, width, height, cam_to_base):
        if cam_to_base is None:
            return None
        p_base = cam_to_base @ np.array([x, y, z, 1.0])

        marker = Marker()
        marker.header.stamp = header.stamp
        marker.header.frame_id = "base_link"
        marker.ns = "cones"
        marker.id = idx
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = float(p_base[0])
        marker.pose.position.y = float(p_base[1]) + self.lateral_correction
        # base_link's Z=0 is ground level, so resting the cylinder's base on
        # the ground (rather than trusting the noisy single-pixel depth
        # sample for height) just means centring it at half its own height.
        marker.pose.position.z = height / 2.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = width
        marker.scale.y = width
        marker.scale.z = height
        marker.color = CLASS_COLORS.get(cls_id, CLASS_COLORS[0])
        marker.lifetime.nanosec = 300_000_000
        return marker

    @staticmethod
    def _draw_box(img, x1, y1, x2, y2, label, conf, z=None):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        text = f"{label} {conf:.2f}" + (f" {z:.1f}m" if z is not None else "")
        cv2.putText(
            img, text, (p1[0], max(0, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )


def main(args=None):
    rclpy.init(args=args)
    node = ConeLocator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
