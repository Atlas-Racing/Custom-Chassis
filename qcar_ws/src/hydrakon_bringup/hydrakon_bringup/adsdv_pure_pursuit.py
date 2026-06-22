#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import String, Bool
import math
import numpy as np
import re
from collections import defaultdict
import time

class CombinedController(Node):
    def should_run_controller(self):
        """Check if controller should run based on AMI state"""
        if self.current_ami_state is None:
            return False

        # Only run when AMI state is SKIDPAD
        return self.current_ami_state == 'SKIDPAD'

    def __init__(self):
        super().__init__('combined_controller')

        # Subscribe to camera detections
        self.detection_subscription = self.create_subscription(
            Detection2DArray,
            '/zed2/detections_data',
            self.detection_callback,
            10
        )

        # Subscribe to CAN state
        self.state_subscription = self.create_subscription(
            String,
            '/hydrakon_can/state_str',
            self.state_callback,
            10
        )

        # Publishers
        self.command_publisher = self.create_publisher(
            AckermannDriveStamped,
            '/hydrakon_can/command',
            10
        )



        # Camera parameters
        self.image_width = 1280
        self.image_height = 720
        self.image_center_x = self.image_width // 2

        # Steering parameters
        self.max_steering_angle = 0.4
        self.steering_gain = 0.003
        self.min_cone_distance = 50
        self.estimated_track_width_pixels = 240  # Standard Formula Student track width (~3.5m) in pixels

        # Cone class IDs
        self.YELLOW_CONE = 0
        self.BLUE_CONE = 1
        self.ORANGE_CONE = 2
        self.LARGE_ORANGE_CONE = 3

        # Default drive parameters
        self.default_speed = 0.0
        self.default_acceleration = 0.0
        self.emergency_brake_value = 60.0  # Brake value for emergency stop
        self.cone_pair_acceleration = 0.9  # Acceleration when both cone types are detected

        # State tracking for AMI control
        self.current_as_state = None
        self.current_ami_state = None

        # Orange cone ignore timer
        self.start_time = time.time()
        self.orange_ignore_duration = 5.0  # Ignore orange cones for first 5 seconds

        # Last steering angle for continuity
        self.last_steering_angle = 0.0

        # Midpoint steering parameters
        self.midpoint_smoothing = 0.7  # Smoothing factor for midpoint calculation
        self.last_midpoint_x = None

        self.get_logger().info('Combined Controller initialized')
        self.get_logger().info('- Camera-based steering control')
        self.get_logger().info('- Runs only when AMI state is SKIDPAD')
        self.get_logger().info('- Enhanced midpoint steering between yellow and blue cones')
        self.get_logger().info('- Improved sharp turn steering for single-color detection')
        self.get_logger().info('- Acceleration when cone pairs detected')
        self.get_logger().info('- Position validation: Yellow left, Blue right')
        self.get_logger().info(f'- Orange cones ignored for first {self.orange_ignore_duration} seconds')
        self.get_logger().info(f'Image dimensions: {self.image_width}x{self.image_height}')
        self.get_logger().info(f'Steering gain: {self.steering_gain}, Max angle: {self.max_steering_angle}')
        self.get_logger().info(f'Cone pair acceleration: {self.cone_pair_acceleration}')
        self.get_logger().info(f'Midpoint smoothing factor: {self.midpoint_smoothing}')

    def parse_state_string(self, state_str):
        """Parse the state string to extract AS and AMI values"""
        try:
            as_match = re.search(r'AS:(\w+)', state_str)
            as_state = as_match.group(1) if as_match else None

            ami_match = re.search(r'AMI:(\w+)', state_str)
            ami_state = ami_match.group(1) if ami_match else None

            return as_state, ami_state
        except Exception as e:
            self.get_logger().error(f'Error parsing state string: {e}')
            return None, None

    def state_callback(self, msg):
        """Callback for CAN state updates"""
        as_state, ami_state = self.parse_state_string(msg.data)

        if as_state is not None and ami_state is not None:
            self.current_as_state = as_state
            self.current_ami_state = ami_state
            self.get_logger().debug(f'Updated state - AS: {as_state}, AMI: {ami_state}')

    def should_ignore_orange_cones(self):
        """Check if orange cones should be ignored (first 5 seconds)"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if elapsed_time < self.orange_ignore_duration:
            return True
        else:
            return False

    def detection_callback(self, msg):
        """Process camera detections and calculate steering angle"""
        yellow_cones = []
        blue_cones = []
        orange_cones = []

        # Extract cone positions from detections
        for detection in msg.detections:
            if len(detection.results) > 0:
                class_id = int(detection.results[0].hypothesis.class_id)
                confidence = detection.results[0].hypothesis.score

                # Get cone center position
                center_x = detection.bbox.center.position.x
                center_y = detection.bbox.center.position.y

                # Filter by confidence and classify by color
                if confidence > 0.5:
                    if class_id == self.YELLOW_CONE:
                        yellow_cones.append((center_x, center_y, confidence))
                    elif class_id == self.BLUE_CONE:
                        blue_cones.append((center_x, center_y, confidence))
                    elif class_id == self.ORANGE_CONE or class_id == self.LARGE_ORANGE_CONE:
                        orange_cones.append((center_x, center_y, confidence))

        # Calculate steering angle and acceleration
        steering_angle = self.calculate_steering_angle(yellow_cones, blue_cones)
        acceleration = self.calculate_acceleration(yellow_cones, blue_cones)

        # Only publish steering commands if AMI state is SKIDPAD
        if self.should_run_controller():
            self.publish_steering_command(steering_angle, acceleration, msg.header.stamp)

            # Log detection info
            if len(yellow_cones) > 0 or len(blue_cones) > 0:
                accel_status = "ACCELERATING" if acceleration > 0 else "COASTING"
                self.get_logger().info(f'SKIDPAD ACTIVE: {len(yellow_cones)} yellow, {len(blue_cones)} blue cones. Steering: {steering_angle:.3f} rad, {accel_status}: {acceleration:.1f}')

            # Log orange cone detection with ignore status
            if len(orange_cones) > 0:
                if self.should_ignore_orange_cones():
                    current_time = time.time()
                    elapsed_time = current_time - self.start_time
                    remaining_time = self.orange_ignore_duration - elapsed_time
                    self.get_logger().debug(f'Orange cones detected: {len(orange_cones)} (IGNORED - {remaining_time:.1f}s remaining)')
                else:
                    self.get_logger().debug(f'Orange cones detected: {len(orange_cones)} (timer expired, would be processed)')
        else:
            # Send zero steering and acceleration when not in SKIDPAD mode
            self.publish_steering_command(0.0, 0.0, msg.header.stamp)
            if len(yellow_cones) > 0 or len(blue_cones) > 0:
                self.get_logger().debug(f'NOT SKIDPAD: Cones detected but AMI state is {self.current_ami_state}, not running controller')

    def calculate_acceleration(self, yellow_cones, blue_cones):
        """Calculate acceleration based on cone detection with enhanced position validation"""
        # Accelerate when any cones are detected (both colors, single color, or mixed)
        if len(yellow_cones) > 0 and len(blue_cones) > 0:
            # Both colors detected - best case scenario
            closest_yellow = self.find_closest_cone(yellow_cones)
            closest_blue = self.find_closest_cone(blue_cones)

            if closest_yellow is None or closest_blue is None:
                return self.default_acceleration

            yellow_x = closest_yellow[0]
            blue_x = closest_blue[0]

            # More lenient position validation - allow for some overlap
            # Yellow should generally be on the left, blue on the right
            separation_threshold = 50  # Minimum separation in pixels

            if abs(yellow_x - blue_x) > separation_threshold:
                # Check if yellow is generally on the left side of blue
                if yellow_x < blue_x:
                    self.get_logger().debug(f'Valid cone pair: Yellow at {yellow_x:.1f}, Blue at {blue_x:.1f}, separation: {abs(yellow_x-blue_x):.1f} - ACCELERATING')
                    return self.cone_pair_acceleration
                else:
                    self.get_logger().debug(f'Reversed cone positioning: Yellow at {yellow_x:.1f}, Blue at {blue_x:.1f} - Slow acceleration')
                    return self.cone_pair_acceleration * 0.5  # Reduced acceleration for unclear positioning
            else:
                self.get_logger().debug(f'Cones too close: Yellow at {yellow_x:.1f}, Blue at {blue_x:.1f}, separation: {abs(yellow_x-blue_x):.1f} - Normal acceleration')
                return self.cone_pair_acceleration * 0.7  # Moderate acceleration when cones are close

        elif len(yellow_cones) > 0:
            # Only yellow cones detected - accelerate for left turn
            self.get_logger().debug(f'Yellow-only detected: {len(yellow_cones)} cones - ACCELERATING for left turn')
            return self.cone_pair_acceleration * 0.8  # Slightly reduced acceleration for single color

        elif len(blue_cones) > 0:
            # Only blue cones detected - accelerate for right turn
            self.get_logger().debug(f'Blue-only detected: {len(blue_cones)} cones - ACCELERATING for right turn')
            return self.cone_pair_acceleration * 0.8  # Slightly reduced acceleration for single color

        else:
            # No cones detected - coast
            return self.default_acceleration

    def calculate_steering_angle(self, yellow_cones, blue_cones):
        """Calculate steering angle with enhanced midpoint targeting and sharp turn logic"""

        # If no cones detected, return last steering angle for continuity
        if len(yellow_cones) == 0 and len(blue_cones) == 0:
            return self.last_steering_angle * 0.9  # Gradually reduce steering

        target_x = None
        steering_mode = "unknown"

        # Case 1: Both yellow and blue cones detected - MIDPOINT STEERING
        if len(yellow_cones) > 0 and len(blue_cones) > 0:
            # Find the best pair of cones for midpoint calculation
            closest_yellow = self.find_closest_cone(yellow_cones)
            closest_blue = self.find_closest_cone(blue_cones)

            if closest_yellow is not None and closest_blue is not None:
                yellow_x = closest_yellow[0]
                blue_x = closest_blue[0]

                # Calculate raw midpoint
                raw_midpoint = (yellow_x + blue_x) / 2.0

                # Apply smoothing if we have a previous midpoint
                if self.last_midpoint_x is not None:
                    target_x = (self.midpoint_smoothing * self.last_midpoint_x +
                               (1 - self.midpoint_smoothing) * raw_midpoint)
                else:
                    target_x = raw_midpoint

                # Store for next iteration
                self.last_midpoint_x = target_x

                steering_mode = "midpoint"

                # Enhanced logging for midpoint steering
                self.get_logger().debug(f'Midpoint steering: Yellow at {yellow_x:.1f}, Blue at {blue_x:.1f}')
                self.get_logger().debug(f'Raw midpoint: {raw_midpoint:.1f}, Smoothed target: {target_x:.1f}')

        # Case 2: Only yellow cones detected - LEFT TURN, follow curvature
        elif len(yellow_cones) > 0:
            closest_yellow = self.find_closest_cone(yellow_cones)
            if closest_yellow is not None:
                yellow_x = closest_yellow[0]

                # Conservative offset to avoid hitting cones - aim slightly inside track center
                conservative_offset = 180  # Increased from 120 to give more clearance from yellow cones
                target_x = yellow_x + conservative_offset

                steering_mode = "yellow_curvature"
                self.get_logger().debug(f'Yellow curvature steering: Yellow at {yellow_x:.1f}, Target (safe): {target_x:.1f}')

        # Case 3: Only blue cones detected - RIGHT TURN, follow curvature
        elif len(blue_cones) > 0:
            closest_blue = self.find_closest_cone(blue_cones)
            if closest_blue is not None:
                blue_x = closest_blue[0]

                # Conservative offset to avoid hitting cones - aim slightly inside track center
                conservative_offset = 180  # Increased from 120 to give more clearance from blue cones
                target_x = blue_x - conservative_offset

                steering_mode = "blue_curvature"
                self.get_logger().debug(f'Blue curvature steering: Blue at {blue_x:.1f}, Target (safe): {target_x:.1f}')

        # Convert target position to steering angle
        if target_x is not None:
            error_pixels = target_x - self.image_center_x
            steering_angle = -error_pixels * self.steering_gain

            # Apply steering angle limits
            steering_angle = max(-self.max_steering_angle,
                               min(self.max_steering_angle, steering_angle))

            # Store for continuity
            self.last_steering_angle = steering_angle

            # Log steering decision
            self.get_logger().debug(f'Steering calculation: Mode={steering_mode}, Target={target_x:.1f}, Error={error_pixels:.1f}, Angle={steering_angle:.3f}')

            return steering_angle

        # Fallback: gradually reduce last steering angle
        fallback_angle = self.last_steering_angle * 0.9
        self.get_logger().debug(f'Fallback steering: {fallback_angle:.3f}')
        return fallback_angle

    def find_closest_cone(self, cones):
        """Find the closest cone based on y-coordinate (bottom of image is closer)"""
        if not cones:
            return None

        # Closest cone is the one with the highest y-coordinate (bottom of image)
        closest_cone = max(cones, key=lambda cone: cone[1])
        return closest_cone

    def publish_steering_command(self, steering_angle, acceleration, timestamp):
        """Publish Ackermann steering command with acceleration"""
        msg = AckermannDriveStamped()

        msg.header.stamp = timestamp
        msg.header.frame_id = 'base_link'

        msg.drive.steering_angle = float(steering_angle)
        msg.drive.steering_angle_velocity = 0.0
        msg.drive.speed = self.default_speed
        msg.drive.acceleration = float(acceleration)
        msg.drive.jerk = 0.0

        self.command_publisher.publish(msg)

def test_mode():
    """Test mode - sends test messages for both driving flag and steering with acceleration"""
    rclpy.init()

    node = Node('combined_controller_test')

    # Publishers for testing
    command_publisher = node.create_publisher(
        AckermannDriveStamped,
        '/hydrakon_can/command',
        10
    )

    node.get_logger().info('TEST MODE: Combined Controller Test with Enhanced Steering and Orange Ignore Timer')
    node.get_logger().info('Testing midpoint steering, sharp turn curvature following, and acceleration...')
    node.get_logger().info('Orange cones ignored for first 5 seconds of operation')
    node.get_logger().info('NOTE: In normal operation, controller only runs when AMI state is SKIDPAD')
    node.get_logger().info('Press Ctrl+C to stop')

    import time
    test_sequence = [
        {'steering': 0.0, 'acceleration': 0.0, 'description': 'Straight ahead'},
        {'steering': 0.1, 'acceleration': 0.9, 'description': 'Right turn with acceleration'},
        {'steering': 0.2, 'acceleration': 0.9, 'description': 'Sharp right with acceleration'},
        {'steering': 0.0, 'acceleration': 0.9, 'description': 'Straight with acceleration'},
        {'steering': -0.1, 'acceleration': 0.9, 'description': 'Left turn with acceleration'},
        {'steering': -0.2, 'acceleration': 0.0, 'description': 'Sharp left, single cone'},
        {'steering': 0.0, 'acceleration': 0.0, 'description': 'Coast to stop'},
    ]

    sequence_index = 0

    try:
        while rclpy.ok():
            test_data = test_sequence[sequence_index % len(test_sequence)]

            # Publish steering command with acceleration
            cmd_msg = AckermannDriveStamped()
            cmd_msg.header.stamp = node.get_clock().now().to_msg()
            cmd_msg.header.frame_id = 'base_link'
            cmd_msg.drive.steering_angle = float(test_data['steering'])
            cmd_msg.drive.steering_angle_velocity = 0.0
            cmd_msg.drive.speed = 0.0
            cmd_msg.drive.acceleration = float(test_data['acceleration'])
            cmd_msg.drive.jerk = 0.0

            status = "ACCELERATING" if test_data['acceleration'] > 0 else "COASTING"

            command_publisher.publish(cmd_msg)

            node.get_logger().info(f'Published - {test_data["description"]}: Steering: {test_data["steering"]:.1f} rad, {status}')

            sequence_index += 1
            time.sleep(2.0)
            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        node.get_logger().info('Test mode stopped')

    node.destroy_node()
    rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    combined_controller = CombinedController()

    try:
        rclpy.spin(combined_controller)
    except KeyboardInterrupt:
        pass

    combined_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1 or '--test' in sys.argv:
        print("Running in TEST MODE - testing steering, acceleration, and sharp turn logic with orange ignore timer")
        print("Use 'ros2 run <package> <node>' for normal operation")
        test_mode()
    else:
        main()
