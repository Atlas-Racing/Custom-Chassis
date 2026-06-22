#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import Twist, PointStamped
import math
import numpy as np
from .pure_pursuit import PurePursuitNode

class AccelerationNode(PurePursuitNode):
    def __init__(self):
        super().__init__()
        self.get_logger().info("Acceleration Track Node Started")

        self.target_speed = 1.0 #adjust speed
        self.track_width = 3.0
        self.lookahead_base = 6.0    
        self.lookahead_gain = 0.45   
        
        # start variables
        self.start_x = None 
        self.start_y = None
        self.dist_traveled = 0.0
        self.mission_complete = False
        
        self.min_dist_to_finish = 60.0 #min distance before looking for finish
        self.stop_distance_threshold = 8.0  #detects cones within 8m for finish line

        #stats variables
        self.max_speed_recorded = 0.0
        self.run_start_time = None
        self.run_finish_time = None
        self.racing_active = False

        self.finish_detected = False
        self.stop_distance_target = 0.0 # Distance to stop after finish line for full speed run
        
        # variables for accurate timing
        self.physical_finish_line_dist = 0.0
        self.stats_locked = False
        self.race_finish_dist = 0.0
        self.race_finish_speed = 0.0


    def odom_callback(self, msg):
        super().odom_callback(msg)
        
        # If mission is complete, ensure car is stopped
        if self.mission_complete:
            self.stop_car()
            return

        #stats related stuff
        if self.current_speed > self.max_speed_recorded:
            self.max_speed_recorded = self.current_speed

        
        if not self.racing_active and self.current_speed > 0.1:
            self.racing_active = True
            self.run_start_time = self.get_clock().now().nanoseconds / 1e9

        # Distance tracking
        pose = msg.pose.pose
        if self.start_x is None:
            self.start_x = pose.position.x
            self.start_y = pose.position.y
        else:
            self.dist_traveled = math.hypot(pose.position.x - self.start_x, pose.position.y - self.start_y)

        #if the finish line has been detected
        if self.finish_detected:
            #Check if we just crossed the physical line. to know the finish line is after 8 m
            if not self.stats_locked and self.dist_traveled >= self.physical_finish_line_dist:
                self.run_finish_time = self.get_clock().now().nanoseconds / 1e9
                self.race_finish_dist = self.dist_traveled
                self.race_finish_speed = self.max_speed_recorded
                self.stats_locked = True
                self.get_logger().warn("CROSSED FINISH LINE")

            #stop 15m after finish line for full speed run
            if self.dist_traveled >= self.stop_distance_target:
                self.get_logger().warn("STOPPING NOW")
                self.mission_complete = True
                self.print_race_report()
                self.stop_car()
            else:
                self.publish_override(self.target_speed, 0.0)   

    # override marker callback for extra logic
    def marker_callback(self, msg):
        if self.mission_complete:
            self.stop_car()
            return
        
        if self.finish_detected:
            return 
            
        blues = []
        yellows = []
        oranges = []
        
        for m in msg.markers:
            if m.pose.position.x < 0.0 or m.pose.position.x > 30.0: continue #only care about cones in front within 30m
            
            pos = np.array([m.pose.position.x, m.pose.position.y])

            if m.color.b > 0.9 and m.color.r < 0.5: blues.append(pos)
            elif m.color.r > 0.9 and m.color.g > 0.9: yellows.append(pos)
            elif m.color.r > 0.9 and m.color.g < 0.6: oranges.append(pos)

        # start looking for finish line cones after min distance
        if self.dist_traveled > self.min_dist_to_finish:
            finish_cones = [p for p in oranges if p[0] < self.stop_distance_threshold]
            if len(finish_cones) >= 1:
                self.get_logger().warn("FINISH SIGHTED! Driving 8m to line...")
                self.finish_detected = True
                
                # We are at X, cones are at X+8. So line is at X+8.
                self.physical_finish_line_dist = self.dist_traveled + 8.0
                # Buffer distance to stop after finish line for full speed run
                self.stop_distance_target = self.physical_finish_line_dist + 25.0
                return

        # dynamic lookahead calculation on speed
        Ld = self.lookahead_base + (self.lookahead_gain * self.current_speed)
        target = None

        # looking for oranges cones at start
        if self.dist_traveled < 10.0:
            close_oranges = [p for p in oranges if p[0] < 12.0]
            if len(close_oranges) >= 2:
                avg_x = sum(p[0] for p in close_oranges) / len(close_oranges)
                avg_y = sum(p[1] for p in close_oranges) / len(close_oranges)
                target = np.array([avg_x, avg_y])

        # if no target from oranges, use blues and yellows
        if target is None:
            def get_closest(cones):
                if not cones: return None
                return min(cones, key=lambda p: abs(p[0] - Ld))

            best_blue = get_closest(blues)
            best_yellow = get_closest(yellows)

            if best_blue is not None and best_yellow is not None:
                target = (best_blue + best_yellow) / 2.0
            elif best_blue is not None:
                target = best_blue + np.array([0.0, -self.track_width / 2.0])
            elif best_yellow is not None:
                target = best_yellow + np.array([0.0, self.track_width / 2.0])

        # Drive to target if found
        if target is not None:
            self.publish_point(self.target_pub, target)
            self.drive_to_target(target)
            
        else:
            if self.current_speed > 0.5:
                self.publish_override(0.0, 0.0)

    # printing stats
    def print_race_report(self):
        duration = 0.0
        if self.run_start_time and self.run_finish_time:
            duration = self.run_finish_time - self.run_start_time
        
        max_kmh = self.race_finish_speed * 3.6
        
        print("\n" + "="*40)
        print("         RACE FINISHED           ")
        print("="*40)
        print(f"  Total Time:   {duration:.3f} seconds")
        print(f"  Top Speed:    {self.race_finish_speed:.2f} m/s ({max_kmh:.1f} km/h)")
        print(f"  Distance:     {self.race_finish_dist:.2f} meters")
        print("="*40 + "\n")

    # manual override publisher
    def publish_override(self, linear, angular):
        t = Twist()
        t.linear.x = float(linear)
        t.angular.z = float(angular)
        self.cmd_vel_pub.publish(t)
    
    # stop the car manually
    def stop_car(self):
        self.publish_override(0.0, 0.0)

def main(args=None):
    rclpy.init(args=args)
    node = AccelerationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
