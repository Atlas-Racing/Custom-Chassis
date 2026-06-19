# Array of fused cone observations
# Published at each fusion cycle with all currently-visible cones

std_msgs/Header header    # Timestamp + frame_id (should be "odom")
ConeObservation[] cones   # All currently observed cones
