# A single fused cone observation in the odom frame
# Published by cone_fusion_node, consumed by ekf_slam_node

float64 x               # X position in odom frame (meters)
float64 y               # Y position in odom frame (meters)

# Unique track ID from the fusion node
uint32 track_id

# Color classification
# 0 = unknown, 1 = yellow, 2 = blue, 3 = orange
uint8 color_id

# Bayesian color probabilities [unknown, yellow, blue, orange]
# Allows the SLAM node to do its own gating with full probability info
# rather than just the argmax
float32[4] color_probs

# Observation confidence — how many times this cone has been seen
# by the fusion tracker (higher = more reliable)
uint32 observation_count

# 2x2 covariance from the fusion Kalman filter [xx, xy, yx, yy]
# Row-major. Gives the SLAM node per-observation uncertainty
float64[4] covariance
