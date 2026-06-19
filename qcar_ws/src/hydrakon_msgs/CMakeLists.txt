cmake_minimum_required(VERSION 3.5)
project(hydrakon_msgs)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/ConeObservation.msg
  msg/ConeObservationArray.msg
  DEPENDENCIES std_msgs
)

ament_package()
