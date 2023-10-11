cmake_minimum_required(VERSION 3.0.2)
project(detection_client)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
)

find_package(OpenCV REQUIRED)  # 这里包括 OpenCV

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES yolo
#  CATKIN_DEPENDS roscpp
#  DEPENDS system_lib
)

include_directories(
  include
  /usr/include/tritonclient
  ${catkin_INCLUDE_DIRS}
)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${tritonclientgrpc_INCLUDE_DIRS})
add_executable(detection_client src/yolo_client.cpp)
target_link_libraries(detection_client 
  ${catkin_LIBRARIES}
)

target_link_directories(detection_client PRIVATE /usr/lib/tritonclient)
target_link_libraries(detection_client ${OpenCV_LIBS})
target_link_libraries(
  detection_client  
  grpcclient
  httpclient
)
