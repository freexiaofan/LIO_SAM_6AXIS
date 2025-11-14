# cmake_minimum_required(VERSION 2.8.3)
cmake_minimum_required(VERSION 3.22)
project(lio_sam_6axis)


add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
add_link_options(-fsanitize=address)

# --- Prefer config packages over Find*.cmake ---
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")
find_package(Boost REQUIRED COMPONENTS timer)
find_package(catkin REQUIRED COMPONENTS
        tf
        roscpp
        rospy
        cv_bridge
        pcl_conversions
        std_msgs
        sensor_msgs
        geometry_msgs
        nav_msgs
        message_generation
        visualization_msgs
        rosbag
        chcnav
        )

# Remove any PCL 1.12 include leaks from catkin
if (catkin_INCLUDE_DIRS)
  list(FILTER catkin_INCLUDE_DIRS EXCLUDE REGEX ".*/pcl-1\\.12.*")
  message(STATUS "catkin include dirs: ${catkin_INCLUDE_DIRS}")
endif()

# --- Force FLANN (if present) ---
# find_package(FLANN REQUIRED)

# --- Force using custom PCL 1.15 config ---
# Point this to your PCL 1.15 installed or build config directory
set(PCL_DIR "/home/tyjt/Documents/pcl-pcl-1.15.1/build" CACHE PATH "Path to PCL 1.15 PCLConfig.cmake" FORCE)
# Make sure CMake searches this first
set(CMAKE_PREFIX_PATH "${PCL_DIR};${CMAKE_PREFIX_PATH}")

# Harden: if the provided PCL_DIR has a config, use it exclusively
if (EXISTS "${PCL_DIR}/PCLConfig.cmake")
  message(STATUS "Using PCL config at: ${PCL_DIR}")
  find_package(PCL 1.15 REQUIRED CONFIG PATHS "${PCL_DIR}" NO_DEFAULT_PATH)
else()
  message(WARNING "PCLConfig.cmake not found under PCL_DIR=${PCL_DIR}. Falling back to default search. Consider installing PCL 1.15 and setting PCL_DIR to its config path.")
  find_package(PCL 1.15 REQUIRED)
endif()

# After locating PCL, filter out any system PCL 1.12 include dirs defensively
if (PCL_INCLUDE_DIRS)
  set(_PCL_INCLUDE_DIRS_ORIG ${PCL_INCLUDE_DIRS})
  list(FILTER PCL_INCLUDE_DIRS EXCLUDE REGEX ".*/pcl-1\\.12.*")
  if (NOT _PCL_INCLUDE_DIRS_ORIG STREQUAL PCL_INCLUDE_DIRS)
    message(STATUS "Filtered system PCL includes. Using: ${PCL_INCLUDE_DIRS}")
  else()
    message(STATUS "PCL include dirs: ${PCL_INCLUDE_DIRS}")
  endif()
endif()
# Hard fail if any 1.12 include sneaks in via catkin or transitive deps
foreach(_inc IN LISTS PCL_INCLUDE_DIRS)
  if (_inc MATCHES "/pcl-1\\.12")
    message(FATAL_ERROR "Detected system PCL 1.12 include path: ${_inc}. Please remove it from your environment and ensure PCL 1.15 only.")
  endif()
endforeach()

# Require version info
if (DEFINED PCL_VERSION)
  message(STATUS "Resolved PCL version: ${PCL_VERSION}")
  if (NOT PCL_VERSION VERSION_GREATER_EQUAL "1.15")
    message(FATAL_ERROR "PCL version (${PCL_VERSION}) is older than 1.15. Fix your PCL_DIR/CMAKE_PREFIX_PATH.")
  endif()
endif()

#   GTSAM 4.1.1  on unbuntu 22.04
find_package(GTSAM REQUIRED)
if(GTSAM_FOUND)
    message(STATUS "Found GTSAM ${GTSAM_VERSION}")
else()
    message(STATUS "GTSAM Not Found")
endif()
message(STATUS "GTSAM_DIR=${GTSAM_DIR}")
find_package(libxml2 QUIET)
if (NOT LIBXML2_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(Y REQUIRED libxml-2.0)
endif ()

add_message_files(
        DIRECTORY msg
        FILES
        cloud_info.msg
)

add_service_files(
        DIRECTORY srv
        FILES
        save_map.srv
)

generate_messages(
        DEPENDENCIES
        geometry_msgs
        std_msgs
        nav_msgs
        sensor_msgs
)

add_subdirectory(./ThirdParty/GeographicLib/)
# add_subdirectory(./ThirdParty/nano_gicp)
add_library(nano_gicp_lib
  ThirdParty/nano_gicp/nanoflann.cc
  ThirdParty/nano_gicp/lsq_registration.cc
  ThirdParty/nano_gicp/nano_gicp.cc
)

target_link_libraries(nano_gicp_lib
  ${PCL_LIBRARIES}
  ${OPENMP_LIBS}
)

target_include_directories(nano_gicp_lib PUBLIC
  ThirdParty/nano_gicp/include
  ${PCL_INCLUDE_DIRS}
)


# 添加ndt_omp库
# add_library(ndt_omp_lib
#   ThirdParty/ndt_omp/src/pclomp/voxel_grid_covariance_omp.cpp
#   ThirdParty/ndt_omp/src/pclomp/ndt_omp.cpp
#   ThirdParty/ndt_omp/src/pclomp/gicp_omp.cpp
# )

# 包含ndt_omp头文件目录
# target_include_directories(ndt_omp_lib PUBLIC
#   ThirdParty/ndt_omp/include
#   ${PCL_INCLUDE_DIRS}
# )

# # Ensure OpenMP flags for compilation as well
# if (OpenMP_CXX_FOUND)
#   target_compile_options(ndt_omp_lib PRIVATE ${OpenMP_CXX_FLAGS})
# endif()

# target_link_libraries(ndt_omp_lib
#   ${PCL_LIBRARIES}
# )


catkin_package(
        INCLUDE_DIRS include
        DEPENDS PCL GTSAM

        CATKIN_DEPENDS
        std_msgs
        nav_msgs
        geometry_msgs
        sensor_msgs
        message_runtime
        message_generation
        visualization_msgs
)

# include directories
include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${GTSAM_INCLUDE_DIR}
        ${Y_INCLUDE_DIRS}
        ${LIBXML2_INCLUDE_DIRS}
        ./ThirdParty/GeographicLib/include/
        ./ThirdParty/ndt_omp/include
)

# link directories
link_directories(
        include
        ${PCL_LIBRARY_DIRS}
        ${OpenCV_LIBRARY_DIRS}
        ${GTSAM_LIBRARY_DIRS}
        ${LIBXML2_LIBRARY_DIR}
        ${Y_LIBRARY_DIRS}
)



# Ensure OpenMP flags for nano_gicp_lib too
if (OpenMP_CXX_FOUND)
  target_compile_options(nano_gicp_lib PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# if(OpenMP_CXX_FOUND)
#   target_link_libraries(ndt_omp_lib OpenMP::OpenMP_CXX)
# endif()

###########
## Build ##
###########


# Gps odometry
add_executable(${PROJECT_NAME}_gpsOdometry src/simpleGpsOdom.cpp)
target_link_libraries(${PROJECT_NAME}_gpsOdometry ${catkin_LIBRARIES} ${LIBXML2_LIBRARIES} ${Y_LIBRARIES} libGeographic Geographic)

# Range Image Projection
add_executable(${PROJECT_NAME}_imageProjection src/imageProjection.cpp)
add_dependencies(${PROJECT_NAME}_imageProjection ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_imageProjection ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${LIBXML2_LIBRARIES} ${Y_LIBRARIES})

# Feature Association
add_executable(${PROJECT_NAME}_featureExtraction src/featureExtraction.cpp)
add_dependencies(${PROJECT_NAME}_featureExtraction ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_featureExtraction ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${LIBXML2_LIBRARIES} ${Y_LIBRARIES})

# Mapping Optimization
add_executable(${PROJECT_NAME}_mapOptmization src/mapOptmizationGps.cpp src/dataSaver.cpp)
add_dependencies(${PROJECT_NAME}_mapOptmization ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_mapOptmization PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_mapOptmization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam ${LIBXML2_LIBRARIES} ${Y_LIBRARIES} ${GeographicLib_LIBRARIES} 
        /usr/local/lib/libtbb.so
        /usr/local/lib/libtbbmalloc.so
        /usr/local/lib/libtbbmalloc_proxy.so
        nano_gicp_lib 
        # ndt_omp_lib
        ${PCL_LIBRARIES}
        )


# IMU Preintegration
add_executable(${PROJECT_NAME}_imuPreintegration src/imuPreintegration.cpp)
target_link_libraries(${PROJECT_NAME}_imuPreintegration ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam ${LIBXML2_LIBRARIES} ${Y_LIBRARIES}
        /usr/local/lib/libtbb.so
        /usr/local/lib/libtbbmalloc.so
        /usr/local/lib/libtbbmalloc_proxy.so)
