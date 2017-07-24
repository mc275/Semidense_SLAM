# ORB-SLAM 项目管理。

# 确定最低版本CMake。
cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM_MC)

# 设置项目编译方式 Release。
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release )
ENDIF()

MESSAGE("Build Type: " ${CMAKE_BUILD_TYPE})

# 检查C++11或C++0x 特性支持。
add_definitions(-DCOMPILEDWITHC11)
if( CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -msse3 -std=c++11 -pthread -g -march=native")
else( CMAKE_BUILD_TYPE MATCHES "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -msse3 -std=c++11 -pthread -O3 -march=native")
endif( CMAKE_BUILD_TYPE MATCHES "Debug")

# CMake文件列表。
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# 查找依赖库。
find_package(OpenCV 2.4.3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)




# include工程目录。
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

# 设置CMake库输出目录。
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# 添加库 
add_library(${PROJECT_NAME} SHARED
src/System.cpp
src/Tracking.cpp
src/LocalMapping.cpp
src/LoopClosing.cpp
src/ORBextractor.cpp
src/ORBmatcher.cpp
src/FrameDrawer.cpp
src/Converter.cpp
src/MapPoint.cpp
src/KeyFrame.cpp
src/Map.cpp
src/MapDrawer.cpp
src/Optimizer.cpp
src/PnPsolver.cpp
src/Frame.cpp
src/KeyFrameDatabase.cpp
src/Sim3Solver.cpp
src/Initializer.cpp
src/Viewer.cpp
src/ProbabilityMapping.cpp
)

# 链接库。 
target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)

# 编译Vocabulary二进制文件
# 设置可执行文件输出目录。
#SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Vocabulary )
#add_executable(bin_vocabulary Vocabulary/bin_vocabulary.cpp)
#target_link_libraries(bin_vocabulary ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so ${OpenCV_LIBS})


# 编译单目KITTI数据集可执行文件
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)
add_executable(mono_kitti Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

# 编译单目TUM数据集可执行文件
add_executable(mono_tum Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

# 编译单目Semidense TUM数据集可执行文件
add_executable(mono_semidense Examples/Monocular/mono_semidense.cpp)
target_link_libraries(mono_semidense ${PROJECT_NAME})

# 编译单目Euroc数据集可执行文件
add_executable(mono_EuRoC Examples/Monocular/mono_EuRoC.cpp)
target_link_libraries(mono_EuRoC ${PROJECT_NAME})



# 编译双目kitti数据集。
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)
#add_executable(stereo_kitti Examples/Stereo/stereo_kitti.cpp)
#target_link_libraries(stereo_kitti ${PROJECT_NAME})

# 编译双目EuRoc数据集。
#add_executable(stereo_EuRoC Examples/Stereo/stereo_EuRoC.cpp)
#target_link_libraries(stereo_EuRoC ${PROJECT_NAME})



# 编译RGB-D Tum数据集。
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)
#add_executable(rgbd_tum Examples/RGB-D/rgbd_tum.cc)
#target_link_libraries(rgbd_tum ${PROJECT_NAME})












  
