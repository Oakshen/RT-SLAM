cmake_minimum_required(VERSION 3.2)
project(ORB_SLAM3)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if (COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    add_definitions(-DCOMPILEDWITHC17)
    message(STATUS "Using flag -std=c++17.")
elseif (COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    add_definitions(-DCOMPILEDWITHC14)
    message(STATUS "Using flag -std=c++14.")
elseif (COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++11.")
elseif (COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

add_definitions(-w)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/Thirdparty/libtorch)

find_package(OpenCV 4.8 REQUIRED)
MESSAGE("OPENCV VERSION:")
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 4.4 not found.")
endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(realsense2)
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(epoxy)

# Set TensorRT directory (modify accordingly)
set(TENSORRT_DIR $ENV{HOME}/TensorRT-8.6.1.6/)

execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE GPU_COMPUTE_CAP
    RESULT_VARIABLE GPU_COMPUTE_CAP_RESULT
)

if (GPU_COMPUTE_CAP_RESULT EQUAL 0)
    string(REPLACE "\n" ";" GPU_COMPUTE_CAP_LIST ${GPU_COMPUTE_CAP})
    foreach(GPU_CAP ${GPU_COMPUTE_CAP_LIST})
        string(STRIP ${GPU_CAP} GPU_CAP)
        message("GPU Compute Capability: ${GPU_CAP}")
        string(REGEX REPLACE "\\." ";" COMP_CAP_LIST ${GPU_CAP})
        list(GET COMP_CAP_LIST 0 COMPUTE_CAP_MAJOR)
        list(GET COMP_CAP_LIST 1 COMPUTE_CAP_MINOR)
        set(CUDA_COMPUTE "compute_${COMPUTE_CAP_MAJOR}${COMPUTE_CAP_MINOR}")
        set(CUDA_SM "sm_${COMPUTE_CAP_MAJOR}${COMPUTE_CAP_MINOR}")
        message("Setting -gencode;arch=${CUDA_COMPUTE};code=${CUDA_SM} for GPU ${GPU_CAP}")
        set(CUDA_NVCC_FLAGS_RELEASE_${GPU_CAP} ${CUDA_NVCC_FLAGS_RELEASE};-O3;-gencode;arch=${CUDA_COMPUTE};code=${CUDA_SM})
        set(CUDA_NVCC_FLAGS_DEBUG_${GPU_CAP} ${CUDA_NVCC_FLAGS_DEBUG};-g;-G;-gencode;arch=${CUDA_COMPUTE};code=${CUDA_SM})
    endforeach()
else()
    message("Failed to query GPU compute capability.")
endif()

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/CameraModels
        ${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        /usr/local/cuda/include
        ${TENSORRT_DIR}/include
)

link_directories(
        /usr/local/cuda/lib64
        ${TENSORRT_DIR}/lib
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(TENSORRT_SOURCES
    src//TRTInfer.cpp
    src//RtDetrUltralytics.cpp
)

add_library(${PROJECT_NAME} SHARED
        src/System.cc
        src/Tracking.cc
        src/LocalMapping.cc
        src/LoopClosing.cc
        src/ORBextractor.cc
        src/ORBmatcher.cc
        src/FrameDrawer.cc
        src/Converter.cc
        src/MapPoint.cc
        src/KeyFrame.cc
        src/Atlas.cc
        src/Map.cc
        src/MapDrawer.cc
        src/Optimizer.cc
        src/Frame.cc
        src/KeyFrameDatabase.cc
        src/Sim3Solver.cc
        src/Viewer.cc
        src/ImuTypes.cc
        src/G2oTypes.cc
        src/CameraModels/Pinhole.cpp
        src/CameraModels/KannalaBrandt8.cpp
        src/OptimizableTypes.cpp
        src/MLPnPsolver.cpp
        src/GeometricTools.cc
        src/TwoViewReconstruction.cc
        src/Config.cc
        src/Settings.cc
        src/RtDetect.cpp
        ${TENSORRT_SOURCES}
        include/System.h
        include/Tracking.h
        include/LocalMapping.h
        include/LoopClosing.h
        include/ORBextractor.h
        include/ORBmatcher.h
        include/FrameDrawer.h
        include/Converter.h
        include/MapPoint.h
        include/KeyFrame.h
        include/Atlas.h
        include/Map.h
        include/MapDrawer.h
        include/Optimizer.h
        include/Frame.h
        include/KeyFrameDatabase.h
        include/Sim3Solver.h
        include/Viewer.h
        include/ImuTypes.h
        include/G2oTypes.h
        include/CameraModels/GeometricCamera.h
        include/CameraModels/Pinhole.h
        include/CameraModels/KannalaBrandt8.h
        include/OptimizableTypes.h
        include/MLPnPsolver.h
        include/GeometricTools.h
        include/TwoViewReconstruction.h
        include/SerializationUtils.h
        include/Config.h
        include/Settings.h
        include/RtDetect.h
        include//TRTInfer.hpp
        include//RtDetrUltralytics.hpp)

add_subdirectory(Thirdparty/g2o)

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        ${TORCH_LIBRARIES}
        ${PCL_LIBRARIES}
        nvinfer
        nvonnxparser
        cudart
        -lboost_serialization
        -lcrypto
)

if(realsense2_FOUND)
    include_directories(${PROJECT_NAME}
            ${realsense_INCLUDE_DIR}
            )
    target_link_libraries(${PROJECT_NAME}
            ${realsense2_LIBRARY}
            )
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum
        Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(rgbd_realsense_D435i
            Examples/RGB-D/rgbd_realsense_D435i.cc)
    target_link_libraries(rgbd_realsense_D435i ${PROJECT_NAME})
endif()
