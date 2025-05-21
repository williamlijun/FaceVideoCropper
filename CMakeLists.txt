cmake_minimum_required(VERSION 3.15)
project(FaceVideoCropper LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 寻找 OpenCV
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBS}) # 确保链接库也被包含

# 寻找 Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# 设置 ONNXRuntime 路径
set(ONNXRUNTIME_DIR /usr/local/onnxruntime)
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# 包含项目头文件目录
include_directories(${CMAKE_SOURCE_DIR}/config)
include_directories(${CMAKE_SOURCE_DIR}/modules)
include_directories(${CMAKE_SOURCE_DIR}/utils)

# 添加源文件（主程序）
set(SOURCES
    main.cpp
    modules/video_io.cpp
    modules/face_detector.cpp
    modules/face_recognizer.cpp
    modules/target_tracker.cpp
    modules/video_clipper.cpp
    modules/target_processing.cpp
    utils/vector_math.cpp
    utils/timing_logger.cpp
    config/config.cpp
)

# 创建主程序可执行文件 (FaceVideoCropper)
add_executable(${PROJECT_NAME} ${SOURCES})

# 链接库（主程序）
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    Eigen3::Eigen
    onnxruntime
)

# 添加新的源文件（面部裁剪模块）
set(FACE_CLIPPER_SOURCES
    modules/face_clipper.cpp
)

# 包含 face_clipper 的头文件目录
include_directories(${CMAKE_SOURCE_DIR}/modules)

# 添加新的源文件（面部裁剪命令行工具）
set(FACE_CROPPING_SOURCES
    FaceCropping.cpp
)

# 创建新的可执行文件 (FaceCropping)
add_executable(FaceCropping ${FACE_CROPPING_SOURCES} ${FACE_CLIPPER_SOURCES})

# 链接库（面部裁剪命令行工具） - 需要 OpenCV
target_link_libraries(FaceCropping
    ${OpenCV_LIBS}
)