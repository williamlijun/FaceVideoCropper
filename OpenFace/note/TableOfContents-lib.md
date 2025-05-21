
    lib
        3rdParty
            CameraEnumerator
                CameraEnumerator.vcxproj           // Visual Studio项目文件，用于编译摄像头枚举器
                CameraEnumerator.vcxproj.filters   // Visual Studio项目过滤器文件，用于组织项目文件
                DeviceEnumerator.cpp               // 摄像头设备枚举器实现
                DeviceEnumerator.h                 // 摄像头设备枚举器头文件
                OpenCVDeviceEnumerator.cpp         // 使用OpenCV的摄像头设备枚举器实现
                readme.md                          // 摄像头枚举器说明文档
            OpenBLAS
                bin                                // OpenBLAS库的二进制文件（动态链接库）
                include                            // OpenBLAS库的头文件
                lib                                // OpenBLAS库的静态链接库
                OpenBLAS_64.props                 // Visual Studio属性表，用于64位OpenBLAS配置
                OpenBLAS_x86.props                // Visual Studio属性表，用于32位OpenBLAS配置
                readme.txt                         // OpenBLAS库的说明文档
                readme_2.txt                       // OpenBLAS库的说明文档
            OpenCV
                classifiers                        // OpenCV的分类器数据（用于人脸检测等）
                include/opencv2                    // OpenCV的头文件
                x64/v141/lib                       // 64位OpenCV库文件（Visual Studio 2017）
                x86/v141/lib                       // 32位OpenCV库文件（Visual Studio 2017）
                openCV.props                       // Visual Studio属性表，用于OpenCV配置
            dlib
                include/dlib                       // dlib库的头文件
                lib                                // dlib库的静态链接库
                LICENSE.txt                        // dlib库的许可证文件
                MANIFEST.in                        // Python打包相关文件
                README.md                          // dlib库的说明文档
                README.txt                         // dlib库的说明文档
                dlib.props                         // Visual Studio属性表，用于dlib配置
                installation_instructions_windows.txt // dlib库在windows下的安装说明
        local
            CppInerop
                AssemblyInfo.cpp                   // C++互操作程序集信息
                CppInerop.vcxproj                   // Visual Studio项目文件，用于编译C++互操作库
                CppInerop.vcxproj.filters           // Visual Studio项目过滤器文件
                CppInterop.cpp                     // C++互操作实现
                FaceAnalyserInterop.h              // 人脸分析器互操作头文件
                FaceDetectorInterop.h              // 人脸检测器互操作头文件
                GazeAnalyserInterop.h              // 视线分析器互操作头文件
                ImageReader.h                      // 图像读取器头文件
                LandmarkDetectorInterop.h          // 地标检测器互操作头文件
                OpenCVWrappers.h                   // OpenCV包装器头文件
                RecorderInterop.h                  // 录制器互操作头文件
                SequenceReader.h                   // 序列读取器头文件
                VisualizerInterop.h                // 可视化器互操作头文件
            FaceAnalyser
                AU_predictors                      // 动作单元预测器数据
                include                            // 人脸分析器头文件
                src                                // 人脸分析器源代码
                CMakeLists.txt                     // CMake构建配置文件
                FaceAnalyser.vcxproj                // Visual Studio项目文件
                FaceAnalyser.vcxproj.filters        // Visual Studio项目过滤器文件
            GazeAnalyser
                include                            // 视线分析器头文件
                src                                // 视线分析器源代码
                CMakeLists.txt                     // CMake构建配置文件
                GazeAnalyser.vcxproj                // Visual Studio项目文件
                GazeAnalyser.vcxproj.filters        // Visual Studio项目过滤器文件
            LandmarkDetector
                include                            // 地标检测器头文件
                model                              // 地标检测器模型数据
                src                                // 地标检测器源代码
                CMakeLists.txt                     // CMake构建配置文件
                LandmarkDetector.vcxproj            // Visual Studio项目文件
                LandmarkDetector.vcxproj.filters    // Visual Studio项目过滤器文件
            Utilities
                include                            // 实用工具库头文件
                src                                // 实用工具库源代码
                CMakeLists.txt                     // CMake构建配置文件
                Utilities.vcxproj                  // Visual Studio项目文件
                Utilities.vcxproj.filters          // Visual Studio项目过滤器文件