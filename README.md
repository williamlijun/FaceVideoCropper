# 项目简介

一个专门用于目标人物识别和目标人物片段裁剪的工具。

项目的整体架构如下图所示
![项目整体架构](/show/系统架构.png)

该项目的基本功能如下：

- 识别一段视频中的目标人物并标注
- 将含有目标人物的片段剪切下来并形成新的视频
- 剪切目标人物的面部图片
- 可以训练不同的针对人脸的分类任务

对于目标人物处理模块，该模块的各个类设计与关系如下图所示：
![类设计](/show/人脸处理模块类图.png)

在项目下有一个 OpenFace 文件夹，该部分是用于结合openface工具实现面部地标提取、视线分析等功能的，如果需要使用这部分内容，请参考openface项目官网：https://github.com/TadasBaltrusaitis/OpenFace
如果存在 openface 配置问题可以参考 ./OpenFace/InstallOpenface2.0.md

在项目下有一个 python 文件夹，该部分是用于深度学习训练，用于解决一些人脸分类问题（比如情绪识别、注意力分析、疲劳度检测等）（对于这一部分的使用参考 ./python/README.md），我们设计了一个通用的针对人脸分类任务的网络，该网络结构如下图所示：
![分类网络结构](/show/分类网络.png)

# 项目配置

- 操作系统： linux
- 开发语言： C++、Python
- 依赖库： OpenCV、ONNXRuntime、Eigen、FFmpeg
- 有关环境配置请查看 ./notes/环境配置.md
- 模型下载请查看 ./notes/模型下载.md
- 请着重查看 CMakeLists.txt 文件

由于GitHub的上传大小限制，对于某些大型文件请在百度网盘获取：

通过网盘分享的文件：FaceVideoCropper_大型文件
链接: https://pan.baidu.com/s/1lrRAs8U3XGtRiIIzc0yprg?pwd=32bp 提取码: 32bp

环境配置成功后即可编译源码，步骤如下（注意：请在项目根目录下执行）：

    mkdir build
    cd ./build
    cmake ..
    make

编译成功后应在 ./build 目录下生成两个可执行文件： FaceVideoCropper FaceCropping

# 如何使用

- FaceVideoCropper：用于实现目标人物识别、标注、目标人物裁剪等功能
- FaceCropping：用于裁剪目标人物面部图像

FaceVideoCropper

在根目录下运行

    ./build/FaceVideoCropper 输入视频路径 目标人物图片路径 输出文件路径 人脸相似度阈值

例如

    ./build/FaceVideoCropper ./data/input_video.mp4 ./data/target_face.jpg ./data/output 0.35

- ./data/input_video.mp4 ：需要处理的视频的路径
- ./data/target_face.jpg ：需要识别的目标人物图像的路径（需要提供你想要识别的目标人物图像，该图像要含有清晰的面部）
- ./data/output ：输出结果保存路径
- 0.35 :这是进行人脸识别的相似度判断阈值（一般设置为 0.35 即可，如果要提高准确度可以适当增加该阈值）

该命令会输出以下信息

- 首先实时显示识别结果，在视频中进行标注，目标人物面部使用绿色标注框，其他人物使用红色标注框，标注框上显示了人物的id以及相似度信息
- 输出三个文件分别是：output_tarcked.avi（带有标注框的原始视频）、target_clips.avi（剪切的目标人物片段）、video_rois.txt（目标人物片段的帧信息）
- video_rois.txt 有四个字段，它们分别表示 frame_id（该片段在原始视频中的帧序号）, x, y, width, height（这四个字段是人脸标注框信息）
- 此外会复制原始视频和目标人物图片到输出文件夹

FaceCropping
注意：该程序需要在 FaceVideoCropper 执行之后才能运行，因为它需要使用到target_clips.avi文件和video_rois.txt文件
在根目录下运行

    ./build/FaceCropping 处理好的目标人物片段 目标人物片段的帧信息 输出文件路径

例如

    ./build/FaceCropping ./data/output/target_clips.avi ./data/output/video_rois.txt ./data/output/face

- ./data/output/target_clips.avi：这是经过FaceVideoCropper剪切好的目标人物片段
- ./data/output/video_rois.txt：这是对应的帧信息
- ./data/output/face：输出路径

# 效果展示

由于该项目的处理对象是视频，所以文件比较大无法上传，如需查看项目效果，请在之前在百度网盘（https://pan.baidu.com/s/1lrRAs8U3XGtRiIIzc0yprg?pwd=32bp 提取码: 32bp）中下载的大型文件中的data文件夹中查看

# 声明

该项目使用了 OpenFace 项目用于结构化数据提取。

视频素材均为网络查找所得，若有侵权行为欢迎联系作者进行删除