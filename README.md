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

输入：./data/黄渤_test.mp4

    ./build/FaceVideoCropper ./data/黄渤_test.mp4 ./data/黄渤.png ./data/黄渤_output 0.4

    成功设置相似度阈值为: 0.4
    文件夹已存在: ./data/黄渤_output
    视频文件: ./data/黄渤_test.mp4
    -----------------------------------
    帧率 (FPS): 30
    帧尺寸 (宽度 x 高度): 636 x 360
    编码格式 (FourCC): h264
    总帧数: 2281
    视频总时长: 1 分钟 16.03 秒
    -----------------------------------
    [ WARN:0@1.126] global matrix_expressions.cpp:1333 assign OpenCV/MatExpr: processing of multi-channel arrays might be changed in the future: https://github.com/opencv/opencv/issues/16739

    ------------------- 耗时统计 -------------------
    总处理帧数: 2281
    总处理时间: 1350.719 秒
    平均每帧处理时间: 0.592 秒

    各模块耗时:
    - 人脸检测总耗时: 237.470 秒
        - 平均每帧耗时: 0.104 秒
    - 目标跟踪总耗时: 0.016 秒
        - 平均每帧耗时: 0.000 秒
    - 人脸识别总耗时: 1098.259 秒
        - 平均每帧耗时: 0.481 秒
    ------------------------------------------------

    开始剪辑包含目标人物的视频片段...
    已保存片段: ./data/黄渤_output/clip_0.avi (帧 98 - 139)
    已保存片段: ./data/黄渤_output/clip_1.avi (帧 176 - 178)
    已保存片段: ./data/黄渤_output/clip_2.avi (帧 184 - 212)
    已保存片段: ./data/黄渤_output/clip_3.avi (帧 216 - 216)
    已保存片段: ./data/黄渤_output/clip_4.avi (帧 225 - 226)
    已保存片段: ./data/黄渤_output/clip_5.avi (帧 229 - 231)
    已保存片段: ./data/黄渤_output/clip_6.avi (帧 284 - 308)
    已保存片段: ./data/黄渤_output/clip_7.avi (帧 310 - 313)
    已保存片段: ./data/黄渤_output/clip_8.avi (帧 330 - 340)
    已保存片段: ./data/黄渤_output/clip_9.avi (帧 385 - 441)
    已保存片段: ./data/黄渤_output/clip_10.avi (帧 571 - 622)
    已保存片段: ./data/黄渤_output/clip_11.avi (帧 862 - 868)
    已保存片段: ./data/黄渤_output/clip_12.avi (帧 870 - 872)
    已保存片段: ./data/黄渤_output/clip_13.avi (帧 874 - 876)
    已保存片段: ./data/黄渤_output/clip_14.avi (帧 878 - 881)
    已保存片段: ./data/黄渤_output/clip_15.avi (帧 889 - 898)
    已保存片段: ./data/黄渤_output/clip_16.avi (帧 1024 - 1126)
    已保存片段: ./data/黄渤_output/clip_17.avi (帧 1130 - 1179)
    已保存片段: ./data/黄渤_output/clip_18.avi (帧 1475 - 1480)
    已保存片段: ./data/黄渤_output/clip_19.avi (帧 1505 - 1597)
    已保存片段: ./data/黄渤_output/clip_20.avi (帧 1658 - 1675)
    已保存片段: ./data/黄渤_output/clip_21.avi (帧 1677 - 1698)
    已保存片段: ./data/黄渤_output/clip_22.avi (帧 1704 - 1704)
    已保存片段: ./data/黄渤_output/clip_23.avi (帧 1707 - 1727)
    已保存片段: ./data/黄渤_output/clip_24.avi (帧 1731 - 1734)
    已保存片段: ./data/黄渤_output/clip_25.avi (帧 1738 - 1751)
    已保存片段: ./data/黄渤_output/clip_26.avi (帧 1753 - 1764)
    已保存片段: ./data/黄渤_output/clip_27.avi (帧 1850 - 1919)
    已保存片段: ./data/黄渤_output/clip_28.avi (帧 1933 - 1956)
    已保存片段: ./data/黄渤_output/clip_29.avi (帧 1958 - 1969)
    已保存片段: ./data/黄渤_output/clip_30.avi (帧 1971 - 1971)
    已保存片段: ./data/黄渤_output/clip_31.avi (帧 1980 - 2002)
    已保存片段: ./data/黄渤_output/clip_32.avi (帧 2008 - 2021)
    已保存片段: ./data/黄渤_output/clip_33.avi (帧 2023 - 2023)
    已保存片段: ./data/黄渤_output/clip_34.avi (帧 2090 - 2135)
    已保存片段: ./data/黄渤_output/clip_35.avi (帧 2185 - 2198)
    ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023 the FFmpeg developers
    built with gcc 13 (Ubuntu 13.2.0-23ubuntu3)
    configuration: --prefix=/usr --extra-version=3ubuntu5 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --disable-omx --enable-gnutls --enable-libaom --enable-libass --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libdav1d --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libglslang --enable-libgme --enable-libgsm --enable-libharfbuzz --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzimg --enable-openal --enable-opencl --enable-opengl --disable-sndio --enable-libvpl --disable-libmfx --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-chromaprint --enable-frei0r --enable-ladspa --enable-libbluray --enable-libjack --enable-libpulse --enable-librabbitmq --enable-librist --enable-libsrt --enable-libssh --enable-libsvtav1 --enable-libx264 --enable-libzmq --enable-libzvbi --enable-lv2 --enable-sdl2 --enable-libplacebo --enable-librav1e --enable-pocketsphinx --enable-librsvg --enable-libjxl --enable-shared
    libavutil      58. 29.100 / 58. 29.100
    libavcodec     60. 31.102 / 60. 31.102
    libavformat    60. 16.100 / 60. 16.100
    libavdevice    60.  3.100 / 60.  3.100
    libavfilter     9. 12.100 /  9. 12.100
    libswscale      7.  5.100 /  7.  5.100
    libswresample   4. 12.100 /  4. 12.100
    libpostproc    57.  3.100 / 57.  3.100
    Input #0, concat, from './data/黄渤_output/concat_list.txt':
    Duration: N/A, start: 0.000000, bitrate: 5084 kb/s
    Stream #0:0: Video: mjpeg (Baseline) (MJPG / 0x47504A4D), yuvj420p(pc, bt470bg/unknown/unknown), 636x360, 5084 kb/s, 30 fps, 30 tbr, 30 tbn
    Output #0, avi, to './data/黄渤_output/target_clips.avi':
    Metadata:
        ISFT            : Lavf60.16.100
    Stream #0:0: Video: mjpeg (Baseline) (MJPG / 0x47504A4D), yuvj420p(pc, bt470bg/unknown/unknown), 636x360, q=2-31, 5084 kb/s, 30 fps, 30 tbr, 30 tbn
    Stream mapping:
    Stream #0:0 -> #0:0 (copy)
    Press [q] to stop, [?] for help
    [out#0/avi @ 0x555db487bd00] video:18409kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.134753%
    size=   18434kB time=00:00:26.80 bitrate=5634.7kbits/s speed= 322x    

    已成功剪辑包含目标人物的视频并保存到: ./data/黄渤_output/target_clips.avi
    目标 ROI 信息已保存到: ./data/黄渤_output/video_rois.txt
    目标文件夹中已存在同名文件: ./data/黄渤_output/黄渤_test.mp4，跳过复制。
    目标文件夹中已存在同名文件: ./data/黄渤_output/黄渤.png，跳过复制。

    ./build/FaceCropping ./data/黄渤_output/target_clips.avi ./data/黄渤_output/video_rois.txt ./data/黄渤_output/face
    
    成功加载 805 个标注框信息。
    共保存 805 张面部图像到 ./data/黄渤_output/face
    面部裁剪完成。

输出：./data/黄渤_output/

[输入视频](data/黄渤_test.mp4){:target="input_video"}

[输出视频](data/黄渤_output/output_tracked.avi)

[剪切视频](data/黄渤_output/target_clips.avi)

人脸剪切

![图片alt](/data/HuangBo_output/face/face_000000.jpg)

[rois文件](data/黄渤_output/video_rois.txt)

有关详细数据请查看 /data/HuangBo_output 文件夹内容

# 声明

该项目使用了 OpenFace 项目用于结构化数据提取。
视频素材均为网络查找所得，若有侵权行为欢迎联系作者进行删除