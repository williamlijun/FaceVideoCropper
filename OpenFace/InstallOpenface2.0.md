# 如何在windows环境下安装并使用openface2.0

## 使用虚拟技术安装Ubuntu

安装wsl以及一个linux系统

然后进入wsl的Linux环境

## 创建虚拟环境

安装 Python 和创建虚拟环境

在 Ubuntu 中安装 Python 通过 WSL 的终端进入 Ubuntu 环境，运行以下命令：

    sudo apt update
    sudo apt install -y python3 python3-venv python3-pip

创建虚拟环境 在你的项目目录（如 /home/yourname/projects/openface_project）下，运行：

    mkdir -p ~/projects/openface_project
    cd ~/projects/openface_project
    python3 -m venv venv
    source venv/bin/activate

验证虚拟环境 确保虚拟环境中 Python 和 pip 可用：

    python --version
    pip --version

## 安装 OpenFace 和依赖

克隆 OpenFace 仓库 使用以下命令将 OpenFace 仓库克隆到本地：

    git clone https://github.com/TadasBaltrusaitis/OpenFace.git

进入OpenFace安装目录

    cd OpenFace

安装依赖 在虚拟环境中安装 OpenFace 的依赖：

    sudo apt install -y cmake g++ libopenblas-dev liblapack-dev libopencv-dev
    pip install numpy opencv-python dlib

注意在下面操作之前要首先在系统中安装C++的dlib

    sudo apt install libdlib-dev

也可能需要安装boost：

    sudo apt install libboost-all-dev

## 下载模型文件

下载openface必须的模型

教程链接如下

https://github.com/TadasBaltrusaitis/OpenFace/wiki/Model-download

（此外如果需要请访问openface的github wiki教程查看）
或者采用下面的教程

https://blog.csdn.net/qq_45738497/article/details/143205414

## 编译OpenFace

编译 OpenFace 的 C++ 部分：

    mkdir build
    cd build
    cmake ..
    make -j$(nproc)

编译成功后，你将在 build/bin 下看到可执行文件 FeatureExtraction。

## 测试

    (venv) lee@DESKTOP-M1B3C6C:~/projects/decoding_the_game$ ./OpenFace/build/bin FaceLandmarkImg -f ./test.jpg

如上进入相应目录（记得先下载图片）进行测试

之后在decoding_the_game目录下会新建一个processed文件，里面为输出检测的结果，表示安装成功

Could not find the HAAR face detector location问题，此问题可以忽略，因为没有使用到HAAR face detector


注意一定要启用虚拟环境 source venv/bin/activate