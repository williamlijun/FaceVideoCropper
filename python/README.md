当前工作目录 ./python

# 配置python环境

创建一个python的虚拟环境

    python3 -m venv venv

启动虚拟环境

    source ./venv/bin/activate

配置ipynb文件运行环境（可以不用配置）

首先安装Jupyter 内核与 IPython 之间的桥梁，VS Code 需要它来运行 Jupyter Notebook 文件

    pip install ipykernel

安装 ipykernel 后，将这个虚拟环境的内核注册到 Jupyter 可以找到的位置。在激活的虚拟环境中运行以下命令：

    python -m ipykernel install --user --name=venv

venv 是这个内核的命名

比如

    python -m ipykernel install --user --name=venv


cuda 安装：https://www.cnblogs.com/caicai45/p/17303607.html
https://blog.csdn.net/LeMonMannnn/article/details/130987243
以上教程仅供参考，只需要在下列连接下安装电脑对应cuda版本即可：https://developer.nvidia.com/cuda-toolkit-archive


安装依赖
在 ./python 下执行

    pip install -r requirements.txt

# 数据下载

https://huggingface.co/datasets/chitradrishti/AffectNet/tree/main

请将数据解压到 ./data/unstructured 即数据集应在 ./data/unstructured/affectnet

# config.json 文件说明

    {
        "data_dir": "./data",               // 原始数据集所在位置
        "dataname": "affectnet",            // 数据集名称（位置）（./data/affectnet）
        "image_size": 224,                  // 图片大小 224*224
        "batch_size": 32,                   // 批处理大小
        "image_feature_dim": 512,           // 图像特征维度
        "struct_feature_dim": 256,          // 结构化特征维度
        "fused_dim": 512,                   // 最终融合的特征维度
        "num_classes": 8,                   // 分类的数量，与数据集标签一致
        "train_ratio": 0.8,                 // 训练集划分比例 0.8 : 0.2
        "trim_ratio": 0.01,                 // 读取的数据集大小 1.0%
        "mode": "fusion",                   // 分类模型选择，可选项有："image_only"、"structured_only"、"fusion"
        "backbone": "mobilenet_v3_small",   // 图像特征提取的模型选择，可选项有：'mobilenet_v3_small', 'efficientnet_b0', 'squeezenet'
        "structured_features": [
        ],                                  // OpenFace的输出字段，比如"gaze_0_x", "gaze_0_y", "gaze_0_z"
        "learning_rate": 0.001,             // 学习率
        "num_epochs": 20                    // 训练轮数
    }

# 如何使用（训练）
首先当前工作目录是 ./python

进行数据处理，将原始数据经过Openface处理后存入./data/structured文件夹
之后根据你的文件夹名称修改 ./data/join.py 中的文件路径，进行数据预处理
此时应在 ./data/dataset 下出现你的数据集关联文件。
最后修改 config.json 文件执行 ./main.py 脚本即可