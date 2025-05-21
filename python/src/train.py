import torch
import torch.nn as nn
import torch.optim as optim
from .data_loader import get_data_loaders
from .model import FaceNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt

def train_model(config):
    """训练模型并保存训练过程信息"""
    # 获取当前时间并格式化为字符串，用于创建唯一的训练文件夹名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_id = f"train_{timestamp}"
    log_dir = os.path.join("./log", train_id)
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, f"training_{timestamp}.log")

    def log(message):
        """同时将信息打印到控制台和日志文件"""
        print(message)
        with open(log_file_path, "a") as f:
            f.write(f"{message}\n")

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"使用设备: {device}")

    # 保存设备信息到训练文件夹
    with open(os.path.join(log_dir, "device_info.txt"), "w") as f:
        f.write(f"Device: {device}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA Device Name: {torch.cuda.get_device_name(0)}\n")
            f.write(f"CUDA Version: {torch.version.cuda}\n")
        else:
            f.write("CUDA is not available.\n")

    # 加载数据
    train_loader, val_loader = get_data_loaders(config)
    train_dataset_size = len(train_loader.dataset)
    val_dataset_size = len(val_loader.dataset)
    log(f"训练集大小: {train_dataset_size}")
    log(f"验证集大小: {val_dataset_size}")

    # 保存数据集大小信息到训练文件夹
    with open(os.path.join(log_dir, "dataset_info.txt"), "w") as f:
        f.write(f"Train dataset size: {train_dataset_size}\n")
        f.write(f"Validation dataset size: {val_dataset_size}\n")

    # 保存config信息到训练文件夹
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # 初始化模型
    model = FaceNet(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    # 存储训练过程中的损失和准确率
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # 记录开始训练的时间
    start_time = time.time()

    # 训练循环
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for i, batch in enumerate(train_loader):
            if config['mode'] == 'image_only':
                images, labels = batch
                images = images.to(device)
                structured = None
            elif config['mode'] == 'structured_only':
                structured, labels = batch
                structured = structured.to(device)
                images = None
            else:  # fusion
                images, structured, labels = batch
                images = images.to(device)
                structured = structured.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, structured)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        log(f'Epoch [{epoch+1}/{config["num_epochs"]}], Average Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                if config['mode'] == 'image_only':
                    images, labels = batch
                    images = images.to(device)
                    structured = None
                elif config['mode'] == 'structured_only':
                    structured, labels = batch
                    structured = structured.to(device)
                    images = None
                else:  # fusion
                    images, structured, labels = batch
                    images = images.to(device)
                    structured = structured.to(device)
                labels = labels.to(device)

                outputs = model(images, structured)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        log(f'Epoch [{epoch+1}/{config["num_epochs"]}], Validation Loss: {avg_val_loss:.4f}, Validation Acc: {val_acc:.2f}%\n')

        scheduler.step(avg_val_loss) # 更新学习率

    # 记录结束训练的时间
    end_time = time.time()
    training_time = end_time - start_time
    log(f"训练总耗时: {training_time:.2f} 秒 ({training_time/60:.2f} 分钟)")

    # 保存训练时间信息到训练文件夹
    with open(os.path.join(log_dir, "training_time.txt"), "w") as f:
        f.write(f"Start Time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End Time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n")

    # 保存模型
    model_name = os.path.join(log_dir, f"{config['mode']}_model_{train_id}.pth")
    torch.save(model.state_dict(), model_name)
    log(f"模型保存在: {model_name}")

    # 绘制并保存训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plot_path = os.path.join(log_dir, "training_curves.png")
    plt.savefig(plot_path)
    log(f"训练曲线图保存在: {plot_path}")
    plt.close()