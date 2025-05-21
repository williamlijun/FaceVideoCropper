import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
import json
import numpy as np
import warnings
import random

class LoadDataset(Dataset):
    """数据集加载器"""
    def __init__(self, data_dir, dataname, mode, structured_features, transform=None):
        """
        初始化数据集
        Args:
            data_dir (str): 数据根目录
            dataname (str): 数据集名称
            mode (str): 输入模式 ('image_only', 'structured_only', 'fusion')
            structured_features (list): 选择的结构化特征
            transform: 图像预处理变换
        """
        self.data_dir = data_dir
        self.dataname = dataname
        self.mode = mode
        self.structured_features = structured_features
        self.transform = transform
        self.label_map = {}
        csv_dir = os.path.join(data_dir, 'dataset', dataname)
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        for idx, csv_file in enumerate(sorted(csv_files)):
            emotion = os.path.splitext(csv_file)[0]  # 提取文件名（不含 .csv）
            self.label_map[emotion] = idx
        
        # 加载所有 CSV 文件
        self.data = []
        for emotion in self.label_map.keys():
            csv_path = os.path.join(data_dir, 'dataset', dataname, f'{emotion}.csv')
            df = pd.read_csv(csv_path)
            self.data.append(df)
        self.data = pd.concat(self.data, ignore_index=True)
        
        # 清洗结构化特征数据
        if mode in ['structured_only', 'fusion']:
            # 转换为数值类型，处理非数值数据
            for feature in self.structured_features:
                try:
                    self.data[feature] = pd.to_numeric(self.data[feature], errors='coerce')
                except Exception as e:
                    warnings.warn(f"特征 {feature} 包含无法转换的数据: {str(e)}")
            
            # 处理缺失值
            num_missing = self.data[self.structured_features].isna().sum().sum()
            if num_missing > 0:
                warnings.warn(f"发现 {num_missing} 个缺失值，已填充为 0")
                self.data[self.structured_features] = self.data[self.structured_features].fillna(0)
            
            # 检查数据类型
            for feature in self.structured_features:
                if not np.issubdtype(self.data[feature].dtype, np.number):
                    warnings.warn(f"特征 {feature} 的类型为 {self.data[feature].dtype}，可能导致问题")
            
            # 结构化特征归一化
            self.scaler = StandardScaler()
            self.data[self.structured_features] = self.scaler.fit_transform(self.data[self.structured_features])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单条数据
        Args:
            idx (int): 数据索引
        Returns:
            tuple: 根据模式返回不同数据
                   - image_only: (image, label)
                   - structured_only: (structured, label)
                   - fusion: (image, structured, label)
        """
        row = self.data.iloc[idx]
        img_path = os.path.join(self.data_dir, 'unstructured', self.dataname, row['img_path'])
        label = self.label_map[row['label']]
        
        if self.mode == 'image_only':
            # 仅加载图像
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
            except Exception as e:
                warnings.warn(f"无法加载图像 {img_path}: {str(e)}")
                image = torch.zeros(3, 224, 224)
            return image, label
        
        elif self.mode == 'structured_only':
            # 仅加载结构化特征
            try:
                structured_data = row[self.structured_features].values
                structured_data = np.array(structured_data, dtype=np.float32)
                structured = torch.from_numpy(structured_data)
            except Exception as e:
                warnings.warn(f"结构化特征转换失败 (index {idx}): {str(e)}")
                structured = torch.zeros(len(self.structured_features))
            return structured, label
        
        else:  # fusion
            # 加载图像
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
            except Exception as e:
                warnings.warn(f"无法加载图像 {img_path}: {str(e)}")
                image = torch.zeros(3, 224, 224)
            
            # 加载结构化特征
            try:
                structured_data = row[self.structured_features].values
                structured_data = np.array(structured_data, dtype=np.float32)
                structured = torch.from_numpy(structured_data)
            except Exception as e:
                warnings.warn(f"结构化特征转换失败 (index {idx}): {str(e)}")
                structured = torch.zeros(len(self.structured_features))
            
            return image, structured, label

def split_dataset(dataset, train_ratio=0.8, trim_ratio=1.0, random_seed=42):
    """
    按指定比例随机划分数据集，并设置随机种子以保证可重复性。

    Args:
        dataset (torch.utils.data.Dataset): 要划分的数据集。
        train_ratio (float): 训练集所占的比例 (0.0 到 1.0)。
        trim_ratio (float): 最终保留的原始数据集的比例 (0.0 到 1.0)。
        random_seed (int): 随机种子。

    Returns:
        tuple: 包含训练集和验证集两个 Dataset 对象。
    """
    # 设置 Python 的随机种子
    random.seed(random_seed)
    # 设置 NumPy 的随机种子
    np.random.seed(random_seed)
    # 设置 PyTorch 的随机种子 (CPU)
    torch.manual_seed(random_seed)
    # 设置 PyTorch 的随机种子 (GPU，如果可用)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    n_samples = len(dataset)
    indices = list(range(n_samples))
    # 随机选择要保留的数据的索引
    trimmed_indices = random.sample(indices, int(n_samples * trim_ratio))
    trimmed_dataset = torch.utils.data.Subset(dataset, trimmed_indices)

    train_size = int(train_ratio * len(trimmed_dataset))
    val_size = len(trimmed_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trimmed_dataset, [train_size, val_size])
    return train_dataset, val_dataset

def get_data_loaders(config):
    """创建数据加载器"""
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LoadDataset(
        data_dir=config['data_dir'],
        dataname=config['dataname'],
        mode=config['mode'],
        structured_features=config['structured_features'],
        transform=transform
    )
    
    # 裁剪并划分数据集
    train_dataset, val_dataset = split_dataset(dataset, train_ratio=config['train_ratio'], trim_ratio=config['trim_ratio'], random_seed=42)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    return train_loader, val_loader