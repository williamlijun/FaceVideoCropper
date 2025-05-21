import json
from src.train import train_model

def main():
    """主函数"""
    # 加载配置文件
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # 训练模型
    train_model(config)

if __name__ == '__main__':
    main()