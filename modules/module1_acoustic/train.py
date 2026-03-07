"""
弃用

train.py
统一训练脚本，支持：
- scikit-learn 模型 (One-Class SVM, Isolation Forest)
- PyTorch 自编码器模型 (AEDCASEBaseline)

训练 sklearn One-Class SVM 代码：
python train.py --backend sklearn --model_type svm --train_dir data/train_acoustic/normal --save_path modules/module1_acoustic/models/ocsvm.pkl --nu 0.1
训练 PyTorch 自编码器代码：
python train.py --backend pytorch --train_dir data/train_acoustic/normal --save_path modules/module1_acoustic/models/ae.pt --epochs 50 --batch_size 16
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import joblib
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

# 导入自定义模块
from audio_dataset import AudioDataset
from feature_extractor import FeatureExtractor
from audio_preprocessor import AudioPreprocessor
from acoustic_anomaly_model import AEDCASEBaseline, AnomalyModel

def train_sklearn(args):
    """训练 scikit-learn 模型（如 One-Class SVM）"""
    print("训练 sklearn 模型...")
    # 提取特征
    preprocessor = AudioPreprocessor(target_sr=16000)
    feature_extractor = FeatureExtractor(sr=16000)

    dataset = AudioDataset(root_dir=args.train_dir, target_sr=16000)
    features = []
    for i in range(len(dataset)):
        waveform = dataset[i].numpy()
        # 预处理（降噪、VAD等）
        clean = preprocessor.process(waveform)
        feat = feature_extractor.extract(clean)
        features.append(feat)

    X = np.array(features)
    print(f"提取到 {X.shape[0]} 个样本，特征维度 {X.shape[1]}")

    # 选择模型
    if args.model_type == 'svm':
        model = OneClassSVM(kernel='rbf', gamma='auto', nu=args.nu)
    elif args.model_type == 'iforest':
        model = IsolationForest(contamination=args.contamination, random_state=42)
    else:
        raise ValueError(f"未知 sklearn 模型类型: {args.model_type}")

    model.fit(X)
    # 保存模型
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    joblib.dump(model, args.save_path)
    print(f"模型已保存至 {args.save_path}")

def train_pytorch(args):
    """训练 PyTorch 自编码器模型"""
    print("训练 PyTorch 自编码器...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集与数据加载器
    dataset = AudioDataset(root_dir=args.train_dir, target_sr=16000)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 初始化模型
    model = AEDCASEBaseline().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 训练循环
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in dataloader:
            # batch: (batch_size, samples)
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_error = model(batch)  # 返回每个样本的重构误差
            # 训练目标是让重构误差最小化，即让自编码器学会重构正常音频
            # 这里我们用重构误差的均值作为损失（等同于MSE）
            loss = recon_error.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}")

    # 保存模型权重
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"PyTorch 模型已保存至 {args.save_path}")

def main():
    parser = argparse.ArgumentParser(description="训练异常检测模型")
    parser.add_argument('--backend', type=str, choices=['sklearn', 'pytorch'], default='sklearn',
                        help='选择训练框架')
    parser.add_argument('--model_type', type=str, default='svm',
                        help='sklearn 模型类型: svm 或 iforest')
    parser.add_argument('--train_dir', type=str, required=True,
                        help='训练数据目录（仅包含正常音频）')
    parser.add_argument('--save_path', type=str, required=True,
                        help='模型保存路径（.pkl 或 .pt）')
    parser.add_argument('--nu', type=float, default=0.1,
                        help='One-Class SVM nu 参数')
    parser.add_argument('--contamination', type=float, default=0.1,
                        help='Isolation Forest contamination 参数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='PyTorch 训练批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='PyTorch 训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='PyTorch 学习率')
    args = parser.parse_args()

    if args.backend == 'sklearn':
        train_sklearn(args)
    else:
        train_pytorch(args)

if __name__ == "__main__":
    main()


