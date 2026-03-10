import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from pathlib import Path
import sys

# --- 路径环境配置 ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.module3_semantic.models.risk_net import RiskNet
from modules.module3_semantic.models.embeddings import FeatureExtractor
from modules.module3_semantic.models.vector_db.hnsw_manager import HNSWManager

class HybridVGuardDataset(Dataset):
    def __init__(self, jsonl_path, extractor, vdb_manager):
        print(f"[*] 正在构建融合训练集 (10,000条)，预计算 VDB 参考特征...")
        self.texts, self.contexts, self.vdb_scores, self.labels = [], [], [], []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        raw_texts = [json.loads(line)['text'] for line in lines]
        raw_contexts = [json.loads(line)['context'] for line in lines]
        raw_labels = [json.loads(line)['ground_truth_score'] for line in lines]

        # 1. 批量提取文本 Embedding
        print("[*] 正在执行大规模语义 Embedding 提取...")
        text_embeddings = extractor.encode_text(raw_texts)
        
        # 2. 模拟推理过程，生成 VDB 参考特征
        print("[*] 正在模拟 RAG 检索过程生成训练特征...")
        for i in range(len(lines)):
            ctx_vec = extractor.encode_context(raw_contexts[i])
            
            # 融合向量 (必须与 reasoning/index_builder 保持一致)
            fusion_vec = np.hstack((text_embeddings[i] * 1.0, ctx_vec * 5.0)).astype(np.float32)
            
            # 检索历史相似场景
            match, _ = vdb_manager.search(np.expand_dims(fusion_vec, axis=0), threshold=0.5)
            
            # 这里的 vdb_ref 是让模型学会“看参考分”的关键
            vdb_ref = match['ground_truth_score'] if match else 0.5
            
            self.texts.append(text_embeddings[i])
            self.contexts.append(ctx_vec)
            self.vdb_scores.append([vdb_ref])
            self.labels.append([raw_labels[i]])
            
            if (i + 1) % 2000 == 0:
                print(f"[>] 已完成 {i+1}/10000 条特征对齐...")

        # 转为 Tensor
        self.texts = torch.tensor(np.array(self.texts), dtype=torch.float32)
        self.contexts = torch.tensor(np.array(self.contexts), dtype=torch.float32)
        self.vdb_scores = torch.tensor(np.array(self.vdb_scores), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)

    def __len__(self): return len(self.labels)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.contexts[idx], self.vdb_scores[idx], self.labels[idx]

def run_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_PATH = PROJECT_ROOT / "semantic_safety_dataset_10000.jsonl"
    
    # 确保建好的库能被找到
    extractor = FeatureExtractor()
    vdb_manager = HNSWManager()
    
    dataset = HybridVGuardDataset(DATA_PATH, extractor, vdb_manager)
    loader = DataLoader(dataset, batch_size=128, shuffle=True) # 增加 batch_size 提高速度
    
    # 这里的 RiskNet 必须是支持三路输入的版本！
    model = RiskNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print("\n=== 开始三路特征融合深度训练 (Epochs: 50) ===")
    for epoch in range(50):
        model.train()
        total_loss = 0
        for t, c, v, l in loader:
            t, c, v, l = t.to(device), c.to(device), v.to(device), l.to(device)
            
            optimizer.zero_grad()
            pred = model(t, c, v) # 三路输入：文本, 环境, VDB参考
            loss = criterion(pred, l)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:02d}/50] | MSE Loss: {total_loss/len(loader):.6f}")

    # 保存权重到 models 目录下
    save_path = CURRENT_DIR / "vguard_risknet_best.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n[+] 训练完成！最强大脑已保存至: {save_path}")

if __name__ == "__main__":
    run_train()