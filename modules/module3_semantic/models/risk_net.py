# c:\Users\Leo\Desktop\gemini\models\risk_net.py
import torch
import torch.nn as nn

class RiskNet(nn.Module):
    def __init__(self, text_dim=384, context_dim=16, hidden_dim=256):
        super(RiskNet, self).__init__()
        
        # 支路 1：语义分支 (处理指令)
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 支路 2：环境分支 (处理车速、天气、行人等物理事实)
        self.context_branch = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU()
        )
        
        # 支路 3：历史经验分支 (处理向量库 VDB 返回的参考分)
        self.vdb_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )
        
        # 最终融合层 (128 + 32 + 16 = 176)
        self.fusion_net = nn.Sequential(
            nn.Linear(208, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # 输出 0-1 之间的风险概率
        )

    def forward(self, text_emb, context_vec, vdb_score):
        t_feat = self.text_branch(text_emb)
        c_feat = self.context_branch(context_vec)
        v_feat = self.vdb_branch(vdb_score)
        
        # 将三路特征在维度上进行拼接 (Concatenate)
        combined = torch.cat((t_feat, c_feat, v_feat), dim=1)
        return self.fusion_net(combined)