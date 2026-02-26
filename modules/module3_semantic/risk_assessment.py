import torch
import numpy as np
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import os

class RiskAssessmentModel(nn.Module):
    def __init__(self, input_dim):
        super(RiskAssessmentModel, self).__init__()
        # 简单的多层感知机 (MLP)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1), # 输出风险分数 0-1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

class RiskManager:
    def __init__(self, input_dim):
        self.model = RiskAssessmentModel(input_dim)
        self.risk_matrix = None
        # 实际项目中这里应该 load_pretrained_weights
        # self.model.load_state_dict(torch.load('models/risk_model.pth'))
        print("风险模型初始化完成 (随机权重用于演示)。")

    def prepare_input(self, command_embedding, state_vector):
        """拼接语义向量和状态向量"""
        # command_embedding: [384] (假设 sentence-transformers 维度)
        # state_vector: [6]
        # 为了演示，我们压缩 command_embedding 到 10 维，或者只取 state_vector 做演示
        # 实际应拼接：combined = np.concatenate([command_embedding[:10], state_vector])
        combined = np.concatenate([command_embedding[:10], state_vector]) 
        return torch.FloatTensor(combined).unsqueeze(0)

    def evaluate(self, input_tensor):
        """模型推理"""
        with torch.no_grad():
            risk_score = self.model(input_tensor).item()
        return risk_score

    def generate_risk_matrix(self, risk_score, save_path='outputs/risk_matrix.png'):
        """构建经典的风险矩阵 (可能性 x 严重性)"""
        # 【中文字体设置】
        import matplotlib.font_manager as fm
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 定义矩阵标签 (中文)
        likelihood = ["低", "中", "高"]
        severity = ["低", "中", "高"]
        
        # 根据 score 简单映射位置
        if risk_score < 0.3:
            li_idx, se_idx = 0, 0 
            decision = "允许执行"
            decision_en = "ALLOW"
            color = "green"
        elif risk_score < 0.7:
            li_idx, se_idx = 1, 1 
            decision = "警告提示"
            decision_en = "WARNING"
            color = "orange"
        else:
            li_idx, se_idx = 2, 2 
            decision = "拒绝执行"
            decision_en = "DENY"
            color = "red"

        # 构建数据矩阵
        data = np.zeros((3, 3))
        data[li_idx, se_idx] = risk_score 

        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(data, annot=True, fmt=".2f", cmap="RdYlGn_r", 
                    xticklabels=severity, yticklabels=likelihood,
                    linewidths=2, linecolor='white', square=True,
                    annot_kws={'size': 14, 'weight': 'bold'})
        
        # 设置标题 (中文)
        plt.title(f"安全风险评估矩阵\n决策：{decision} ({decision_en}) | 风险分数：{risk_score:.2f}", 
                 fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("严重性 (Severity)", fontsize=12)
        plt.ylabel("可能性 (Likelihood)", fontsize=12)
        
        # 设置坐标轴标签字体大小
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        
        # 添加颜色说明
        cbar = ax.collections[0].colorbar
        cbar.set_label('风险评分', fontsize=11, rotation=270, labelpad=15)
        
        # 在图下方添加决策说明
        plt.figtext(0.5, 0.01, f"系统决策：{decision} | 风险等级：{['低', '中', '高'][min(2, int(risk_score*3))]}风险", 
                   ha='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 风险矩阵图已保存至：{save_path}")
        plt.show()
        
        return decision, risk_score