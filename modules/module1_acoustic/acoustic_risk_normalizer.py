"""
弃用

风险归一化模块
将异常检测模型输出的原始分数映射到 [0,1] 区间
支持 sigmoid 映射或线性映射
"""
import numpy as np

class RiskNormalizer:
    def __init__(self, method='sigmoid', scale=1.0, offset=0.0):
        """
        :param method: 'sigmoid' 或 'linear'
        :param scale:  缩放系数 (用于线性映射: risk = (score - offset) * scale)
        :param offset: 偏移量
        """
        self.method = method
        self.scale = scale
        self.offset = offset

    def normalize(self, raw_score):
        if self.method == 'sigmoid':
            # 假设 raw_score 越大越正常，则风险 = 1 / (1 + exp(raw_score))
            # 若原始模型输出异常值为负，则风险 = 1 / (1 + exp(-raw_score))？需要根据实际情况调整
            # 这里假设 raw_score 为 decision_function，正常为正值，异常为负值，则风险 = 1 / (1 + exp(raw_score))
            risk = 1.0 / (1.0 + np.exp(raw_score))
        elif self.method == 'linear':
            risk = (raw_score - self.offset) * self.scale
            risk = np.clip(risk, 0, 1)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return float(risk)
