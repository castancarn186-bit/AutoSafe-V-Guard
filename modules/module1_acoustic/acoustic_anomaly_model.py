"""
异常检测模型封装
支持 One-Class SVM 和 Isolation Forest
"""
import joblib
import numpy as np

class AnomalyModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.model_type = type(self.model).__name__

    def predict_risk(self, features):
        """
        返回原始异常分数（数值越小越异常）
        对于 One-Class SVM，返回 decision_function 值（正常为正值）
        对于 Isolation Forest，返回 decision_function 值（正常为正值）
        """
        if hasattr(self.model, 'decision_function'):
            score = self.model.decision_function(features.reshape(1, -1))[0]
        else:
            # 如果模型没有 decision_function，使用预测标签（但可能不连续）
            score = -self.model.score_samples(features.reshape(1, -1))[0]  # 示例，需调整
        return score