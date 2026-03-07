"""
modules/module_a_acoustic/detector.py
声学物理层检测器（基于 RawNet2 端到端深度学习模型）
继承 core.base_module.BaseDetector，实现 setup() 和 detect(ctx)
使用简化版 AudioPreprocessor 进行归一化和长度调整。
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from core.base_module import BaseDetector
from core.protocol import SystemContext, RiskReport
from .audio_preprocessor import AudioPreprocessor
from .rawnet2_model import RawNet2


class AcousticDetector(BaseDetector):
    """
    使用 RawNet2 预训练模型检测声学欺骗攻击（重放、合成等）。
    """
    def __init__(self, config=None):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.preprocessor = None
        # 模型期望的输入长度（样本点数），根据 RawNet2 默认设置
        self.expected_length = self.config.get('expected_length', 64600)
        # 风险阈值（可配置）
        self.thresholds = self.config.get('thresholds', {'PASS': 0.3, 'CONFIRM': 0.6})

    def setup(self):
        """加载 RawNet2 预训练权重并初始化预处理器"""
        # 初始化预处理器（简单模式，仅归一化和长度调整）
        self.preprocessor = AudioPreprocessor(target_sr=16000)

        # 模型路径
        model_path = self.config.get(
            'model_path',
            os.path.join(os.path.dirname(__file__), 'models', 'rawnet2.pth')
        )
        if not os.path.exists(model_path):
            self.logger.error(f"RawNet2 model file not found: {model_path}")
            raise FileNotFoundError(f"RawNet2 model missing: {model_path}")

        # 初始化模型并加载权重
        self.model = RawNet2(pretrained_path=model_path)
        self.model.to(self.device)
        self.model.eval()
        self.logger.info(f"RawNet2 model loaded from {model_path} to {self.device}")

    def detect(self, ctx: SystemContext) -> RiskReport:
        """
        核心检测逻辑。
        ctx.audio_frame: 原始音频块 (numpy array, 16kHz 单声道)
        """
        audio = ctx.audio_frame
        if audio is None or len(audio) == 0:
            return RiskReport(
                risk_score=0.0,
                suggestion="PASS",
                reason="No audio input",
                evidence={}
            )

        try:
            # 1. 预处理：归一化、长度调整
            processed_audio = self.preprocessor.process(
                audio,
                orig_sr=16000,               # 假设输入音频已是16kHz，但保留参数以备后续
                target_length=self.expected_length
            )

            # 2. 转换为 tensor 并添加 batch 和 channel 维度
            tensor = torch.from_numpy(processed_audio).float().to(self.device)
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)

            # 3. 模型推理
            with torch.no_grad():
                logits = self.model(tensor)           # (1, 2)
                probs = F.softmax(logits, dim=-1)     # (1, 2)
                # 假设类别索引：0=bonafide, 1=spoof
                risk_score = probs[0, 1].item()       # spoof 概率作为风险分数

            # 4. 生成建议
            suggestion = self._get_suggestion(risk_score)

            # 5. 构建证据（可用于调试和可视化）
            evidence = {
                "spoof_prob": risk_score,
                "bonafide_prob": probs[0, 0].item(),
                "logits": logits.cpu().numpy().tolist(),
                "input_length": len(audio)
            }

            return RiskReport(
                risk_score=risk_score,
                suggestion=suggestion,
                reason="Acoustic spoofing detection with RawNet2",
                evidence=evidence
            )

        except Exception as e:
            self.logger.exception("Error in acoustic detection")
            # 发生异常时返回保守值（0.5）和 PASS 建议，避免系统崩溃
            return RiskReport(
                risk_score=0.5,
                suggestion="PASS",
                reason=f"Acoustic detection error: {str(e)}",
                evidence={}
            )

    def _get_suggestion(self, risk_score: float) -> str:
        """根据风险分数返回建议"""
        if risk_score < self.thresholds['PASS']:
            return "PASS"
        elif risk_score < self.thresholds['CONFIRM']:
            return "CONFIRM"
        else:
            return "REJECT"
