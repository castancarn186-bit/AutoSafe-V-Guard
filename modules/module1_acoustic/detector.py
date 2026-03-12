"""
modules/module_a_acoustic/detector.py
声学物理层检测器（基于 AASIST 模型）
继承 core.base_module.BaseDetector，实现 setup() 和 detect(ctx)
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from core.base_module import BaseDetector
from core.protocol import SystemContext, RiskReport

# 从同目录的 aasist_model.py 导入 Model 类并重命名为 AASIST
from .aasist_model import Model as AASIST


class AcousticDetector(BaseDetector):
    """
    使用 AASIST 预训练模型检测声学欺骗攻击（AI合成语音、重放等）。
    """
    def __init__(self, config=None):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.mel_transform = None
        # AASIST 期望的输入长度（4秒 @16kHz = 64600 样本）
        self.expected_length = self.config.get('expected_length', 64600)
        # 风险阈值
        self.thresholds = self.config.get('thresholds', {'PASS': 0.3, 'CONFIRM': 0.6})

    def setup(self):
        """加载 AASIST 预训练模型并初始化谱图转换器"""
        # 1. 初始化 Mel 谱图转换器（参数与 AASIST 官方一致）
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=160,          # 10ms 帧移
            win_length=400,           # 25ms 窗长
            n_mels=80,
            f_min=0,
            f_max=8000,
            power=2.0
        )
        self.log_offset = 1e-6

        # 2. 定义模型配置（请从官方仓库 main.py 中获取准确值）
        d_args = {
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.5, 0.5, 0.5],
            "temperatures": [2, 2, 100],
            "first_conv": 1024,
        }

        # 3. 实例化模型
        self.model = AASIST(d_args)

        # 4. 加载预训练权重
        model_path = self.config.get(
            'model_path',
            os.path.join(os.path.dirname(__file__), 'models', 'aasist.pth')
        )
        if not os.path.exists(model_path):
            self.logger.error(f"AASIST model file not found: {model_path}")
            raise FileNotFoundError(f"AASIST model missing: {model_path}")

        state_dict = torch.load(model_path, map_location='cpu')
        # 处理权重字典可能包含的键名
        if 'model' in state_dict:
            state_dict = state_dict['model']
        # 移除 'module.' 前缀（如果权重是从 DataParallel 保存的）
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.logger.info(f"AASIST model loaded from {model_path} to {self.device}")

    def _preprocess(self, audio):
        """
        将原始音频转换为 AASIST 输入格式：Log-Mel 谱图
        输入: audio (numpy array, 16kHz)
        输出: tensor (1, 1, 80, T)  # batch=1, channel=1, freq=80, time=T
        """
        # 长度调整
        if len(audio) > self.expected_length:
            audio = audio[:self.expected_length]
        elif len(audio) < self.expected_length:
            pad = self.expected_length - len(audio)
            audio = np.pad(audio, (0, pad), 'constant')

        # 转为 tensor 并添加 batch 维度
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # (1, samples)
        audio_tensor = audio_tensor.to(self.device)

        # 计算 Mel 谱图
        with torch.no_grad():
            mel = self.mel_transform(audio_tensor)  # (1, n_mels, time)
            log_mel = torch.log(mel + self.log_offset)
            # 添加 channel 维度 (1, 1, n_mels, time)
            log_mel = log_mel.unsqueeze(1)
        return log_mel

    def detect(self, ctx: SystemContext) -> RiskReport:
        audio = ctx.audio_frame
        if audio is None or len(audio) == 0:
            return RiskReport(
                risk_score=0.0,
                suggestion="PASS",
                reason="No audio input",
                evidence={}
            )

        try:
            # 1. 预处理为 Log-Mel 谱图
            input_tensor = self._preprocess(audio)  # (1,1,80,T)

            # 2. 模型推理
            with torch.no_grad():
                # AASIST 的 forward 返回 (last_hidden, output)，我们只需要 output
                _, output = self.model(input_tensor)  # output shape: (1, 2)
                probs = F.softmax(output, dim=-1)
                risk_score = probs[0, 1].item()     # spoof 概率

            # 3. 生成建议
            suggestion = self._get_suggestion(risk_score)

            # 4. 证据
            evidence = {
                "spoof_prob": risk_score,
                "bonafide_prob": probs[0, 0].item(),
                "input_length": len(audio),
                "spectrogram_shape": list(input_tensor.shape)
            }

            return RiskReport(
                risk_score=risk_score,
                suggestion=suggestion,
                reason="Acoustic spoofing detection with AASIST",
                evidence=evidence
            )

        except Exception as e:
            self.logger.exception("Error in acoustic detection")
            return RiskReport(
                risk_score=0.5,
                suggestion="PASS",
                reason=f"Acoustic detection error: {str(e)}",
                evidence={}
            )

    def _get_suggestion(self, risk_score):
        if risk_score < self.thresholds['PASS']:
            return "PASS"
        elif risk_score < self.thresholds['CONFIRM']:
            return "CONFIRM"
        else:
            return "REJECT"
