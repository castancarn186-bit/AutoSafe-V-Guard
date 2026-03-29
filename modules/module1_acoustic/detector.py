"""
modules/module1_acoustic/detector.py
声学物理层检测器（多模型融合：LA + PA）
支持加载两个 AASIST 模型（LA 和 PA），输出融合后的风险分数。
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from .aasist_model import Model as AASIST   # 确保 aasist_model.py 中存在 Model 类

class AcousticDetector:
    """
    声学检测器，不依赖 core 模块，可独立使用。
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.la_model = None
        self.pa_model = None
        self.mel_transform = None
        self.log_offset = 1e-6

        # 模型路径（默认从本文件所在目录的 models/ 下加载）
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, 'models')
        self.la_model_path = self.config.get('la_model_path', os.path.join(models_dir, 'aasist_la.pth'))
        self.pa_model_path = self.config.get('pa_model_path', os.path.join(models_dir, 'aasist_pa.pth'))

        # 参数配置
        self.expected_length = self.config.get('expected_length', 64600)
        self.fusion_weights = self.config.get('fusion_weights', {'la': 0.5, 'pa': 0.5})
        self.thresholds = self.config.get('thresholds', {'PASS': 0.3, 'CONFIRM': 0.6})

    def setup(self):
        """加载模型并初始化预处理"""
        # 初始化 Mel 谱图转换器
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=160,
            win_length=400,
            n_mels=80,
            f_min=0,
            f_max=8000,
            power=2.0
        )

        # 加载 LA 模型
        if os.path.exists(self.la_model_path):
            model_config = self._get_aasist_config()
            self.la_model = AASIST(model_config).to(self.device)
            state_dict = torch.load(self.la_model_path, map_location=self.device)
            self.la_model.load_state_dict(state_dict, strict=True)
            self.la_model.eval()
            print(f"[Acoustic] LA model loaded from {self.la_model_path}")
        else:
            print(f"[Acoustic] Warning: LA model not found at {self.la_model_path}")

        # 加载 PA 模型
        if os.path.exists(self.pa_model_path):
            model_config = self._get_aasist_config()
            self.pa_model = AASIST(model_config).to(self.device)
            state_dict = torch.load(self.pa_model_path, map_location=self.device)
            self.pa_model.load_state_dict(state_dict, strict=True)
            self.pa_model.eval()
            print(f"[Acoustic] PA model loaded from {self.pa_model_path}")
        else:
            print(f"[Acoustic] Warning: PA model not found at {self.pa_model_path}")

    def _get_aasist_config(self):
        """返回 AASIST 模型配置（与训练时一致）"""
        return {
            "architecture": "AASIST",
            "nb_samp": self.expected_length,
            "first_conv": 128,
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.7, 0.5, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0]
        }

    def _preprocess_for_aasist(self, audio):
        """将原始波形 (numpy array, 16kHz) 转换为 Log-Mel 谱图，形状 (1,1,80,T)"""
        # 长度调整
        if len(audio) > self.expected_length:
            audio = audio[:self.expected_length]
        else:
            pad = self.expected_length - len(audio)
            audio = np.pad(audio, (0, pad), 'constant')
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        mel = self.mel_transform(audio_tensor)
        log_mel = torch.log(mel + self.log_offset)
        return log_mel.unsqueeze(0)   # (1,1,80,T)

    def _predict_aasist(self, model, audio):
        """返回模型预测的 spoof 概率（风险分数）"""
        if model is None:
            return 0.0
        input_tensor = self._preprocess_for_aasist(audio)
        with torch.no_grad():
            _, out = model(input_tensor)   # out shape: (1,2)
            prob = torch.softmax(out, dim=-1)[0, 1].item()
        return prob

    def detect(self, audio):
        """
        输入：原始音频波形 (numpy array, 16kHz, 单声道)
        输出：字典包含 risk_score, suggestion, evidence
        """
        if audio is None or len(audio) == 0:
            return {"risk_score": 0.0, "suggestion": "PASS", "reason": "No audio", "evidence": {}}

        try:
            la_risk = self._predict_aasist(self.la_model, audio)
            pa_risk = self._predict_aasist(self.pa_model, audio)

            # 加权融合
            final_risk = (self.fusion_weights.get('la', 0.0) * la_risk +
                          self.fusion_weights.get('pa', 0.0) * pa_risk)
            final_risk = np.clip(final_risk, 0.0, 1.0)

            # 建议
            if final_risk < self.thresholds['PASS']:
                suggestion = "PASS"
            elif final_risk < self.thresholds['CONFIRM']:
                suggestion = "CONFIRM"
            else:
                suggestion = "REJECT"

            evidence = {
                "la_risk": la_risk,
                "pa_risk": pa_risk,
                "final_risk": final_risk,
                "fusion_weights": self.fusion_weights
            }
            return {
                "risk_score": final_risk,
                "suggestion": suggestion,
                "reason": "Multi-model acoustic detection",
                "evidence": evidence
            }
        except Exception as e:
            print(f"[Acoustic] Detection error: {e}")
            return {"risk_score": 0.5, "suggestion": "PASS", "reason": f"Error: {e}", "evidence": {}}
