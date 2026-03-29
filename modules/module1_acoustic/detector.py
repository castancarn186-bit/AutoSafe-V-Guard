"""
modules/module1_acoustic/detector.py
声学物理层检测器（多模型融合：LA + PA + 对抗检测）
继承 core.base_module.BaseDetector，实现 setup() 和 detect(ctx)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from core.base_module import BaseDetector
from core.protocol import SystemContext, RiskReport

# ==================== 自编码器定义（对抗检测） ====================
class AudioAutoencoder(nn.Module):
    """轻量级自编码器，用于对抗扰动检测"""
    def __init__(self, input_dim=64600, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z).view(x.size(0), 1, -1)
        return recon

# ==================== AASIST 模型导入 ====================
from .aasist_model import Model as AASIST

# ==================== 主检测器类 ====================
class AcousticDetector(BaseDetector):
    """
    多模型融合声学检测器：
    - LA 模型（反欺骗，识别 TTS/VC 攻击）
    - PA 模型（反重放，识别录音回放攻击）
    - 自编码器（对抗检测，识别对抗性噪声）
    """
    def __init__(self, config=None):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.la_model = None
        self.pa_model = None
        self.autoencoder = None
        self.mel_transform = None
        self.log_offset = 1e-6

        # 模型文件默认路径（相对于本文件所在目录的 models 子目录）
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, 'models')

        self.la_model_path = self.config.get('la_model_path', os.path.join(models_dir, 'aasist_la.pth'))
        self.pa_model_path = self.config.get('pa_model_path', os.path.join(models_dir, 'aasist_pa.pth'))
        self.ae_model_path = self.config.get('ae_model_path', os.path.join(models_dir, 'autoencoder.pth'))

        self.ae_threshold = self.config.get('ae_threshold', 0.05)
        self.expected_length = self.config.get('expected_length', 64600)
        self.fusion_weights = self.config.get('fusion_weights', {'la': 0.5, 'pa': 0.5, 'ae': 0.0})
        self.thresholds = self.config.get('thresholds', {'PASS': 0.3, 'CONFIRM': 0.6})

    def setup(self):
        """加载所有启用的模型"""
        # 初始化公共的 Mel 谱图转换器（AASIST 需要）
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

        # 1. LA 模型
        if os.path.exists(self.la_model_path):
            model_config = self._get_aasist_config()
            self.la_model = AASIST(model_config).to(self.device)
            state_dict = torch.load(self.la_model_path, map_location=self.device)
            self.la_model.load_state_dict(state_dict, strict=True)
            self.la_model.eval()
            self.logger.info(f"LA model loaded from {self.la_model_path}")
        else:
            self.logger.warning(f"LA model not found at {self.la_model_path}")

        # 2. PA 模型
        if os.path.exists(self.pa_model_path):
            model_config = self._get_aasist_config()
            self.pa_model = AASIST(model_config).to(self.device)
            state_dict = torch.load(self.pa_model_path, map_location=self.device)
            self.pa_model.load_state_dict(state_dict, strict=True)
            self.pa_model.eval()
            self.logger.info(f"PA model loaded from {self.pa_model_path}")
        else:
            self.logger.warning(f"PA model not found at {self.pa_model_path}")

        # 3. 自编码器（对抗检测）
        if os.path.exists(self.ae_model_path):
            self.autoencoder = AudioAutoencoder(input_dim=self.expected_length).to(self.device)
            state_dict = torch.load(self.ae_model_path, map_location=self.device)
            self.autoencoder.load_state_dict(state_dict, strict=True)
            self.autoencoder.eval()
            self.logger.info(f"Autoencoder loaded from {self.ae_model_path}")
        else:
            self.logger.warning(f"Autoencoder not found at {self.ae_model_path}")

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
        """
        将原始波形转换为 AASIST 输入：Log-Mel 谱图
        输出形状: (1, 1, 80, T)
        """
        if len(audio) > self.expected_length:
            audio = audio[:self.expected_length]
        else:
            pad = self.expected_length - len(audio)
            audio = np.pad(audio, (0, pad), 'constant')
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        mel = self.mel_transform(audio_tensor)
        log_mel = torch.log(mel + self.log_offset)
        return log_mel.unsqueeze(0)  # (1,1,80,T)

    def _preprocess_for_ae(self, audio):
        """
        自编码器预处理：归一化 + 固定长度
        输出形状: (1, 1, samples)
        """
        if len(audio) > self.expected_length:
            audio = audio[:self.expected_length]
        else:
            pad = self.expected_length - len(audio)
            audio = np.pad(audio, (0, pad), 'constant')
        max_val = np.max(np.abs(audio))
        if max_val > 1e-8:
            audio = audio / max_val
        tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor

    def _predict_aasist(self, model, audio):
        """使用 AASIST 模型预测 spoof 概率（风险分数）"""
        if model is None:
            return 0.0
        input_tensor = self._preprocess_for_aasist(audio)
        with torch.no_grad():
            _, out = model(input_tensor)   # out shape: (1,2)
            prob = torch.softmax(out, dim=-1)[0, 1].item()
        return prob

    def _predict_ae(self, audio):
        """自编码器重构误差 -> 风险分数 [0,1]"""
        if self.autoencoder is None:
            return 0.0
        x = self._preprocess_for_ae(audio)
        with torch.no_grad():
            recon = self.autoencoder(x)
            mse = torch.mean((recon - x) ** 2).item()
        # 将误差映射到 [0,1]，阈值 self.ae_threshold 为中心点
        risk = 1.0 / (1.0 + np.exp(-10 * (mse - self.ae_threshold)))
        return risk

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
            # 计算各模型风险
            la_risk = self._predict_aasist(self.la_model, audio)
            pa_risk = self._predict_aasist(self.pa_model, audio)
            ae_risk = self._predict_ae(audio)

            # 加权融合
            final_risk = (self.fusion_weights.get('la', 0.0) * la_risk +
                          self.fusion_weights.get('pa', 0.0) * pa_risk +
                          self.fusion_weights.get('ae', 0.0) * ae_risk)
            final_risk = np.clip(final_risk, 0.0, 1.0)

            suggestion = self._get_suggestion(final_risk)
            evidence = {
                "la_risk": la_risk,
                "pa_risk": pa_risk,
                "ae_risk": ae_risk,
                "final_risk": final_risk,
                "fusion_weights": self.fusion_weights
            }

            return RiskReport(
                risk_score=final_risk,
                suggestion=suggestion,
                reason="Multi-model acoustic detection",
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

    def _get_suggestion(self, risk_score: float) -> str:
        if risk_score < self.thresholds['PASS']:
            return "PASS"
        elif risk_score < self.thresholds['CONFIRM']:
            return "CONFIRM"
        else:
            return "REJECT"
            return "CONFIRM"
        else:
            return "REJECT"
