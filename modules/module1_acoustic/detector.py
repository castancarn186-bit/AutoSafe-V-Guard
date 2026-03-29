"""
modules/module1_acoustic/detector.py
多模型融合声学检测器（LA + PA + 对抗检测）
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from core.base_module import BaseDetector
from core.protocol import SystemContext, RiskReport

# ---------- 自编码器定义（对抗检测）----------
class AudioAutoencoder(nn.Module):
    def __init__(self, input_dim=64600, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 5, 2, 2), nn.ReLU(),
            nn.Conv1d(16, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, 2, 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, input_dim), nn.Tanh()
        )
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z).view(x.size(0), 1, -1)
        return recon

# ---------- AASIST 模型导入 ----------
from .aasist_model import Model as AASIST

class AcousticDetector(BaseDetector):
    def __init__(self, config=None):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.la_model = None
        self.pa_model = None
        self.autoencoder = None
        self.mel_transform = None
        self.log_offset = 1e-6

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
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=160,
            win_length=400, n_mels=80, f_min=0, f_max=8000, power=2.0
        )
        # LA
        if os.path.exists(self.la_model_path):
            cfg = self._get_aasist_config()
            self.la_model = AASIST(cfg).to(self.device)
            self.la_model.load_state_dict(torch.load(self.la_model_path, map_location=self.device), strict=True)
            self.la_model.eval()
            self.logger.info(f"LA model loaded from {self.la_model_path}")
        else:
            self.logger.warning("LA model not found")
        # PA
        if os.path.exists(self.pa_model_path):
            cfg = self._get_aasist_config()
            self.pa_model = AASIST(cfg).to(self.device)
            self.pa_model.load_state_dict(torch.load(self.pa_model_path, map_location=self.device), strict=True)
            self.pa_model.eval()
            self.logger.info(f"PA model loaded from {self.pa_model_path}")
        else:
            self.logger.warning("PA model not found")
        # AE
        if os.path.exists(self.ae_model_path):
            self.autoencoder = AudioAutoencoder(input_dim=self.expected_length).to(self.device)
            self.autoencoder.load_state_dict(torch.load(self.ae_model_path, map_location=self.device), strict=True)
            self.autoencoder.eval()
            self.logger.info(f"Autoencoder loaded from {self.ae_model_path}")
        else:
            self.logger.warning("Autoencoder not found")

    def _get_aasist_config(self):
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
        if len(audio) > self.expected_length:
            audio = audio[:self.expected_length]
        else:
            pad = self.expected_length - len(audio)
            audio = np.pad(audio, (0, pad), 'constant')
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        mel = self.mel_transform(audio_tensor)
        log_mel = torch.log(mel + self.log_offset)
        return log_mel.unsqueeze(0)

    def _preprocess_for_ae(self, audio):
        if len(audio) > self.expected_length:
            audio = audio[:self.expected_length]
        else:
            pad = self.expected_length - len(audio)
            audio = np.pad(audio, (0, pad), 'constant')
        max_val = np.max(np.abs(audio))
        if max_val > 1e-8:
            audio = audio / max_val
        return torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(self.device)

    def _predict_aasist(self, model, audio):
        if model is None:
            return 0.0
        inp = self._preprocess_for_aasist(audio)
        with torch.no_grad():
            _, out = model(inp)
            prob = torch.softmax(out, dim=-1)[0, 1].item()
        return prob

    def _predict_ae(self, audio):
        if self.autoencoder is None:
            return 0.0
        x = self._preprocess_for_ae(audio)
        with torch.no_grad():
            recon = self.autoencoder(x)
            mse = torch.mean((recon - x) ** 2).item()
        risk = 1.0 / (1.0 + np.exp(-10 * (mse - self.ae_threshold)))
        return risk

    def detect(self, ctx: SystemContext) -> RiskReport:
        audio = ctx.audio_frame
        if audio is None or len(audio) == 0:
            return RiskReport(0.0, "PASS", "No audio", {})

        try:
            la_risk = self._predict_aasist(self.la_model, audio)
            pa_risk = self._predict_aasist(self.pa_model, audio)
            ae_risk = self._predict_ae(audio)

            final_risk = (self.fusion_weights.get('la',0)*la_risk +
                          self.fusion_weights.get('pa',0)*pa_risk +
                          self.fusion_weights.get('ae',0)*ae_risk)
            final_risk = np.clip(final_risk, 0.0, 1.0)

            suggestion = self._get_suggestion(final_risk)
            evidence = {"la": la_risk, "pa": pa_risk, "ae": ae_risk, "final": final_risk}
            return RiskReport(final_risk, suggestion, "Multi-model fusion", evidence)
        except Exception as e:
            self.logger.exception("Detection error")
            return RiskReport(0.5, "PASS", f"Error: {e}", {})

    def _get_suggestion(self, risk):
        if risk < 0.3: return "PASS"
        elif risk < 0.6: return "CONFIRM"
        else: return "REJECT"
        else:
            return "REJECT"
