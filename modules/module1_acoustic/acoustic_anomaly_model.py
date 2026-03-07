"""
弃用

acoustic_anomaly_model.py
异常检测模型封装，支持：
- scikit-learn 模型 (One-Class SVM, Isolation Forest) 通过 joblib 加载
- PyTorch 自编码器模型 (参考 DCASE 基线)，通过重构误差作为异常分数
"""
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T

class AEDCASEBaseline(nn.Module):
    """
    自编码器模型，参考 DCASE 2020 挑战赛基线
    输入：原始波形 (batch, samples) 16kHz
    输出：重构误差 (MSE)
    """
    def __init__(self, frames=5, n_mels=128, hop_length=512, sample_rate=16000):
        super().__init__()
        self.frames = frames
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        # Mel 谱图变换
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
            mel_scale="htk",
        )

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(640, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Linear(128, 8),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 640)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化权重，偏置置零（模仿 Keras）"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        x: (batch, samples) 原始波形
        返回重构误差 (batch,)
        """
        # 计算 Mel 谱图并转为对数刻度
        mel = self.mel_spec(x)  # (batch, n_mels, time)
        mel = 10 * torch.log10(mel + 1e-8)

        batch_size, n_mels, time_steps = mel.shape
        # 构建特征向量：滑动窗口拼接 frames 帧
        vector_dim = self.frames * n_mels
        num_vectors = time_steps - self.frames + 1
        if num_vectors <= 0:
            # 音频太短，填充
            pad = self.frames - time_steps
            mel = torch.nn.functional.pad(mel, (0, pad))
            num_vectors = 1

        feature_vectors = []
        for i in range(num_vectors):
            segment = mel[:, :, i:i+self.frames]  # (batch, n_mels, frames)
            # 展平为 (batch, n_mels * frames)
            feature_vectors.append(segment.reshape(batch_size, -1))
        # 堆叠所有时间窗口的特征
        features = torch.stack(feature_vectors, dim=1)  # (batch, num_vectors, vector_dim)

        # 编码-解码
        encoded = self.encoder(features)          # (batch, num_vectors, 128)
        bottleneck = self.bottleneck(encoded)     # (batch, num_vectors, 8)
        decoded = self.decoder(bottleneck)        # (batch, num_vectors, 640)

        # 计算每个窗口的重构误差（MSE）
        loss_per_vector = torch.mean((features - decoded) ** 2, dim=-1)  # (batch, num_vectors)
        # 取所有窗口的平均作为最终误差
        recon_error = torch.mean(loss_per_vector, dim=1)  # (batch,)
        return recon_error


class AnomalyModel:
    """
    统一异常检测模型接口
    根据 model_path 后缀自动选择：
        - .pkl  -> 加载 scikit-learn 模型（joblib）
        - .pt / .pth -> 加载 PyTorch 自编码器模型
    """
    def __init__(self, model_path, device='cpu'):
        self.model_path = model_path
        self.device = torch.device(device)
        self.model = None
        self.model_type = None
        self._load_model()

    def _load_model(self):
        """根据文件扩展名加载模型"""
        ext = os.path.splitext(self.model_path)[-1].lower()
        if ext == '.pkl':
            # scikit-learn 模型
            self.model = joblib.load(self.model_path)
            self.model_type = 'sklearn'
        elif ext in ['.pt', '.pth']:
            # PyTorch 模型
            self.model = AEDCASEBaseline()
            state_dict = torch.load(self.model_path, map_location=self.device)
            # 如果保存的是完整模型而非 state_dict，需处理
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            self.model_type = 'pytorch'
        else:
            raise ValueError(f"Unsupported model file extension: {ext}")

    def predict_risk(self, audio: np.ndarray) -> float:
        """
        输入：预处理后的音频 (numpy array, 16kHz)
        返回：原始异常分数（越大越异常）
        """
        if self.model_type == 'sklearn':
            # 对于 sklearn 模型，需要先提取特征（此处由外部传入特征？）
            # 但当前设计是 model 只接收特征，所以这里假设 audio 已经是特征向量
            # 为保持兼容，我们要求外部传入特征，而不是原始音频
            # 注意：在 detector.py 中，调用 predict_risk 前会先提取特征
            # 所以此处 audio 应为特征向量 (numpy array)
            if audio.ndim == 1:
                audio = audio.reshape(1, -1)
            if hasattr(self.model, 'decision_function'):
                score = self.model.decision_function(audio)[0]
            else:
                # 对于没有 decision_function 的模型（如 IsolationForest 有 score_samples）
                score = self.model.score_samples(audio)[0]
                # 将分数映射为异常程度：score_samples 值越大越正常，所以取负
                score = -score
            return float(score)

        elif self.model_type == 'pytorch':
            # 输入应为原始波形 (numpy array)
            # 转换为 tensor，添加 batch 维度
            audio_tensor = torch.from_numpy(audio).float().to(self.device)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # (1, samples)
            with torch.no_grad():
                recon_error = self.model(audio_tensor)  # (batch,)
            # 返回第一个样本的误差
            return recon_error[0].item()

        else:
            raise RuntimeError("Model not loaded properly")

