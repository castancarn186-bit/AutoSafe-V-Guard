"""
弃用

audio_preprocessor.py
简化版预处理：仅包含归一化和长度调整，用于 RawNet2 模型。
"""
import numpy as np
import librosa

class AudioPreprocessor:
    """
    简单音频预处理器，仅做：
    - 重采样（可选）
    - 归一化到 [-1, 1]
    - 固定长度截断/填充
    """
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr

    def resample(self, audio, orig_sr):
        """重采样到目标采样率"""
        if orig_sr == self.target_sr:
            return audio
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=self.target_sr)

    def normalize(self, audio):
        """归一化到 [-1, 1]"""
        max_val = np.max(np.abs(audio))
        if max_val > 1e-8:
            return audio / max_val
        return audio

    def adjust_length(self, audio, target_length):
        """将音频截断或填充到指定长度"""
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            pad = target_length - len(audio)
            audio = np.pad(audio, (0, pad), 'constant')
        return audio

    def process(self, audio, orig_sr=None, target_length=None):
        """
        统一预处理入口
        :param audio: 原始音频 (numpy array)
        :param orig_sr: 原始采样率（如为 None 则假设已是 target_sr）
        :param target_length: 目标长度（必须提供）
        :return: 处理后的音频 (numpy array)
        """
        if target_length is None:
            raise ValueError("target_length must be provided for simple preprocessing")

        if orig_sr is not None:
            audio = self.resample(audio, orig_sr)

        audio = self.normalize(audio)
        audio = self.adjust_length(audio, target_length)

        return audio

