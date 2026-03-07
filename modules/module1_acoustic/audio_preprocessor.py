"""
audio_preprocessor.py
音频预处理流水线，支持两种模式：
- 简单模式 (simple): 仅归一化和长度调整（用于 RawNet2）
- 完整模式 (full): 包含谱减法降噪和 VAD（用于传统特征提取）
"""
import numpy as np
import librosa
import scipy.signal

class AudioPreprocessor:
    def __init__(self, target_sr=16000, mode='simple', **kwargs):
        """
        :param target_sr: 目标采样率
        :param mode: 'simple' 或 'full'
        :param kwargs: 其他参数（如 vad_aggressiveness, noise_floor 等，仅在 full 模式下使用）
        """
        self.target_sr = target_sr
        self.mode = mode
        if mode == 'full':
            # 仅在完整模式下初始化 VAD 和谱减法相关参数
            import webrtcvad
            self.vad = webrtcvad.Vad(kwargs.get('vad_aggressiveness', 2))
            self.noise_floor = kwargs.get('noise_floor', 0.001)
            self.pre_emphasis = 0.97
        # 简单模式下无需额外初始化

    def resample(self, audio, orig_sr):
        """重采样到目标采样率（两种模式共用）"""
        if orig_sr == self.target_sr:
            return audio
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=self.target_sr)

    def normalize(self, audio):
        """归一化到 [-1, 1]（两种模式共用）"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio

    def adjust_length(self, audio, target_length):
        """将音频截断或填充到指定长度（简单模式专用）"""
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            pad = target_length - len(audio)
            audio = np.pad(audio, (0, pad), 'constant')
        return audio

    # ---------- 完整模式下的方法 ----------
    def pre_emphasis_filter(self, audio):
        """预加重"""
        return np.append(audio[0], audio[1:] - self.pre_emphasis * audio[:-1])

    def spectral_subtraction(self, audio, n_fft=512, hop_length=128):
        """谱减法降噪"""
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(D)
        phase = np.angle(D)
        noise_est = np.mean(magnitude[:, :5], axis=1, keepdims=True)
        magnitude_clean = magnitude - noise_est
        magnitude_clean = np.maximum(magnitude_clean, self.noise_floor)
        D_clean = magnitude_clean * np.exp(1j * phase)
        return librosa.istft(D_clean, hop_length=hop_length)

    def vad_segment(self, audio, frame_duration_ms=30):
        """VAD 截取有效语音段"""
        audio_int16 = (audio * 32767).astype(np.int16)
        frame_len = int(self.target_sr * frame_duration_ms / 1000)
        if len(audio_int16) < frame_len:
            return None, None
        is_speech = []
        for i in range(0, len(audio_int16) - frame_len + 1, frame_len):
            frame = audio_int16[i:i+frame_len]
            is_speech.append(self.vad.is_speech(frame.tobytes(), self.target_sr))
        # 合并连续语音段（具体实现略，可保留原代码）
        # ...（此处省略详细 VAD 段提取，可保持原有代码）
        return start, end  # 返回有效段起止索引

    # ---------- 统一处理入口 ----------
    def process(self, audio, orig_sr=None, target_length=None):
        """
        统一预处理入口
        :param audio: 原始音频 (numpy array)
        :param orig_sr: 原始采样率（如为 None 则假设已是 target_sr）
        :param target_length: 目标长度（仅 simple 模式需要）
        :return: 处理后的音频
        """
        # 1. 重采样
        if orig_sr is not None:
            audio = self.resample(audio, orig_sr)

        if self.mode == 'simple':
            # 简单模式：仅归一化和长度调整
            audio = self.normalize(audio)
            if target_length is not None:
                audio = self.adjust_length(audio, target_length)
            return audio

        elif self.mode == 'full':
            # 完整模式：预加重、降噪、归一化、VAD
            audio = self.pre_emphasis_filter(audio)
            audio = self.spectral_subtraction(audio)
            audio = self.normalize(audio)
            start, end = self.vad_segment(audio)
            if start is not None and end is not None:
                audio = audio[start:end]
            return audio
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
