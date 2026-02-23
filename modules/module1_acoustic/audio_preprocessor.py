"""
音频预处理流水线
- 重采样（统一到16kHz）
- 归一化
- 谱减法降噪
- VAD 截取有效语音段
"""
import numpy as np
import librosa
import scipy.signal
import webrtcvad
import struct

class AudioPreprocessor:
    def __init__(self, target_sr=16000, vad_aggressiveness=2, noise_floor=0.001):
        """
        :param target_sr: 目标采样率
        :param vad_aggressiveness: VAD 激进程度 (0-3)
        :param noise_floor: 谱减法降噪底噪阈值
        """
        self.target_sr = target_sr
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.noise_floor = noise_floor
        # 预加重系数
        self.pre_emphasis = 0.97

    def resample(self, audio, orig_sr):
        """重采样到目标采样率"""
        if orig_sr == self.target_sr:
            return audio
        # librosa.resample 要求输入为浮点型
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=self.target_sr)

    def normalize(self, audio):
        """归一化到 [-1, 1]"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio

    def pre_emphasis_filter(self, audio):
        """预加重，提升高频"""
        return np.append(audio[0], audio[1:] - self.pre_emphasis * audio[:-1])

    def spectral_subtraction(self, audio, n_fft=512, hop_length=128):
        """
        简单谱减法降噪
        假设前5帧为噪声估计
        """
        # 计算STFT
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(D)
        phase = np.angle(D)

        # 噪声估计（前5帧）
        noise_est = np.mean(magnitude[:, :5], axis=1, keepdims=True)

        # 谱减
        magnitude_clean = magnitude - noise_est
        magnitude_clean = np.maximum(magnitude_clean, self.noise_floor)

        # 重建信号
        D_clean = magnitude_clean * np.exp(1j * phase)
        audio_clean = librosa.istft(D_clean, hop_length=hop_length)
        return audio_clean

    def vad_segment(self, audio, frame_duration_ms=30):
        """
        使用 WebRTC VAD 截取有效语音段
        返回第一个有效语音段的起始和结束索引（若没有则返回 None）
        """
        # 转换为16位 PCM (webrtcvad 需要 bytes)
        # 音频已经是 float32 在 [-1,1]，需要转为 int16
        audio_int16 = (audio * 32767).astype(np.int16)

        # 计算每帧样本数
        frame_len = int(self.target_sr * frame_duration_ms / 1000)
        # 确保音频长度至少为一帧
        if len(audio_int16) < frame_len:
            return None, None

        # 分帧并标记是否为语音
        is_speech = []
        for i in range(0, len(audio_int16) - frame_len + 1, frame_len):
            frame = audio_int16[i:i+frame_len]
            # 转为 bytes
            frame_bytes = frame.tobytes()
            is_speech.append(self.vad.is_speech(frame_bytes, self.target_sr))

        # 合并连续语音段
        in_speech = False
        start_idx = 0
        segments = []
        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                start_idx = i * frame_len
                in_speech = True
            elif not speech and in_speech:
                end_idx = i * frame_len
                segments.append((start_idx, end_idx))
                in_speech = False
        if in_speech:
            segments.append((start_idx, len(audio_int16)))

        if not segments:
            return None, None
        # 返回最长语音段
        longest = max(segments, key=lambda seg: seg[1]-seg[0])
        return longest[0], longest[1]

    def process(self, audio, orig_sr=None):
        """
        完整预处理流水线：
        1. 重采样 (若提供 orig_sr)
        2. 预加重
        3. 谱减法降噪
        4. 归一化
        5. VAD 截取（若截取失败，返回整个音频）
        返回 clean_audio_segment (numpy array)
        """
        if orig_sr is not None:
            audio = self.resample(audio, orig_sr)
        # 预加重
        audio = self.pre_emphasis_filter(audio)
        # 降噪
        audio = self.spectral_subtraction(audio)
        # 归一化
        audio = self.normalize(audio)
        # VAD
        start, end = self.vad_segment(audio)
        if start is not None and end is not None and end - start > 0:
            audio = audio[start:end]
        return audio