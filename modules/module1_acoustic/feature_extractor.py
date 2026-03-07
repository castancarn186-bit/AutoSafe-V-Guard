"""
弃用

声学特征提取模块
输入：干净音频段 (numpy array, 16kHz)
输出：特征向量 (numpy array)
"""
import numpy as np
import librosa

class FeatureExtractor:
    def __init__(self, sr=16000, n_mfcc=13, n_fft=2048, hop_length=512, high_freq_thresh=4000):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.high_freq_thresh = high_freq_thresh

    def extract(self, audio):
        """
        提取特征，返回特征向量 (numpy array)
        特征顺序：
        - 13维 MFCC 均值
        - 谱质心均值
        - 谱平坦度均值
        - 高频能量占比
        - 能量变化率 (帧能量标准差)
        """
        # 确保音频长度足够 STFT
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)), mode='constant')

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sr, n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        mfcc_mean = np.mean(mfcc, axis=1)  # (13,)

        # 谱质心
        cent = librosa.feature.spectral_centroid(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        cent_mean = np.mean(cent)

        # 谱平坦度
        flat = librosa.feature.spectral_flatness(
            y=audio, n_fft=self.n_fft, hop_length=self.hop_length
        )
        flat_mean = np.mean(flat)

        # 功率谱
        S = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)) ** 2
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)

        # 高频能量占比 (> high_freq_thresh)
        high_mask = freqs > self.high_freq_thresh
        total_energy = np.sum(S, axis=0) + 1e-10
        high_energy = np.sum(S[high_mask, :], axis=0)
        high_ratio = np.mean(high_energy / total_energy)

        # 能量变化率 (帧能量标准差)
        energy_std = np.std(total_energy)

        # 合并所有特征
        features = np.concatenate([
            mfcc_mean,
            [cent_mean, flat_mean, high_ratio, energy_std]
        ])

        return features
