# audio_preprocessor.py
"""
音频预处理模块 - 包含对抗性扰动洗涤
功能：随机平滑、频带重采样、噪声注入
"""

import numpy as np
import librosa
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    音频预处理类
    包含对抗性扰动洗涤功能
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 enable_adversarial_defense: bool = True,
                 defense_strength: str = "medium"):
        """
        初始化音频预处理器

        Args:
            sample_rate: 目标采样率
            enable_adversarial_defense: 是否启用对抗性防御
            defense_strength: 防御强度 ("light", "medium", "strong")
        """
        self.sample_rate = sample_rate
        self.enable_adversarial_defense = enable_adversarial_defense
        self.defense_strength = defense_strength

        # 根据强度设置参数
        self._init_defense_params()

        logger.info(f"音频预处理器初始化: SR={sample_rate}, 防御={enable_adversarial_defense}({defense_strength})")

    def _init_defense_params(self):
        """初始化防御参数"""
        if self.defense_strength == "light":
            self.noise_std = 0.0005  # 极低噪声
            self.downsample_rates = [8000]  # 仅下采样到8k
            self.smooth_window = 3  # 小窗口平滑
        elif self.defense_strength == "medium":
            self.noise_std = 0.001  # 低噪声
            self.downsample_rates = [8000, 12000]  # 多个采样率
            self.smooth_window = 5  # 中窗口平滑
        else:  # strong
            self.noise_std = 0.002  # 中等噪声
            self.downsample_rates = [8000, 12000, 4000]  # 多个采样率
            self.smooth_window = 7  # 大窗口平滑

    def adversarial_purify(self,
                           audio: np.ndarray,
                           sample_rate: int,
                           method: str = "combined") -> np.ndarray:
        """
        对抗性扰动洗涤 - 核心方法

        通过多种技术破坏对抗性扰动：
        1. 频带重采样（下采样再上采样）
        2. 随机噪声注入
        3. 平滑滤波

        Args:
            audio: 输入音频 (float32, 范围[-1, 1])
            sample_rate: 原始采样率
            method: 洗涤方法 ("resample", "noise", "smooth", "combined")

        Returns:
            洗涤后的音频
        """
        if not self.enable_adversarial_defense:
            return audio

        purified = audio.copy()

        if method == "resample":
            purified = self._resample_defense(purified, sample_rate)
        elif method == "noise":
            purified = self._noise_defense(purified)
        elif method == "smooth":
            purified = self._smooth_defense(purified)
        else:  # combined
            purified = self._resample_defense(purified, sample_rate)
            purified = self._noise_defense(purified)
            purified = self._smooth_defense(purified)

        # 确保音频在有效范围内
        purified = np.clip(purified, -1.0, 1.0)

        return purified

    def _resample_defense(self, audio: np.ndarray, original_sr: int) -> np.ndarray:
        """
        频带重采样防御

        原理：对抗性扰动通常在高频区域，重采样会破坏这些精细扰动
        """
        import random

        # 随机选择一个下采样率
        target_sr = random.choice(self.downsample_rates)

        try:
            # 下采样
            audio_down = librosa.resample(
                audio,
                orig_sr=original_sr,
                target_sr=target_sr,
                res_type='scipy'
            )

            # 上采样回原始采样率
            audio_up = librosa.resample(
                audio_down,
                orig_sr=target_sr,
                target_sr=original_sr,
                res_type='scipy'
            )

            # 保持长度一致
            if len(audio_up) > len(audio):
                audio_up = audio_up[:len(audio)]
            elif len(audio_up) < len(audio):
                audio_up = np.pad(audio_up, (0, len(audio) - len(audio_up)), 'constant')

            return audio_up

        except Exception as e:
            logger.warning(f"重采样防御失败: {e}")
            return audio

    def _noise_defense(self, audio: np.ndarray) -> np.ndarray:
        """
        随机噪声注入防御

        原理：添加极低比例的高斯噪声，破坏精细的对抗性扰动
        """
        # 根据音频能量动态调整噪声强度
        audio_energy = np.sqrt(np.mean(audio ** 2))
        noise_std = self.noise_std * (1.0 / (audio_energy + 0.01))  # 自适应
        noise_std = min(noise_std, 0.01)  # 限制最大噪声

        # 添加高斯噪声
        noise = np.random.normal(0, noise_std, audio.shape)
        noisy_audio = audio + noise

        return noisy_audio

    def _smooth_defense(self, audio: np.ndarray) -> np.ndarray:
        """
        平滑滤波防御

        原理：移动平均平滑，抑制高频扰动
        """
        # 使用移动平均滤波
        window = np.ones(self.smooth_window) / self.smooth_window
        smoothed = np.convolve(audio, window, mode='same')

        return smoothed

    def bandpass_filter(self,
                        audio: np.ndarray,
                        low_cut: int = 300,
                        high_cut: int = 8000) -> np.ndarray:
        """
        带通滤波 - 去除人声频带外的噪声

        Args:
            audio: 输入音频
            low_cut: 低频截止频率 (Hz)
            high_cut: 高频截止频率 (Hz)

        Returns:
            滤波后的音频
        """
        from scipy.signal import butter, filtfilt

        nyquist = self.sample_rate / 2
        low = low_cut / nyquist
        high = high_cut / nyquist

        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, audio)

        return filtered

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """音频归一化"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val * 0.95
        return audio

    def prepare_for_asr(self,
                        audio: np.ndarray,
                        sample_rate: int,
                        apply_defense: bool = True) -> np.ndarray:
        """
        完整预处理流程

        Args:
            audio: 输入音频
            sample_rate: 采样率
            apply_defense: 是否应用对抗性防御

        Returns:
            预处理后的音频
        """
        # 1. 重采样到目标采样率
        if sample_rate != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)

        # 2. 对抗性扰动洗涤
        if apply_defense and self.enable_adversarial_defense:
            audio = self.adversarial_purify(audio, self.sample_rate)

        # 3. 归一化
        audio = self.normalize_audio(audio)

        return audio


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("音频预处理器测试")
    print("=" * 60)

    # 创建预处理器
    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        enable_adversarial_defense=True,
        defense_strength="medium"
    )

    # 生成测试音频
    duration = 2.0
    t = np.linspace(0, duration, int(duration * 16000))
    clean_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    # 添加模拟的对抗性扰动（高频噪声）
    adversarial_perturbation = 0.05 * np.sin(2 * np.pi * 8000 * t)
    adversarial_audio = clean_audio + adversarial_perturbation
    adversarial_audio = np.clip(adversarial_audio, -1, 1)

    print(f"\n1. 原始音频能量: {np.sqrt(np.mean(clean_audio ** 2)):.4f}")
    print(f"   对抗扰动能量: {np.sqrt(np.mean(adversarial_perturbation ** 2)):.4f}")

    # 应用防御
    purified = preprocessor.adversarial_purify(adversarial_audio, 16000, method="combined")

    # 计算扰动去除效果
    original_error = np.mean((adversarial_audio - clean_audio) ** 2)
    purified_error = np.mean((purified - clean_audio) ** 2)
    reduction = (1 - purified_error / original_error) * 100

    print(f"\n2. 扰动去除效果:")
    print(f"   原始扰动误差: {original_error:.6f}")
    print(f"   洗涤后误差: {purified_error:.6f}")
    print(f"   扰动减少: {reduction:.1f}%")

    print("\n✅ 测试完成")
