# 信号处理工具 负责 FFT 变换、高频噪声分析、超声波水印检测等物理层特征提取。
import numpy as np


class SignalProcessor:
    def __init__(self, sr=16000):
        self.sr = sr

    def detect_replay_attack(self, audio):
        """
        通过频段幅值衰减率分析，精准识别物理重放攻击
        """
        if audio is None or len(audio) == 0:
            return 0.0

        # 1. 统一量纲，绝对不能让 int16 进来炸毁数值
        audio_norm = np.array(audio, dtype=np.float32)
        if np.max(np.abs(audio_norm)) > 1.0:
            audio_norm = audio_norm / 32768.0

        # 2. 静音防暴走 (防止纯静音乱报警)
        rms = np.sqrt(np.mean(audio_norm ** 2))
        if rms < 0.005:
            return 0.0

            # 3. 计算频域的幅值谱 (只用 abs，千万别用平方，防止底噪尖峰干扰)
        freqs = np.fft.rfftfreq(len(audio_norm), 1 / self.sr)
        fft_mag = np.abs(np.fft.rfft(audio_norm))

        # 4. 核心物理特征：对比【中频段 1k-4k】与【高频段 4k-8k】
        # 避开 1000Hz 以下极不稳定的基频区。手机喇叭在 4k 后能量会断崖下跌。
        mid_mask = (freqs > 1000) & (freqs <= 4000)
        high_mask = (freqs > 4000) & (freqs <= 8000)

        mid_energy = np.sum(fft_mag[mid_mask]) + 1e-6
        high_energy = np.sum(fft_mag[high_mask]) + 1e-6

        ratio = high_energy / mid_energy

        # 正常人声泛音平滑过渡 (ratio 约 0.2~0.5)
        # 手机外放被物理截断 (ratio 极低，通常 < 0.1)
        if ratio < 0.10:
            return 0.95  # 确认为物理重放
        elif ratio < 0.15:
            return 0.65  # 疑似重放
        else:
            return 0.05  # 真人