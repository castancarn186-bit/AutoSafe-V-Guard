# vad_processor.py
"""
WebRTC VAD语音活动检测预处理模块
用于过滤背景噪声，提升ASR准确率
"""

import numpy as np
import webrtcvad
import collections
import sys


class VADProcessor:
    """语音活动检测处理器"""

    def __init__(self, aggressiveness=3, sample_rate=16000, frame_duration_ms=30):
        """
        初始化VAD处理器

        Args:
            aggressiveness: VAD激进程度 (0-3)，3最激进（过滤更干净）
            sample_rate: 采样率 (必须为8000, 16000, 32000, 48000)
            frame_duration_ms: 帧时长 (10, 20, 30ms)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)  # 每帧样本数

        # 初始化VAD
        self.vad = webrtcvad.Vad(aggressiveness)

        print(f"✓ VAD初始化: 激进程度={aggressiveness}, 帧大小={self.frame_size}样本")

    def is_speech(self, audio_frame):
        """
        检测单帧是否为语音

        Args:
            audio_frame: int16格式的音频帧

        Returns:
            bool: 是否为语音
        """
        return self.vad.is_speech(audio_frame, self.sample_rate)

    def process(self, audio: np.ndarray, return_mask: bool = False):
        """
        处理音频，提取语音段

        Args:
            audio: float32格式的音频 [-1, 1]
            return_mask: 是否返回语音掩码

        Returns:
            处理后的音频（仅保留语音段）或 (音频, 掩码)
        """
        # 转换为int16 (WebRTC VAD需要)
        audio_int16 = (audio * 32767).astype(np.int16)

        # 确保音频长度是帧大小的整数倍
        if len(audio_int16) % self.frame_size != 0:
            # 补零到整数倍
            padding = self.frame_size - (len(audio_int16) % self.frame_size)
            audio_int16 = np.pad(audio_int16, (0, padding), 'constant')

        # 分帧处理
        num_frames = len(audio_int16) // self.frame_size
        speech_flags = []

        for i in range(num_frames):
            frame = audio_int16[i * self.frame_size:(i + 1) * self.frame_size]
            frame_bytes = frame.tobytes()

            try:
                is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
                speech_flags.append(is_speech)
            except:
                speech_flags.append(False)

        # 简单的语音段合并（去除孤立的短语音/静音）
        speech_flags = self._smooth_flags(speech_flags)

        # 构建语音掩码
        mask = np.repeat(speech_flags, self.frame_size)

        # 裁剪到原始长度
        mask = mask[:len(audio)]

        # 应用掩码（将非语音段置零）
        processed_audio = audio.copy()
        processed_audio[~mask] = 0

        if return_mask:
            return processed_audio, mask
        else:
            return processed_audio

    def _smooth_flags(self, flags, min_speech_duration=3, min_silence_duration=2):
        """
        平滑VAD标志，去除孤立的短语音/静音

        Args:
            flags: VAD标志列表
            min_speech_duration: 最小语音段长度（帧数）
            min_silence_duration: 最小静音段长度（帧数）
        """
        smoothed = flags.copy()

        # 将列表转换为字符串便于处理
        flag_str = ''.join(['1' if f else '0' for f in flags])

        # 去除孤立的短语音（如 010 -> 000）
        import re
        # 替换孤立的1（长度小于min_speech_duration）
        pattern = f'0(1{{1,{min_speech_duration - 1}}})0'
        flag_str = re.sub(pattern, lambda m: '0' + '0' * len(m.group(1)) + '0', flag_str)

        # 去除孤立的短静音（如 101 -> 111）
        pattern = f'1(0{{1,{min_silence_duration - 1}}})1'
        flag_str = re.sub(pattern, lambda m: '1' + '1' * len(m.group(1)) + '1', flag_str)

        # 转换回布尔列表
        smoothed = [c == '1' for c in flag_str]

        return smoothed

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """便捷调用"""
        return self.process(audio)


# ==================== 测试代码 ====================
def test_vad():
    """测试VAD功能"""
    import time
    import librosa

    print("=" * 60)
    print("测试WebRTC VAD")
    print("=" * 60)

    # 创建测试音频
    duration = 3.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(duration * sample_rate))

    # 生成语音段 (0.5-1.5秒和2.0-2.5秒)
    audio = np.zeros_like(t)

    # 语音段1
    mask1 = (t >= 0.5) & (t < 1.5)
    audio[mask1] = 0.5 * np.sin(2 * np.pi * 200 * t[mask1])

    # 语音段2
    mask2 = (t >= 2.0) & (t < 2.5)
    audio[mask2] = 0.5 * np.sin(2 * np.pi * 250 * t[mask2])

    # 添加背景噪声
    noise = 0.05 * np.random.randn(len(audio))
    audio_noisy = audio + noise

    # 归一化
    audio_noisy = audio_noisy / np.max(np.abs(audio_noisy)) * 0.9

    print(f"\n1. 创建测试音频: {duration}秒")

    # 测试不同激进程度的VAD
    for agg in [0, 1, 2, 3]:
        print(f"\n2. 测试激进程度 {agg}:")
        vad = VADProcessor(aggressiveness=agg)

        start_time = time.time()
        processed, mask = vad.process(audio_noisy, return_mask=True)
        elapsed = (time.time() - start_time) * 1000

        speech_ratio = np.sum(mask) / len(mask)
        print(f"   处理时间: {elapsed:.2f}ms")
        print(f"   语音比例: {speech_ratio:.1%}")

        # 计算准确率（与真实掩码对比）
        true_mask = mask1 | mask2
        true_mask = true_mask[:len(mask)]

        accuracy = np.sum(mask == true_mask) / len(mask)
        print(f"   准确率: {accuracy:.1%}")

    return True


if __name__ == "__main__":
    test_vad()