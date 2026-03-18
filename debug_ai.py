import os
# 解决 libiomp5md.dll 初始化冲突，这是工程中处理多环境冲突的常用补丁
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ... 其余 import 保持不变
import numpy as np
import librosa
import torch
# 导入你现有的模块
from modules.module1_acoustic.feature_extractor import FeatureExtractor
from modules.module1_acoustic.audio_preprocessor import AudioPreprocessor


def analyze_audio_features(audio_path, label="Unknown"):
    print(f"\n" + "=" * 50)
    print(f"📊 正在分析音频: [{label}] -> {audio_path}")

    # 1. 加载音频
    try:
        audio, sr = librosa.load(audio_path, sr=None)  # 保持原始采样率以观察高频
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # 2. 初始化预处理器与特征提取器 [cite: 29, 53]
    preprocessor = AudioPreprocessor(target_sr=16000)
    extractor = FeatureExtractor(sr=16000)

    # 3. 预处理 (归一化并调整到 RawNet2 期望的长度) [cite: 32, 42]
    # 注意：为了分析频谱，我们先不截断，只做重采样和归一化
    audio_16k = preprocessor.resample(audio, orig_sr=sr)
    audio_norm = preprocessor.normalize(audio_16k)

    # 4. 提取特征 [cite: 54]
    # 返回顺序：13维MFCC, 谱质心, 谱平坦度, 高频能量占比, 能量变化率 [cite: 54, 55, 58, 59]
    features = extractor.extract(audio_norm)

    # 5. 解析并打印关键指标
    # 特征索引解析[cite: 59]:
    # 0-12: MFCC, 13: Centroid, 14: Flatness, 15: High-Freq Ratio, 16: Energy Std
    mfcc_mean = np.mean(features[:13])
    spectral_centroid = features[13]
    spectral_flatness = features[14]
    high_freq_ratio = features[15]
    energy_std = features[16]

    print(f"--- 核心安全特征报告 ---")
    print(f"🔹 高频能量占比 (High-Freq Ratio): {high_freq_ratio:.6f}")
    print(f"🔹 谱平坦度 (Spectral Flatness):  {spectral_flatness:.6f}")
    print(f"🔹 谱质心 (Spectral Centroid):   {spectral_centroid:.2f} Hz")
    print(f"🔹 能量波动标准差 (Energy Std):    {energy_std:.6f}")
    print(f"🔹 MFCC 均值:                    {mfcc_mean:.6f}")
    print("=" * 50)

    return high_freq_ratio


if __name__ == "__main__":
    # 请替换为你电脑上的实际路径
    # 建议对比测试：1. 你录制的真人语音 2. 豆包生成的语音
    real_voice = r"D:\essay of crypto\AutoSafe-V-Guard/human.wav"
    ai_voice = r"D:\essay of crypto\AutoSafe-V-Guard/resent.wav"

    print("🚀 开始比对测试...")
    ratio_real = analyze_audio_features(real_voice, label="真人语音")
    ratio_ai = analyze_audio_features(ai_voice, label="重放语音")

    if ratio_real and ratio_ai:
        diff = (ratio_real / (ratio_ai + 1e-10))
        print(f"\n结论：真人高频比重放高出 {diff:.2f} 倍")