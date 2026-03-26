# asr_risk_model.py
"""
极致优化版 - 关闭稳定性测试，只保留置信度分析
新增：对抗性扰动洗涤
目标: <500ms
"""

import numpy as np
import time
from typing import Dict, Optional
from dataclasses import dataclass
import librosa

from asr_engine import create_asr_engine
from confidence_analyzer import ConfidenceAnalyzer
from audio_preprocessor import AudioPreprocessor  # 新增导入

# VAD导入
try:
    import webrtcvad

    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    print("⚠️ 请安装 webrtcvad: pip install webrtcvad")


@dataclass
class OptimizedConfig:
    """极致优化配置"""
    model_size: str = "tiny"  # 保持tiny最快
    enable_vad: bool = True
    vad_aggressiveness: int = 3
    enable_stability: bool = False  # 关闭稳定性测试！
    alpha: float = 1.0  # 只用置信度

    # 新增：对抗性防御配置
    enable_adversarial_defense: bool = True  # 启用对抗性防御
    defense_strength: str = "medium"  # light, medium, strong


class VADProcessor:
    """快速VAD处理器"""

    def __init__(self, aggressiveness=3, sample_rate=16000):
        if not VAD_AVAILABLE:
            raise ImportError("请安装 webrtcvad")

        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration_ms = 30
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """快速VAD处理"""
        audio_int16 = (audio * 32767).astype(np.int16)

        # 补零
        if len(audio_int16) % self.frame_size != 0:
            padding = self.frame_size - (len(audio_int16) % self.frame_size)
            audio_int16 = np.pad(audio_int16, (0, padding), 'constant')

        # VAD检测
        num_frames = len(audio_int16) // self.frame_size
        speech_flags = []

        for i in range(num_frames):
            frame = audio_int16[i * self.frame_size:(i + 1) * self.frame_size]
            try:
                is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
                speech_flags.append(is_speech)
            except:
                speech_flags.append(False)

        # 快速平滑
        if len(speech_flags) > 2:
            for i in range(1, len(speech_flags) - 1):
                if speech_flags[i - 1] == 0 and speech_flags[i] == 1 and speech_flags[i + 1] == 0:
                    speech_flags[i] = 0

        # 应用掩码
        mask = np.repeat(speech_flags, self.frame_size)
        mask = mask[:len(audio)]

        result = audio.copy()
        result[~mask] = 0
        return result


class OptimizedASRRiskModel:
    """极致优化版风险模型 - 集成对抗性防御"""

    def __init__(self, config: Optional[OptimizedConfig] = None):
        self.config = config or OptimizedConfig()

        print("\n" + "=" * 60)
        print("🚀 极致优化ASR风险模型")
        print("=" * 60)
        print(f"模型: {self.config.model_size}")
        print(f"VAD: {'启用' if self.config.enable_vad else '禁用'}")
        print(f"稳定性测试: {'关闭' if not self.config.enable_stability else '开启'}")
        print(f"对抗性防御: {'启用' if self.config.enable_adversarial_defense else '禁用'}")  # 新增
        print(f"防御强度: {self.config.defense_strength}")  # 新增

        # 初始化VAD
        self.vad = None
        if self.config.enable_vad and VAD_AVAILABLE:
            self.vad = VADProcessor(aggressiveness=self.config.vad_aggressiveness)

        # 新增：初始化音频预处理器（含对抗性防御）
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=16000,
            enable_adversarial_defense=self.config.enable_adversarial_defense,
            defense_strength=self.config.defense_strength
        )

        # 初始化ASR引擎
        print("\n📡 初始化ASR引擎...")
        self.engine = create_asr_engine(
            model_size=self.config.model_size,
            device="cpu",
            compute_type="int8"
        )

        # 初始化置信度分析器
        self.confidence_analyzer = ConfidenceAnalyzer(low_conf_threshold=0.5)

        print("✅ 初始化完成")

    def compute_risk(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict:
        """快速风险计算 - 集成对抗性防御"""
        timings = {}

        # 1. VAD预处理
        if self.vad:
            vad_start = time.time()
            audio_processed = self.vad.process(audio)
            timings['vad_ms'] = (time.time() - vad_start) * 1000
        else:
            audio_processed = audio
            timings['vad_ms'] = 0

        # 2. 对抗性扰动洗涤（新增）
        defense_start = time.time()
        audio_processed = self.audio_preprocessor.prepare_for_asr(
            audio_processed,
            sample_rate=sample_rate,
            apply_defense=True
        )
        timings['defense_ms'] = (time.time() - defense_start) * 1000

        # 3. ASR转录
        asr_start = time.time()
        asr_result = self.engine.transcribe(
            audio_processed,
            sample_rate=self.audio_preprocessor.sample_rate,
            language="zh"
        )
        timings['asr_ms'] = (time.time() - asr_start) * 1000

        if not asr_result.success:
            return {
                'text': '',
                'confidence': 0.0,
                'risk_score': 1.0,
                'timings': timings
            }

        # 4. 置信度分析
        confidence = self.confidence_analyzer.analyze(asr_result)

        # 5. 风险计算（只用置信度）
        risk = 1.0 - confidence.confidence_score
        risk = max(0.0, min(1.0, risk))

        timings['total_ms'] = timings['vad_ms'] + timings['defense_ms'] + timings['asr_ms']

        # 风险等级
        if risk < 0.3:
            risk_level = "低风险 ✅"
            decision = "接受"
        elif risk < 0.7:
            risk_level = "中风险 ⚠️"
            decision = "人工确认"
        else:
            risk_level = "高风险 ❌"
            decision = "拒绝"

        return {
            'text': asr_result.text,
            'confidence': confidence.confidence_score,
            'risk_score': risk,
            'risk_level': risk_level,
            'decision': decision,
            'timings': timings
        }

    def cleanup(self):
        """清理资源"""
        self.engine.cleanup()


# ==================== 兼容性别名 ====================
ASRRiskModel = OptimizedASRRiskModel
ASRRiskConfig = OptimizedConfig

__all__ = [
    'OptimizedASRRiskModel',
    'OptimizedConfig',
    'ASRRiskModel',
    'ASRRiskConfig'
]


# ==================== 测试 ====================
def test_optimized():
    """测试优化版"""
    print("测试优化版ASR风险模型（含对抗性防御）")

    config = OptimizedConfig(
        model_size="tiny",
        enable_vad=True,
        enable_adversarial_defense=True,  # 启用防御
        defense_strength="medium"
    )
    model = OptimizedASRRiskModel(config)

    try:
        # 加载音频
        audio, sr = librosa.load("test.wav", sr=16000, mono=True)
        print(f"\n📁 音频: {len(audio) / sr:.1f}秒")

        # 计算风险
        result = model.compute_risk(audio, sr)

        print("\n" + "=" * 60)
        print("📊 评估结果")
        print("=" * 60)
        print(f"识别文本: {result['text']}")
        print(f"置信度:   {result['confidence']:.3f}")
        print(f"风险分数: {result['risk_score']:.3f} {result['risk_level']}")
        print(f"决策:     {result['decision']}")

        print("\n⏱️ 性能统计:")
        print(f"  VAD处理:   {result['timings']['vad_ms']:.1f}ms")
        print(f"  防御处理:   {result['timings']['defense_ms']:.1f}ms")  # 新增
        print(f"  ASR转录:   {result['timings']['asr_ms']:.1f}ms")
        print(f"  总耗时:    {result['timings']['total_ms']:.1f}ms")

        if result['timings']['total_ms'] < 500:
            print(f"\n✅ 实时性达标 (<500ms)")
        else:
            print(f"\n⚠️ 仍需要优化 (>500ms)")

    finally:
        model.cleanup()


if __name__ == "__main__":
    test_optimized()
