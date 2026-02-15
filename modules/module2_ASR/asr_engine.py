# asr_engine.py

import numpy as np
import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from pydantic import BaseModel
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 数据模型定义 ====================
class ASRResult(BaseModel):
    """ASR输出结果的数据模型"""
    text: str = ""  # 完整文本
    tokens: List[str] = []  # token列表
    log_probs: List[float] = []  # 每个token的对数概率
    segments: List[Dict] = []  # 分段信息
    duration: float = 0.0  # 音频时长
    inference_time: float = 0.0  # 推理耗时
    language: str = "unknown"  # 检测到的语言
    success: bool = False  # 是否成功

    class Config:
        arbitrary_types_allowed = True


@dataclass
class ASRConfig:
    """ASR引擎配置 - 简化版本"""
    model_size: str = "tiny"  # tiny, base, small
    device: str = "cpu"  # cpu, cuda
    compute_type: str = "int8"  # int8, float16, float32
    language: Optional[str] = None  # 指定语言，如 "zh", "en"
    beam_size: int = 1  # 减小beam size以提高稳定性
    temperature: float = 0.0  # 确定性输出
    enable_timestamps: bool = False  # 是否启用时间戳（简化版先关闭）
    enable_vad: bool = False  # 是否启用VAD（简化版先关闭）


# ==================== 音频预处理工具 ====================
class AudioPreprocessor:
    """音频预处理工具类"""

    @staticmethod
    def ensure_float32(audio: np.ndarray) -> np.ndarray:
        """确保音频为float32类型"""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return audio

    @staticmethod
    def normalize_audio(audio: np.ndarray, target_max: float = 0.9) -> np.ndarray:
        """归一化音频"""
        if len(audio) == 0:
            return audio

        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * target_max
        return audio

    @staticmethod
    def ensure_mono(audio: np.ndarray) -> np.ndarray:
        """确保音频为单声道"""
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        return audio

    @staticmethod
    def prepare_audio(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        完整的音频预处理流程

        Args:
            audio: 原始音频
            sample_rate: 采样率

        Returns:
            预处理后的音频
        """
        # 1. 转换为单声道
        audio = AudioPreprocessor.ensure_mono(audio)

        # 2. 确保float32类型
        audio = AudioPreprocessor.ensure_float32(audio)

        # 3. 归一化
        audio = AudioPreprocessor.normalize_audio(audio)

        return audio


# ==================== ASR引擎核心 ====================
class ASREngine:
    """简化稳定的ASR引擎"""

    def __init__(self, config: Optional[ASRConfig] = None):
        """
        初始化ASR引擎

        Args:
            config: ASR配置，如果为None则使用默认配置
        """
        self.config = config or ASRConfig()
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """加载Whisper模型 - 简化稳定版本"""
        try:
            logger.info(f"正在加载Whisper模型 {self.config.model_size}...")

            # 动态导入，避免不必要的依赖
            from faster_whisper import WhisperModel

            # 简化模型加载参数
            self.model = WhisperModel(
                model_size_or_path=self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
                download_root="./models",
                local_files_only=False
            )

            logger.info(f"✓ 模型加载完成！设备: {self.config.device}")

        except ImportError:
            logger.error("请先安装 faster-whisper: pip install faster-whisper")
            raise
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def transcribe(
            self,
            audio: np.ndarray,
            sample_rate: int = 16000,
            **kwargs
    ) -> ASRResult:
        """
        执行语音转录 - 简化稳定版本

        Args:
            audio: 音频数据，形状为 (samples,)
            sample_rate: 采样率，默认16000Hz
            **kwargs: 额外参数，会覆盖config中的设置

        Returns:
            ASRResult: 结构化的转录结果
        """
        start_time = time.time()

        # 合并配置
        config = self._merge_config(kwargs)

        try:
            # 1. 音频预处理
            audio = AudioPreprocessor.prepare_audio(audio, sample_rate)

            # 2. 计算音频时长
            duration = len(audio) / sample_rate if len(audio) > 0 else 0

            # 3. 构建转录参数 - 使用最小参数集
            transcribe_kwargs = self._build_transcribe_kwargs(config)

            # 4. 执行转录
            logger.info("开始转录...")
            segments, info = self.model.transcribe(
                audio=audio,
                **transcribe_kwargs
            )

            # 5. 处理结果
            result = self._process_transcription_result(
                segments, info, duration, start_time
            )
            result.success = True

            return result

        except Exception as e:
            logger.error(f"转录过程出错: {e}")

            # 返回错误结果
            inference_time = time.time() - start_time
            duration = len(audio) / sample_rate if len(audio) > 0 else 0

            return ASRResult(
                text="",
                tokens=[],
                log_probs=[],
                segments=[],
                duration=duration,
                inference_time=inference_time,
                language="unknown",
                success=False
            )

    def _merge_config(self, kwargs: Dict) -> Dict:
        """合并配置参数"""
        config = {
            "language": self.config.language,
            "beam_size": self.config.beam_size,
            "temperature": self.config.temperature,
            "enable_timestamps": self.config.enable_timestamps,
            "enable_vad": self.config.enable_vad,
        }
        config.update(kwargs)
        return config

    def _build_transcribe_kwargs(self, config: Dict) -> Dict:
        """构建转录参数 - 简化稳定版本"""
        kwargs = {
            "language": config.get("language"),
            "beam_size": config.get("beam_size", 1),
            "temperature": config.get("temperature", 0.0),
        }

        # 可选参数
        if config.get("enable_timestamps"):
            kwargs["word_timestamps"] = True

        if config.get("enable_vad"):
            kwargs["vad_filter"] = True
            kwargs["vad_parameters"] = {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "max_speech_duration_s": 3600,
                "min_silence_duration_ms": 1000
            }

        # 移除None值
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        return kwargs

    def _process_transcription_result(
            self,
            segments,
            info,
            duration: float,
            start_time: float
    ) -> ASRResult:
        """处理转录结果"""
        all_tokens = []
        all_log_probs = []
        all_segments = []
        full_text_parts = []

        # 处理每个segment
        for segment in segments:
            segment_text = segment.text.strip()
            if segment_text:
                full_text_parts.append(segment_text)

            # 构建segment信息
            segment_info = {
                "start": getattr(segment, 'start', 0.0),
                "end": getattr(segment, 'end', 0.0),
                "text": segment_text,
            }

            # 获取词级信息（如果可用）
            if hasattr(segment, 'words') and segment.words:
                words_info = []
                for word in segment.words:
                    word_text = getattr(word, 'word', '')
                    word_prob = getattr(word, 'probability', 0.0)

                    if word_text:
                        all_tokens.append(word_text)
                        # 处理概率，避免log(0)
                        if word_prob > 0:
                            all_log_probs.append(np.log(word_prob))
                        else:
                            all_log_probs.append(-100.0)

                    words_info.append({
                        "word": word_text,
                        "start": getattr(word, 'start', 0.0),
                        "end": getattr(word, 'end', 0.0),
                        "probability": word_prob
                    })

                segment_info["words"] = words_info

            all_segments.append(segment_info)

        # 如果没有获取到词级信息，从文本中提取
        if not all_tokens and full_text_parts:
            full_text = " ".join(full_text_parts)
            all_tokens = full_text.split()
            # 给默认概率
            all_log_probs = [0.0] * len(all_tokens)

        # 构建完整文本
        full_text = " ".join(full_text_parts) if full_text_parts else ""

        # 获取语言信息
        language = getattr(info, 'language', 'unknown')

        # 计算推理时间
        inference_time = time.time() - start_time

        return ASRResult(
            text=full_text,
            tokens=all_tokens,
            log_probs=all_log_probs,
            segments=all_segments,
            duration=duration,
            inference_time=inference_time,
            language=language,
            success=True
        )

    def get_status(self) -> Dict:
        """获取引擎状态"""
        return {
            "model_size": self.config.model_size,
            "device": self.config.device,
            "compute_type": self.config.compute_type,
            "model_loaded": self.model is not None
        }

    def cleanup(self):
        """清理资源"""
        if self.model is not None:
            del self.model
            self.model = None
            logger.info("✓ 模型资源已清理")


# ==================== 工厂函数 ====================
def create_asr_engine(
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        **kwargs
) -> ASREngine:
    """
    创建ASR引擎的便捷函数

    Args:
        model_size: 模型大小
        device: 运行设备
        compute_type: 计算精度
        **kwargs: 其他配置参数

    Returns:
        ASREngine实例
    """
    config = ASRConfig(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        **kwargs
    )
    return ASREngine(config)


# ==================== 测试函数 ====================
def create_test_audio(
        duration: float = 2.0,
        sample_rate: int = 16000,
        audio_type: str = "noise"
) -> np.ndarray:
    """
    创建测试音频

    Args:
        duration: 音频时长（秒）
        sample_rate: 采样率
        audio_type: 音频类型
            - "silence": 静音
            - "noise": 白噪声
            - "tone": 单音调
            - "speech_like": 类似语音的信号

    Returns:
        音频数组
    """
    n_samples = int(duration * sample_rate)

    if audio_type == "silence":
        return np.zeros(n_samples, dtype=np.float32)

    elif audio_type == "noise":
        # 白噪声
        return np.random.randn(n_samples).astype(np.float32) * 0.1

    elif audio_type == "tone":
        # 440Hz纯音
        t = np.linspace(0, duration, n_samples, dtype=np.float32)
        return 0.5 * np.sin(2 * np.pi * 440 * t)

    elif audio_type == "speech_like":
        # 模拟语音的信号
        t = np.linspace(0, duration, n_samples, dtype=np.float32)

        # 基频变化（模拟语调）
        base_freq = 150 + 100 * np.sin(2 * np.pi * 2 * t)

        # 生成音频
        audio = 0.3 * np.sin(2 * np.pi * base_freq * t)

        # 添加共振峰
        for freq in [600, 1200, 2400]:
            audio += 0.1 * np.sin(2 * np.pi * freq * t)

        # 添加噪声
        audio += 0.05 * np.random.randn(n_samples).astype(np.float32)

        return AudioPreprocessor.normalize_audio(audio)

    else:
        raise ValueError(f"未知的音频类型: {audio_type}")


def test_asr_engine():
    """测试ASR引擎"""
    print("=== ASR引擎测试 ===\n")

    # 创建引擎
    print("1. 创建ASR引擎...")
    engine = create_asr_engine(model_size="tiny", device="cpu")

    print(f"引擎状态: {engine.get_status()}\n")

    # 测试不同类型的音频
    test_cases = [
        ("静音", "silence"),
        ("白噪声", "noise"),
        ("纯音", "tone"),
        ("模拟语音", "speech_like"),
    ]

    for name, audio_type in test_cases:
        print(f"2. 测试 {name} 音频...")

        # 生成测试音频
        audio = create_test_audio(
            duration=2.0,
            audio_type=audio_type
        )

        print(f"   音频: {audio.dtype}, 形状: {audio.shape}, 时长: {len(audio) / 16000:.1f}秒")

        # 执行转录
        result = engine.transcribe(audio, sample_rate=16000)

        # 显示结果
        print(f"   结果: '{result.text[:30]}{'...' if len(result.text) > 30 else ''}'")
        print(f"   语言: {result.language}")
        print(f"   成功: {result.success}")
        print(f"   推理时间: {result.inference_time:.2f}秒")

        if result.tokens:
            print(f"   Token数: {len(result.tokens)}")
            if result.log_probs and len(result.log_probs) > 0:
                avg_prob = np.exp(np.mean(result.log_probs))
                print(f"   平均概率: {avg_prob:.4f}")

        print()

    # 测试带参数的转录
    print("3. 测试带参数的转录...")
    audio = create_test_audio(audio_type="speech_like", duration=3.0)

    result = engine.transcribe(
        audio,
        sample_rate=16000,
        language="zh",  # 指定中文
        beam_size=3,
        enable_timestamps=True,
        enable_vad=True
    )

    print(f"   指定中文结果: '{result.text}'")
    print(f"   分段数: {len(result.segments)}")

    # 清理
    engine.cleanup()
    print("\n✓ 测试完成！")


if __name__ == "__main__":
    test_asr_engine()