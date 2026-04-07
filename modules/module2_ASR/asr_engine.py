# asr_engine.py
"""
ASR推理封装模块
负责加载Whisper模型，执行语音转写，并输出结构化结果
"""

import numpy as np
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict
import logging

logger = logging.getLogger(__name__)


# ==================== 数据模型定义 ====================
class ASRResult(BaseModel):
    """ASR输出结果的数据模型"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str = ""
    tokens: List[str] = []
    log_probs: List[float] = []
    segments: List[Dict] = []
    duration: float = 0.0
    inference_time: float = 0.0
    language: str = "unknown"
    success: bool = False


@dataclass
class ASRConfig:
    """ASR引擎配置 - 简化版"""
    model_size: str = "tiny"
    device: str = "cpu"
    compute_type: str = "int8"
    language: Optional[str] = None
    beam_size: int = 5
    temperature: float = 0.0
    word_timestamps: bool = True
    vad_filter: bool = True


class AudioPreprocessor:
    """音频预处理工具类"""

    @staticmethod
    def prepare_audio(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """完整的音频预处理流程"""
        # 确保单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        # 确保 float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        # 归一化
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.9
        return audio


class ASREngine:
    """ASR推理封装引擎"""

    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """加载Whisper模型"""
        try:
            logger.info(f"正在加载Whisper模型 {self.config.model_size}...")
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                model_size_or_path=self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
                download_root="./models",
                local_files_only=False
            )
            logger.info(f"✓ 模型加载完成！设备: {self.config.device}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def transcribe(
            self,
            audio: np.ndarray,
            sample_rate: int = 16000,
            language: Optional[str] = None,
            task: str = "transcribe"
    ) -> ASRResult:
        """执行语音转录"""
        start_time = time.time()

        try:
            # 音频预处理
            audio = AudioPreprocessor.prepare_audio(audio, sample_rate)

            # 确定语言
            final_language = language or self.config.language

            # 🔴 简化参数，避免错误
            transcribe_kwargs = {
                "audio": audio,
                "language": final_language,
                "task": task,
                "beam_size": self.config.beam_size,
                "temperature": self.config.temperature,
                "word_timestamps": self.config.word_timestamps,
                "vad_filter": self.config.vad_filter,
            }

            # 移除 None 值
            transcribe_kwargs = {k: v for k, v in transcribe_kwargs.items() if v is not None}

            # 执行转录
            segments, info = self.model.transcribe(**transcribe_kwargs)

            # 收集结果
            all_tokens = []
            all_log_probs = []
            all_segments = []
            full_text = []

            for segment in segments:
                segment_info = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": []
                }

                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        word_info = {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability
                        }
                        segment_info["words"].append(word_info)
                        all_tokens.append(word.word)
                        # 安全计算对数概率
                        if word.probability > 0:
                            all_log_probs.append(np.log(word.probability))
                        else:
                            all_log_probs.append(-100.0)

                full_text.append(segment.text.strip())
                all_segments.append(segment_info)

            # 如果没有词级信息，使用文本分割
            if not all_tokens:
                full_text_str = " ".join(full_text)
                all_tokens = full_text_str.split()
                all_log_probs = [0.0] * len(all_tokens)

            full_text_str = " ".join(full_text)
            duration = len(audio) / sample_rate
            inference_time = time.time() - start_time

            detected_language = info.language if hasattr(info, 'language') else final_language or "unknown"

            return ASRResult(
                text=full_text_str,
                tokens=all_tokens,
                log_probs=all_log_probs,
                segments=all_segments,
                duration=duration,
                inference_time=inference_time,
                language=detected_language,
                success=True
            )

        except Exception as e:
            logger.error(f"转录过程出错: {e}")
            duration = len(audio) / sample_rate if len(audio) > 0 else 0.0
            return ASRResult(
                text="",
                tokens=[],
                log_probs=[],
                segments=[],
                duration=duration,
                inference_time=time.time() - start_time,
                language="unknown",
                success=False
            )

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_size": self.config.model_size,
            "device": self.config.device,
            "compute_type": self.config.compute_type,
            "model_loaded": self.model is not None
        }

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def create_asr_engine(
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None
) -> ASREngine:
    """创建ASR引擎的便捷函数"""
    config = ASRConfig(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        language=language
    )
    return ASREngine(config)


def create_test_audio(
        duration: float = 2.0,
        sample_rate: int = 16000,
        audio_type: str = "noise"
) -> np.ndarray:
    """创建测试音频"""
    n_samples = int(duration * sample_rate)

    if audio_type == "silence":
        return np.zeros(n_samples, dtype=np.float32)
    elif audio_type == "noise":
        return np.random.randn(n_samples).astype(np.float32) * 0.1
    elif audio_type == "tone":
        t = np.linspace(0, duration, n_samples, dtype=np.float32)
        return 0.5 * np.sin(2 * np.pi * 440 * t)
    elif audio_type == "speech_like":
        t = np.linspace(0, duration, n_samples, dtype=np.float32)
        base_freq = 150 + 100 * np.sin(2 * np.pi * 2 * t)
        audio = 0.3 * np.sin(2 * np.pi * base_freq * t)
        for freq in [600, 1200, 2400]:
            audio += 0.1 * np.sin(2 * np.pi * freq * t)
        audio += 0.05 * np.random.randn(n_samples).astype(np.float32)
        # 归一化
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.9
        return audio.astype(np.float32)
    else:
        raise ValueError(f"未知的音频类型: {audio_type}")


if __name__ == "__main__":
    print("=== ASR引擎测试 ===\n")

    engine = create_asr_engine(model_size="tiny", device="cpu")
    print(f"引擎状态: {engine.get_model_info()}\n")

    for name, audio_type in [("静音", "silence"), ("白噪声", "noise"), ("纯音", "tone"), ("模拟语音", "speech_like")]:
        print(f"2. 测试 {name} 音频...")
        audio = create_test_audio(duration=2.0, audio_type=audio_type)
        print(f"   音频: {audio.dtype}, 形状: {audio.shape}, 时长: {len(audio) / 16000:.1f}秒")
        result = engine.transcribe(audio, sample_rate=16000)
        print(f"   结果: '{result.text[:30] if result.text else ''}'")
        print(f"   语言: {result.language}")
        print(f"   成功: {result.success}")
        print(f"   推理时间: {result.inference_time:.2f}秒\n")

    print("3. 测试带参数的转录...")
    audio = create_test_audio(audio_type="speech_like", duration=3.0)
    result = engine.transcribe(audio, sample_rate=16000, language="zh")
    print(f"   指定中文结果: '{result.text}'")
    print(f"   分段数: {len(result.segments)}")

    engine.cleanup()
    print("\n✓ 测试完成！")
