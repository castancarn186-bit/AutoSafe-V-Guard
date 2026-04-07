# stability_checker.py
"""
ASR稳定性测试模块
通过多次解码同一音频，评估ASR输出的稳定性
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pydantic import BaseModel
import logging
from jiwer import wer, cer
import difflib
from asr_engine import ASRResult, ASREngine, create_asr_engine

logger = logging.getLogger(__name__)


class StabilityMetrics(BaseModel):
    """稳定性指标"""
    wer_score: float = 0.0
    cer_score: float = 0.0
    similarity_score: float = 0.0
    consistency_score: float = 0.0

    def __str__(self):
        return (f"稳定性分数: {self.consistency_score:.3f} "
                f"(WER: {self.wer_score:.3f}, "
                f"相似度: {self.similarity_score:.3f})")


@dataclass
class StabilityConfig:
    """稳定性测试配置"""
    num_decodings: int = 2  # 减少解码次数，提高速度
    temperatures: List[float] = None  # 不再使用，保留兼容性
    enable_wers: bool = True
    min_text_length: int = 2

    def __post_init__(self):
        if self.temperatures is None:
            # 只使用确定性解码，因为温度参数不被支持
            self.temperatures = [0.0]


class StabilityChecker:
    """稳定性检查器"""

    def __init__(self,
                 config: Optional[StabilityConfig] = None,
                 asr_engine: Optional[ASREngine] = None):
        """
        初始化稳定性检查器

        Args:
            config: 稳定性测试配置
            asr_engine: ASR引擎实例，如果为None则创建新的
        """
        self.config = config or StabilityConfig()

        # 使用提供的引擎或创建新的
        if asr_engine:
            self.engine = asr_engine
            self.own_engine = False
        else:
            self.engine = create_asr_engine(model_size="tiny", device="cpu")
            self.own_engine = True

    def check_stability(self,
                        audio: np.ndarray,
                        sample_rate: int = 16000,
                        reference_text: Optional[str] = None) -> StabilityMetrics:
        """
        检查音频的ASR稳定性

        Args:
            audio: 音频数据
            sample_rate: 采样率
            reference_text: 参考文本（如果有），用于计算准确率

        Returns:
            StabilityMetrics: 稳定性指标
        """
        try:
            # 1. 多次解码
            decoding_results = self._multiple_decodings(audio, sample_rate)

            # 2. 提取文本
            texts = [result.text for result in decoding_results if result.text.strip()]

            if len(texts) < 2:
                logger.warning("有效解码结果不足，无法计算稳定性")
                return self._create_error_metrics()

            # 3. 计算各种稳定性指标
            wer_score = self._calculate_wer(texts) if self.config.enable_wers else 0.0
            cer_score = self._calculate_cer(texts) if self.config.enable_wers else 0.0
            similarity_score = self._calculate_similarity(texts)

            # 4. 计算综合一致性分数
            consistency_score = self._calculate_consistency_score(
                wer_score, cer_score, similarity_score
            )

            metrics = StabilityMetrics(
                wer_score=wer_score,
                cer_score=cer_score,
                similarity_score=similarity_score,
                consistency_score=consistency_score
            )

            logger.info(f"稳定性分析完成: {metrics}")

            return metrics

        except Exception as e:
            logger.error(f"稳定性检查失败: {e}")
            return self._create_error_metrics()

    def _multiple_decodings(self,
                            audio: np.ndarray,
                            sample_rate: int) -> List[ASRResult]:
        """多次解码同一音频（不使用temperature参数）"""
        results = []

        for i in range(self.config.num_decodings):
            try:
                logger.debug(f"解码 {i + 1}/{self.config.num_decodings}")

                # 不使用temperature参数，直接转录
                result = self.engine.transcribe(
                    audio,
                    sample_rate=sample_rate
                )

                if result.success and result.text.strip():
                    results.append(result)

            except Exception as e:
                logger.warning(f"第{i + 1}次解码失败: {e}")
                continue

        return results

    def _calculate_wer(self, texts: List[str]) -> float:
        """计算词错误率（Word Error Rate）"""
        if len(texts) < 2:
            return 1.0

        wers = []

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if texts[i] and texts[j]:
                    try:
                        w = wer(texts[i], texts[j])
                        wers.append(w)
                    except:
                        pass

        return np.mean(wers) if wers else 1.0

    def _calculate_cer(self, texts: List[str]) -> float:
        """计算字错误率（Character Error Rate）"""
        if len(texts) < 2:
            return 1.0

        cers = []

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if texts[i] and texts[j]:
                    try:
                        c = cer(texts[i], texts[j])
                        cers.append(c)
                    except:
                        pass

        return np.mean(cers) if cers else 1.0

    def _calculate_similarity(self, texts: List[str]) -> float:
        """计算文本相似度"""
        if len(texts) < 2:
            return 0.0

        similarities = []

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if texts[i] and texts[j]:
                    seq = difflib.SequenceMatcher(None, texts[i], texts[j])
                    similarity = seq.ratio()
                    similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    def _calculate_consistency_score(self,
                                     wer_score: float,
                                     cer_score: float,
                                     similarity_score: float) -> float:
        """
        计算综合一致性分数
        """
        # 1. 文本相似度指标 (权重0.5)
        similarity_metric = similarity_score

        # 2. WER指标 (权重0.3)
        wer_metric = max(0.0, 1.0 - min(1.0, wer_score))

        # 3. CER指标 (权重0.2)
        cer_metric = max(0.0, 1.0 - min(1.0, cer_score))

        # 权重分配
        weights = [0.5, 0.3, 0.2]
        metrics = [similarity_metric, wer_metric, cer_metric]

        consistency_score = sum(w * m for w, m in zip(weights, metrics))

        # 确保在[0,1]范围内
        consistency_score = max(0.0, min(1.0, consistency_score))

        return consistency_score

    def _create_error_metrics(self) -> StabilityMetrics:
        """创建错误指标"""
        return StabilityMetrics(
            wer_score=1.0,
            cer_score=1.0,
            similarity_score=0.0,
            consistency_score=0.0
        )

    def cleanup(self):
        """清理资源"""
        if self.own_engine and hasattr(self.engine, 'cleanup'):
            self.engine.cleanup()


# ==================== 测试代码 ====================
def test_stability_checker():
    """测试稳定性检查器"""
    print("=== 测试稳定性检查器 ===")

    # 创建稳定性检查器
    checker = StabilityChecker()

    # 测试文本相似度计算
    print("\n2. 测试文本相似度计算:")

    test_cases = [
        ("请打开车窗", "请打开车窗", "完全相同"),
        ("请打开车窗", "请打开窗", "轻微差异"),
        ("请打开车窗", "关闭车窗", "较大差异"),
        ("hello world", "hello world!", "标点差异"),
    ]

    for s1, s2, desc in test_cases:
        sim = difflib.SequenceMatcher(None, s1, s2).ratio()
        print(f"   '{s1}' vs '{s2}' ({desc}): 相似度={sim:.3f}")

    # 测试WER/CER计算
    print("\n3. 测试WER/CER计算:")

    from jiwer import wer, cer
    wer_test = wer("请打开车窗", "请打开窗")
    cer_test = cer("请打开车窗", "请打开窗")

    print(f"   WER('请打开车窗', '请打开窗') = {wer_test:.3f}")
    print(f"   CER('请打开车窗', '请打开窗') = {cer_test:.3f}")

    # 清理
    checker.cleanup()


def test_with_real_audio():
    """使用真实音频测试稳定性"""
    print("\n=== 使用真实音频测试 ===")

    try:
        import librosa

        audio, sr = librosa.load("test.wav", sr=16000, mono=True)
        print(f"加载音频: {len(audio) / sr:.1f}秒")

        checker = StabilityChecker()
        metrics = checker.check_stability(audio, sample_rate=sr)

        print(f"稳定性分析结果:")
        print(f"  综合稳定性分数: {metrics.consistency_score:.3f}")
        print(f"  文本相似度: {metrics.similarity_score:.3f}")
        print(f"  平均WER: {metrics.wer_score:.3f}")
        print(f"  平均CER: {metrics.cer_score:.3f}")

        stability = "高" if metrics.consistency_score > 0.7 else "中" if metrics.consistency_score > 0.4 else "低"
        print(f"  稳定性评估: {stability}")

        checker.cleanup()

    except Exception as e:
        print(f"真实音频测试失败: {e}")


if __name__ == "__main__":
    print("稳定性测试模块")
    print("=" * 50)

    test_stability_checker()
    test_with_real_audio()

    print("\n" + "=" * 50)
    print("测试完成！")
