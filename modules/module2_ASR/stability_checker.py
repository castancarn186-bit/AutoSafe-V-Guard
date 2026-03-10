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
    wer_score: float  # 词错误率（0-∞，0最好）
    cer_score: float  # 字错误率（0-∞，0最好）
    similarity_score: float  # 文本相似度（0-1，1最好）
    consistency_score: float  # 一致性分数（0-1，1最稳定）

    def __str__(self):
        return (f"稳定性分数: {self.consistency_score:.3f} "
                f"(WER: {self.wer_score:.3f}, "
                f"相似度: {self.similarity_score:.3f})")


@dataclass
class StabilityConfig:
    """稳定性测试配置"""
    num_decodings: int = 3  # 解码次数
    temperatures: List[float] = None  # 温度参数列表
    enable_wers: bool = True  # 是否计算WER/CER
    min_text_length: int = 2  # 最小文本长度（字符）

    def __post_init__(self):
        if self.temperatures is None:
            # 默认：一次确定性解码 + 两次随机解码
            self.temperatures = [0.0, 0.6, 0.8]


class StabilityChecker:
    def __init__(self,
                 config: Optional[StabilityConfig] = None,
                 asr_engine: Optional[ASREngine] = None,
                 **kwargs):
        """
        初始化稳定性检查器

        Args:
            config: StabilityConfig对象或字典配置
            asr_engine: ASR引擎实例
            **kwargs: 额外的配置参数（如果config是字典）
        """
        # 处理配置参数
        if config is None:
            self.config = StabilityConfig()
        elif isinstance(config, dict):
            # 如果是字典，转换为StabilityConfig对象
            self.config = StabilityConfig(**config)
        else:
            # 已经是StabilityConfig对象
            self.config = config

        # 更新配置（如果有额外的kwargs）
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # 使用提供的引擎或创建新的
        if asr_engine:
            self.engine = asr_engine
            self.own_engine = False
        else:
            self.engine = create_asr_engine(model_size="base", device="cpu")
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

            # 4. 如果有参考文本，计算相对稳定性
            accuracy_score = 0.0
            if reference_text:
                accuracy_score = self._calculate_accuracy(texts, reference_text)

            # 5. 计算综合一致性分数
            consistency_score = self._calculate_consistency_score(
                wer_score, cer_score, similarity_score, accuracy_score
            )

            metrics = StabilityMetrics(
                wer_score=wer_score,
                cer_score=cer_score,
                similarity_score=similarity_score,
                consistency_score=consistency_score
            )

            logger.info(f"稳定性分析完成: {metrics}")
            logger.debug(f"解码结果: {texts}")

            return metrics

        except Exception as e:
            logger.error(f"稳定性检查失败: {e}")
            return self._create_error_metrics()

    def _multiple_decodings(self,
                            audio: np.ndarray,
                            sample_rate: int) -> List[ASRResult]:
        """多次解码同一音频"""
        results = []

        for i, temperature in enumerate(self.config.temperatures[:self.config.num_decodings]):
            try:
                logger.debug(f"解码 {i + 1}/{self.config.num_decodings}, temperature={temperature}")

                # 使用不同温度参数解码
                result = self.engine.transcribe(
                    audio,
                    sample_rate=sample_rate,
                    temperature=temperature,
                    beam_size=1 if temperature > 0 else 5,  # 随机解码时减小beam size
                    enable_vad=False  # 多次解码时禁用VAD以确保一致性
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
            return 1.0  # 无法计算，返回最差值

        wers = []

        # 计算所有文本对之间的WER
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
                    # 使用difflib计算相似度
                    seq = difflib.SequenceMatcher(None, texts[i], texts[j])
                    similarity = seq.ratio()
                    similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    def _calculate_accuracy(self, texts: List[str], reference: str) -> float:
        """计算相对于参考文本的准确率"""
        if not reference or not texts:
            return 0.0

        accuracies = []

        for text in texts:
            if text:
                try:
                    # 使用1-WER作为准确率估计
                    accuracy = 1.0 - min(1.0, wer(reference, text))
                    accuracies.append(accuracy)
                except:
                    pass

        return np.mean(accuracies) if accuracies else 0.0

    def _calculate_consistency_score(self,
                                     wer_score: float,
                                     cer_score: float,
                                     similarity_score: float,
                                     accuracy_score: float = 0.0) -> float:
        """
        计算综合一致性分数

        算法：基于多个指标加权计算
        1. 文本相似度（权重40%）
        2. 词错误率（负向，权重30%）
        3. 字错误率（负向，权重20%）
        4. 准确率（如果有参考，权重10%）
        """
        # 1. 文本相似度指标 (0-1，越高越好)
        similarity_metric = similarity_score

        # 2. WER指标 (0-∞，越低越好)
        # WER归一化：假设WER超过1.0就很差
        wer_metric = max(0.0, 1.0 - min(1.0, wer_score))

        # 3. CER指标 (0-∞，越低越好)
        cer_metric = max(0.0, 1.0 - min(1.0, cer_score))

        # 4. 准确率指标
        accuracy_metric = accuracy_score

        # 权重分配
        if accuracy_score > 0:
            # 有参考文本
            weights = [0.4, 0.25, 0.2, 0.15]
            metrics = [similarity_metric, wer_metric, cer_metric, accuracy_metric]
        else:
            # 无参考文本
            weights = [0.5, 0.3, 0.2]
            metrics = [similarity_metric, wer_metric, cer_metric]

        # 加权综合
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


# ==================== 文本相似度工具 ====================
class TextSimilarity:
    """文本相似度计算工具"""

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """计算Levenshtein编辑距离"""
        if len(s1) < len(s2):
            return TextSimilarity.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)

        for i, c1 in enumerate(s1):
            current_row = [i + 1]

            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)

                current_row.append(min(insertions, deletions, substitutions))

            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def normalized_edit_similarity(s1: str, s2: str) -> float:
        """归一化编辑相似度"""
        if not s1 or not s2:
            return 0.0

        distance = TextSimilarity.levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))

        return 1.0 - distance / max_len

    @staticmethod
    def jaccard_similarity(s1: str, s2: str) -> float:
        """Jaccard相似度（基于字符集合）"""
        if not s1 or not s2:
            return 0.0

        set1 = set(s1)
        set2 = set(s2)

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0


# ==================== 测试代码 ====================
def test_stability_checker():
    """测试稳定性检查器"""
    print("=== 测试稳定性检查器 ===")

    # 创建稳定性检查器
    checker = StabilityChecker()

    # 测试用例1：创建模拟音频
    print("\n1. 生成测试音频并检查稳定性...")

    try:
        from asr_engine import create_test_audio

        # 生成清晰的模拟语音
        audio = create_test_audio(
            duration=3.0,
            audio_type="speech_like"
        )

        # 检查稳定性
        metrics = checker.check_stability(audio, sample_rate=16000)

        print(f"   稳定性分数: {metrics.consistency_score:.3f}")
        print(f"   文本相似度: {metrics.similarity_score:.3f}")
        print(f"   词错误率(WER): {metrics.wer_score:.3f}")
        print(f"   字错误率(CER): {metrics.cer_score:.3f}")

    except ImportError:
        print("   跳过音频测试（需要asr_engine）")

    # 测试用例2：直接测试文本相似度计算
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

    # 测试用例3：WER/CER计算
    print("\n3. 测试WER/CER计算:")

    wer_test = wer("请打开车窗", "请打开窗")
    cer_test = cer("请打开车窗", "请打开窗")

    print(f"   WER('请打开车窗', '请打开窗') = {wer_test:.3f}")
    print(f"   CER('请打开车窗', '请打开窗') = {cer_test:.3f}")

    # 清理
    checker.cleanup()

    return metrics if 'metrics' in locals() else None


def test_with_real_audio_file():
    """使用真实音频文件测试稳定性"""
    print("\n=== 使用真实音频测试 ===")

    try:
        import librosa

        # 加载之前测试用的音频文件
        audio, sr = librosa.load("test.wav", sr=16000, mono=True)

        print(f"加载音频: {len(audio) / sr:.1f}秒")

        # 创建检查器
        checker = StabilityChecker()

        # 检查稳定性
        metrics = checker.check_stability(audio, sample_rate=sr)

        print(f"稳定性分析结果:")
        print(f"  综合稳定性分数: {metrics.consistency_score:.3f}")
        print(f"  文本相似度: {metrics.similarity_score:.3f}")
        print(f"  平均WER: {metrics.wer_score:.3f}")
        print(f"  平均CER: {metrics.cer_score:.3f}")

        # 判断稳定性
        stability = "高" if metrics.consistency_score > 0.7 else "中" if metrics.consistency_score > 0.4 else "低"
        print(f"  稳定性评估: {stability}")

        checker.cleanup()

        return metrics

    except Exception as e:
        print(f"真实音频测试失败: {e}")
        return None


def compare_stability_vs_confidence():
    """比较稳定性与置信度的关系"""
    print("\n=== 稳定性 vs 置信度比较 ===")

    try:
        from asr_engine import create_asr_engine, create_test_audio
        from confidence_analyzer import ConfidenceAnalyzer

        # 创建组件
        engine = create_asr_engine(model_size="tiny")
        confidence_analyzer = ConfidenceAnalyzer()
        stability_checker = StabilityChecker(asr_engine=engine)

        # 测试不同音频类型
        audio_types = ["silence", "noise", "tone", "speech_like"]

        results = []

        for audio_type in audio_types:
            print(f"\n测试音频类型: {audio_type}")

            # 生成音频
            audio = create_test_audio(duration=3.0, audio_type=audio_type)

            # 单次解码获取置信度
            asr_result = engine.transcribe(audio, sample_rate=16000)
            confidence_metrics = confidence_analyzer.analyze(asr_result)

            # 多次解码获取稳定性
            stability_metrics = stability_checker.check_stability(audio, sample_rate=16000)

            print(f"  置信度分数: {confidence_metrics.confidence_score:.3f}")
            print(f"  稳定性分数: {stability_metrics.consistency_score:.3f}")

            results.append({
                "type": audio_type,
                "confidence": confidence_metrics.confidence_score,
                "stability": stability_metrics.consistency_score
            })

        # 分析关系
        print("\n关系分析:")
        for result in results:
            diff = abs(result["confidence"] - result["stability"])
            relation = "一致" if diff < 0.3 else "不一致"
            print(f"  {result['type']}: 置信度={result['confidence']:.3f}, "
                  f"稳定性={result['stability']:.3f} ({relation})")

        engine.cleanup()

    except Exception as e:
        print(f"比较测试失败: {e}")


if __name__ == "__main__":
    print("稳定性测试模块")
    print("=" * 50)

    # 安装必要依赖
    print("注意：需要安装 jiwer 库")
    print("安装命令: pip install jiwer")
    print("=" * 50)

    # 运行测试
    try:
        import jiwer

        print("✓ jiwer 库已安装")

        test_stability_checker()

        # 如果存在test.wav，测试真实音频
        import os

        if os.path.exists("test.wav"):
            test_with_real_audio_file()

        # 比较测试
        compare_stability_vs_confidence()

    except ImportError:
        print("✗ 需要安装 jiwer 库: pip install jiwer")
        print("\n模拟测试（无jiwer）:")


        # 模拟WER/CER计算
        def simple_wer(s1, s2):
            """简化的WER计算"""
            words1 = s1.split() if s1 else []
            words2 = s2.split() if s2 else []

            if not words1 or not words2:
                return 1.0

            # 简单匹配
            matches = sum(1 for w1, w2 in zip(words1, words2) if w1 == w2)
            return 1.0 - matches / max(len(words1), len(words2))


        test_texts = ["请打开车窗", "请打开窗", "关闭车窗"]
        print(f"测试文本: {test_texts}")

        for i in range(len(test_texts)):
            for j in range(i + 1, len(test_texts)):
                wer_val = simple_wer(test_texts[i], test_texts[j])
                print(f"  WER('{test_texts[i]}', '{test_texts[j]}') ≈ {wer_val:.3f}")

    print("\n" + "=" * 50)
    print("测试完成！")