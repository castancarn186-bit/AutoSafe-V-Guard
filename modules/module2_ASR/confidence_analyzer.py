
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel
import logging
from modules.module2_ASR.asr_engine import ASRResult

logger = logging.getLogger(__name__)


class ConfidenceMetrics(BaseModel):
    """置信度指标"""
    mean_log_prob: float  # 平均对数概率
    mean_linear_prob: float  # 平均线性概率
    low_conf_ratio: float  # 低置信度token比例
    prob_variance: float  # 概率方差
    prob_entropy: float  # 概率熵
    confidence_score: float  # 综合置信度分数 [0,1]，1表示高置信度
    token_count: int = 0  # token数量
    valid_token_count: int = 0  # 有效token数量

    def __str__(self):
        level = "高" if self.confidence_score > 0.7 else "中" if self.confidence_score > 0.3 else "低"
        return (f"置信度分数: {self.confidence_score:.3f} ({level}) "
                f"[平均概率: {self.mean_linear_prob:.3f}, "
                f"低置信token: {self.low_conf_ratio:.1%}]")


class ConfidenceAnalyzer:
    """置信度分析器"""

    def __init__(self,
                 low_conf_threshold: float = 0.5,
                 min_tokens: int = 1,
                 enable_debug: bool = False):
        """
        初始化置信度分析器

        Args:
            low_conf_threshold: 低置信度阈值 (0-1)
            min_tokens: 最少token数量
            enable_debug: 是否启用调试输出
        """
        self.low_conf_threshold = low_conf_threshold
        self.min_tokens = min_tokens
        self.enable_debug = enable_debug

    def analyze(self, asr_result: ASRResult) -> ConfidenceMetrics:
        """
        分析ASR结果的置信度 - 修复版

        Args:
            asr_result: ASR引擎的输出结果

        Returns:
            ConfidenceMetrics: 置信度指标
        """
        # 1. 检查ASR是否成功
        if not asr_result.success:
            if self.enable_debug:
                print("警告: ASR转录失败")
            return self._create_error_metrics("ASR转录失败")

        # 2. 检查是否有token
        if not asr_result.tokens:
            if self.enable_debug:
                print("警告: ASR结果中没有token")
            return self._create_error_metrics("无token")

        # 3. 检查log_probs
        if not asr_result.log_probs:
            if self.enable_debug:
                print("警告: ASR结果中没有log_probs")
            return self._create_error_metrics("无log_probs")

        # 4. 过滤有效对数概率
        # 有效的对数概率应该在(-50, 0)范围内
        # -50 对应概率 e^-50 ≈ 1.9e-22，几乎为0
        # 0 对应概率 1.0
        valid_log_probs = []
        valid_indices = []

        for i, log_p in enumerate(asr_result.log_probs):
            # 检查是否为有效的对数概率
            if isinstance(log_p, (int, float)) and -50 < log_p <= 0:
                valid_log_probs.append(log_p)
                valid_indices.append(i)
            else:
                if self.enable_debug:
                    print(f"调试: 过滤无效log_prob[{i}]={log_p}")

        # 5. 检查有效token数量
        if len(valid_log_probs) < self.min_tokens:
            if self.enable_debug:
                print(f"警告: 有效token数量不足 ({len(valid_log_probs)} < {self.min_tokens})")
            return self._create_error_metrics(f"有效token不足: {len(valid_log_probs)}")

        # 6. 转换为线性概率
        linear_probs = []
        for log_p in valid_log_probs:
            # 对数概率转线性概率: p = e^(log_p)
            # 处理数值稳定性
            if log_p < -30:  # 极低概率
                prob = 0.0
            elif log_p > -0.001:  # 非常接近1
                prob = 0.999  # 避免正好为1.0
            else:
                prob = np.exp(log_p)
            linear_probs.append(prob)

        # 7. 获取对应的token（用于调试）
        valid_tokens = [asr_result.tokens[i] for i in valid_indices] if asr_result.tokens else []

        # 8. 调试输出
        if self.enable_debug:
            print(f"\n=== 置信度分析调试 ===")
            print(f"ASR结果: '{asr_result.text}'")
            print(f"总token数: {len(asr_result.tokens)}")
            print(f"有效token数: {len(valid_log_probs)}")
            print(f"有效概率范围: [{min(linear_probs):.4f}, {max(linear_probs):.4f}]")
            print(f"有效概率均值: {np.mean(linear_probs):.4f}")
            print(f"有效概率中位数: {np.median(linear_probs):.4f}")
            print(f"有效概率标准差: {np.std(linear_probs):.4f}")

            # 显示前5个token的概率
            print(f"\nToken概率详情:")
            for i in range(min(5, len(valid_tokens))):
                token = valid_tokens[i]
                prob = linear_probs[i]
                log_p = valid_log_probs[i]
                conf_level = "高" if prob > 0.7 else "中" if prob > 0.3 else "低"
                print(f"  '{token}': prob={prob:.4f}, log={log_p:.4f} [{conf_level}]")
            print("=" * 40)

        # 9. 计算各种指标
        metrics = self._compute_metrics(linear_probs, valid_log_probs)

        # 10. 添加token计数信息
        metrics.token_count = len(asr_result.tokens)
        metrics.valid_token_count = len(valid_log_probs)

        logger.info(f"置信度分析完成: {metrics}")

        return metrics

    def _compute_metrics(self,
                         linear_probs: List[float],
                         log_probs: List[float]) -> ConfidenceMetrics:
        """计算所有置信度指标"""

        # 1. 平均对数概率
        mean_log_prob = float(np.mean(log_probs))

        # 2. 平均线性概率
        mean_linear_prob = float(np.mean(linear_probs))

        # 3. 低置信度token比例
        low_conf_ratio = self._calc_low_conf_ratio(linear_probs)

        # 4. 概率方差
        prob_variance = float(np.var(linear_probs)) if len(linear_probs) > 1 else 0.0

        # 5. 概率熵
        prob_entropy = self._calc_entropy(linear_probs)

        # 6. 综合置信度分数
        confidence_score = self._calc_confidence_score(linear_probs)

        return ConfidenceMetrics(
            mean_log_prob=mean_log_prob,
            mean_linear_prob=mean_linear_prob,
            low_conf_ratio=low_conf_ratio,
            prob_variance=prob_variance,
            prob_entropy=prob_entropy,
            confidence_score=confidence_score
        )

    def _calc_low_conf_ratio(self, probs: List[float]) -> float:
        """计算低置信度token比例"""
        if not probs:
            return 1.0

        low_count = sum(1 for p in probs if p < self.low_conf_threshold)
        return low_count / len(probs)

    def _calc_entropy(self, probs: List[float]) -> float:
        """计算概率分布的熵"""
        if not probs:
            return 0.0

        # 归一化概率
        probs_array = np.array(probs)
        probs_sum = probs_array.sum()

        if probs_sum <= 0:
            return 0.0

        probs_normalized = probs_array / probs_sum

        # 计算熵 H = -Σ p * log(p)
        # 添加小epsilon避免log(0)
        epsilon = 1e-10
        entropy = -np.sum(probs_normalized * np.log(probs_normalized + epsilon))

        return float(entropy)

    def _calc_confidence_score(self, probs: List[float]) -> float:
        """
        计算综合置信度分数 - 修复版

        修复问题：
        1. 避免概率为1.0导致的过度乐观
        2. 考虑概率分布的合理性
        3. 加入惩罚项

        算法：
        - 平均概率 (40%) - 越高越好，但上限0.95
        - 低置信度比例 (30%) - 越低越好
        - 概率方差 (20%) - 需要合理范围，太小或太大都不好
        - 概率熵 (10%) - 需要合理范围，太小或太大都不好
        """
        if not probs:
            return 0.0

        # 1. 平均概率指标 (权重0.4)
        avg_prob = np.mean(probs)
        # 避免过度乐观：如果平均概率 > 0.95，可能有问题
        if avg_prob > 0.95:
            # 检查是否所有概率都接近1.0
            if all(p > 0.99 for p in probs):
                # 这很可能是异常情况，降低置信度
                avg_prob_score = 0.5
            else:
                avg_prob_score = 0.95
        else:
            avg_prob_score = avg_prob

        # 2. 低置信度比例指标 (权重0.3)
        low_conf_ratio = self._calc_low_conf_ratio(probs)
        low_conf_score = 1.0 - low_conf_ratio

        # 3. 概率方差指标 (权重0.2)
        prob_variance = np.var(probs) if len(probs) > 1 else 0.0

        # 方差需要在合理范围内
        # 太小（<0.01）：可能所有概率都接近1.0或0.0
        # 太大（>0.25）：概率分布过于分散
        if prob_variance < 0.01:
            # 检查是否所有概率都接近1.0
            if all(p > 0.99 for p in probs):
                variance_score = 0.3  # 异常情况
            elif all(p < 0.01 for p in probs):
                variance_score = 0.2  # 全低概率
            else:
                variance_score = 0.5  # 可能是有效的小方差
        elif prob_variance > 0.25:
            variance_score = max(0.0, 1.0 - prob_variance)
        else:
            # 理想方差范围 [0.01, 0.25]
            # 0.1 附近最优
            variance_score = 1.0 - abs(prob_variance - 0.1) * 2
            variance_score = max(0.0, min(1.0, variance_score))

        # 4. 概率熵指标 (权重0.1)
        prob_entropy = self._calc_entropy(probs)

        # 熵也需要在合理范围内
        # 太小（<0.1）：概率分布过于集中
        # 太大（>2.0）：概率分布过于分散
        if prob_entropy < 0.1:
            if all(p > 0.99 for p in probs):
                entropy_score = 0.3  # 异常情况
            elif all(p < 0.01 for p in probs):
                entropy_score = 0.2  # 全低概率
            else:
                entropy_score = 0.5
        elif prob_entropy > 2.0:
            entropy_score = max(0.0, 1.0 - (prob_entropy - 2.0))
        else:
            # 理想熵范围 [0.1, 2.0]
            # 1.0 附近最优
            entropy_score = 1.0 - abs(prob_entropy - 1.0) * 0.5
            entropy_score = max(0.0, min(1.0, entropy_score))

        # 加权综合
        weights = [0.4, 0.3, 0.2, 0.1]
        scores = [avg_prob_score, low_conf_score, variance_score, entropy_score]

        confidence_score = sum(w * s for w, s in zip(weights, scores))

        # 确保在[0,1]范围内
        confidence_score = max(0.0, min(1.0, confidence_score))

        return confidence_score

    def _create_error_metrics(self, reason: str = "") -> ConfidenceMetrics:
        """创建错误指标"""
        if self.enable_debug:
            print(f"创建错误指标: {reason}")

        return ConfidenceMetrics(
            mean_log_prob=-100.0,
            mean_linear_prob=0.0,
            low_conf_ratio=1.0,
            prob_variance=0.0,
            prob_entropy=0.0,
            confidence_score=0.0,
            token_count=0,
            valid_token_count=0
        )

    def analyze_batch(self, asr_results: List[ASRResult]) -> List[ConfidenceMetrics]:
        """批量分析"""
        return [self.analyze(result) for result in asr_results]