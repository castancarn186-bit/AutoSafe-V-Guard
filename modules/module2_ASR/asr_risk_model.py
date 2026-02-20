# asr_risk_model.py
"""
ASR风险融合模块
将置信度和稳定性合并为最终的风险分数
"""
import os
import sys

# --- 关键修复：动态添加当前目录到搜索路径 ---
# 这确保了无论从哪个目录启动，它都能找到同目录下的 confidence_analyzer
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 修复 OpenMP DLL 冲突（双重保险）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 原有的 import 保持不变
import time
from typing import Dict, Any
# 现在下面这两行就不会报错了
from confidence_analyzer import ConfidenceMetrics, ConfidenceAnalyzer
from stability_checker import StabilityMetrics, StabilityChecker
from asr_engine import ASREngine
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from pydantic import BaseModel
import logging
from confidence_analyzer import ConfidenceMetrics, ConfidenceAnalyzer
from stability_checker import StabilityMetrics, StabilityChecker
from asr_engine import ASRResult, ASREngine, create_asr_engine

logger = logging.getLogger(__name__)


class ASRRiskMetrics(BaseModel):
    """ASR风险指标"""
    text:str=""
    confidence_score: float  # 置信度分数 [0,1]，1表示高置信度
    stability_score: float  # 稳定性分数 [0,1]，1表示高稳定性
    asr_risk_score: float  # ASR风险分数 [0,1]，1表示高风险

    # 原始指标（用于调试和分析）
    raw_confidence: Optional[ConfidenceMetrics] = None
    raw_stability: Optional[StabilityMetrics] = None

    def __str__(self):
        risk_level = "低" if self.asr_risk_score < 0.3 else "中" if self.asr_risk_score < 0.7 else "高"
        return (f"ASR风险分数: {self.asr_risk_score:.3f} ({risk_level}风险) "
                f"[置信度: {self.confidence_score:.3f}, "
                f"稳定性: {self.stability_score:.3f}]")


@dataclass
class ASRRiskConfig:
    """ASR风险配置"""
    # 权重参数（来自项目文档：asr_risk = α*(1-confidence_score) + β*(1-stability_score)）
    alpha: float = 0.6  # 置信度权重
    beta: float = 0.4  # 稳定性权重

    # 阈值参数
    confidence_threshold: float = 0.5  # 置信度阈值
    stability_threshold: float = 0.5  # 稳定性阈值
    risk_threshold_low: float = 0.3  # 低风险阈值
    risk_threshold_high: float = 0.7  # 高风险阈值

    # 模块配置
    confidence_analyzer_config: Optional[Dict] = None
    stability_checker_config: Optional[Dict] = None

    def __post_init__(self):
        # 确保权重和为1
        total = self.alpha + self.beta
        if total != 1.0:
            self.alpha = self.alpha / total
            self.beta = self.beta / total

        # 默认配置
        if self.confidence_analyzer_config is None:
            self.confidence_analyzer_config = {"low_conf_threshold": 0.5}

        if self.stability_checker_config is None:
            self.stability_checker_config = {"num_decodings": 3}


class ASRRiskModel:
    """ASR风险模型"""

    def __init__(self,
                 config: Optional[ASRRiskConfig] = None,
                 asr_engine: Optional[ASREngine] = None):
        """
        初始化ASR风险模型

        Args:
            config: 风险配置
            asr_engine: ASR引擎实例，如果为None则创建新的
        """
        self.config = config or ASRRiskConfig()

        # 创建子模块
        self.confidence_analyzer = ConfidenceAnalyzer(
            **self.config.confidence_analyzer_config
        )

        self.stability_checker = StabilityChecker(
            config=self.config.stability_checker_config,
            asr_engine=asr_engine
        )

        logger.info(f"ASR风险模型初始化完成: α={self.config.alpha:.2f}, β={self.config.beta:.2f}")

    def compute_risk(self,
                     audio: np.ndarray,
                     sample_rate: int = 16000,
                     reference_text: Optional[str] = None) -> ASRRiskMetrics:
        """
        计算音频的ASR风险

        Args:
            audio: 音频数据
            sample_rate: 采样率
            reference_text: 参考文本（如果有），用于稳定性测试

        Returns:
            ASRRiskMetrics: ASR风险指标
        """
        try:
            # 0. 音频检查
            if len(audio) == 0:
                logger.warning("音频为空")
                return self._create_high_risk_metrics("音频为空")

            # 1. 单次解码获取ASR结果
            logger.info("执行ASR转录...")
            asr_result = self.stability_checker.engine.transcribe(
                audio,
                sample_rate=sample_rate,
                temperature=0.0,  # 确定性解码
                beam_size=5
            )

            if not asr_result.success:
                logger.warning("ASR转录失败")
                return self._create_high_risk_metrics("ASR转录失败")

            # 2. 分析置信度
            logger.info("分析置信度...")
            confidence_metrics = self.confidence_analyzer.analyze(asr_result)
            confidence_score = confidence_metrics.confidence_score

            # 3. 检查稳定性
            logger.info("检查稳定性...")
            stability_metrics = self.stability_checker.check_stability(
                audio,
                sample_rate=sample_rate,
                reference_text=reference_text
            )
            stability_score = stability_metrics.consistency_score

            # 4. 计算风险分数（根据项目文档公式）
            # asr_risk = α*(1-confidence_score) + β*(1-stability_score)
            asr_risk = (
                    self.config.alpha * (1.0 - confidence_score) +
                    self.config.beta * (1.0 - stability_score)
            )

            # 确保在[0,1]范围内
            asr_risk = max(0.0, min(1.0, asr_risk))

            # 5. 构建结果
            metrics = ASRRiskMetrics(
                text=asr_result.text,
                confidence_score=confidence_score,
                stability_score=stability_score,
                asr_risk_score=asr_risk,
                raw_confidence=confidence_metrics,
                raw_stability=stability_metrics
            )

            logger.info(f"ASR风险计算完成: {metrics}")

            return metrics

        except Exception as e:
            logger.error(f"ASR风险计算失败: {e}")
            return self._create_high_risk_metrics(f"计算失败: {str(e)}")

    def compute_risk_from_asr_result(self,
                                     asr_result: ASRResult,
                                     audio: np.ndarray,
                                     sample_rate: int = 16000) -> ASRRiskMetrics:
        """
        从已有的ASR结果计算风险

        Args:
            asr_result: 已有的ASR结果
            audio: 原始音频（用于稳定性测试）
            sample_rate: 采样率

        Returns:
            ASRRiskMetrics: ASR风险指标
        """
        try:
            # 1. 分析置信度
            confidence_metrics = self.confidence_analyzer.analyze(asr_result)
            confidence_score = confidence_metrics.confidence_score

            # 2. 检查稳定性
            stability_metrics = self.stability_checker.check_stability(
                audio,
                sample_rate=sample_rate
            )
            stability_score = stability_metrics.consistency_score

            # 3. 计算风险分数
            asr_risk = (
                    self.config.alpha * (1.0 - confidence_score) +
                    self.config.beta * (1.0 - stability_score)
            )
            asr_risk = max(0.0, min(1.0, asr_risk))

            # 4. 构建结果
            metrics = ASRRiskMetrics(
                confidence_score=confidence_score,
                stability_score=stability_score,
                asr_risk_score=asr_risk,
                raw_confidence=confidence_metrics,
                raw_stability=stability_metrics
            )

            return metrics

        except Exception as e:
            logger.error(f"从ASR结果计算风险失败: {e}")
            return self._create_high_risk_metrics(f"计算失败: {str(e)}")

    def _create_high_risk_metrics(self, reason: str = "") -> ASRRiskMetrics:
        """创建高风险指标"""
        logger.warning(f"返回高风险指标: {reason}")

        return ASRRiskMetrics(
            confidence_score=0.0,
            stability_score=0.0,
            asr_risk_score=1.0,  # 高风险
            raw_confidence=None,
            raw_stability=None
        )

    def _create_low_risk_metrics(self) -> ASRRiskMetrics:
        """创建低风险指标"""
        return ASRRiskMetrics(
            confidence_score=1.0,
            stability_score=1.0,
            asr_risk_score=0.0,  # 低风险
            raw_confidence=None,
            raw_stability=None
        )

    def get_risk_level(self, risk_score: float) -> str:
        """获取风险等级"""
        if risk_score < self.config.risk_threshold_low:
            return "低风险"
        elif risk_score < self.config.risk_threshold_high:
            return "中风险"
        else:
            return "高风险"

    def should_accept(self, risk_score: float) -> Tuple[bool, str]:
        """
        根据风险分数决定是否接受

        Returns:
            (是否接受, 原因)
        """
        if risk_score < self.config.risk_threshold_low:
            return True, "低风险，可接受"
        elif risk_score < self.config.risk_threshold_high:
            return False, "中风险，需要人工确认"
        else:
            return False, "高风险，应拒绝"

    def cleanup(self):
        """清理资源"""
        self.stability_checker.cleanup()


# ==================== 风险分析工具 ====================
class RiskAnalyzer:
    """风险分析工具类"""

    @staticmethod
    def analyze_risk_pattern(risk_scores: List[float]) -> Dict:
        """分析风险模式"""
        if not risk_scores:
            return {"error": "无数据"}

        risk_array = np.array(risk_scores)

        return {
            "mean": float(np.mean(risk_array)),
            "std": float(np.std(risk_array)),
            "min": float(np.min(risk_array)),
            "max": float(np.max(risk_array)),
            "q1": float(np.percentile(risk_array, 25)),
            "median": float(np.median(risk_array)),
            "q3": float(np.percentile(risk_array, 75)),
            "low_risk_ratio": float(np.sum(risk_array < 0.3) / len(risk_array)),
            "high_risk_ratio": float(np.sum(risk_array > 0.7) / len(risk_array))
        }

    @staticmethod
    def find_risk_correlation(confidence_scores: List[float],
                              stability_scores: List[float]) -> Dict:
        """查找置信度与稳定性的相关性"""
        if len(confidence_scores) != len(stability_scores) or len(confidence_scores) < 2:
            return {"error": "数据不足"}

        conf_array = np.array(confidence_scores)
        stab_array = np.array(stability_scores)

        # 计算相关性
        correlation = np.corrcoef(conf_array, stab_array)[0, 1]

        return {
            "correlation": float(correlation),
            "interpretation": RiskAnalyzer._interpret_correlation(correlation)
        }

    @staticmethod
    def _interpret_correlation(corr: float) -> str:
        """解释相关性系数"""
        if corr > 0.7:
            return "强正相关"
        elif corr > 0.3:
            return "中等正相关"
        elif corr > -0.3:
            return "弱相关或无相关"
        elif corr > -0.7:
            return "中等负相关"
        else:
            return "强负相关"


# ==================== 测试代码 ====================
def test_asr_risk_model():
    """测试ASR风险模型"""
    print("=== 测试ASR风险模型 ===")

    # 创建风险模型
    config = ASRRiskConfig(
        alpha=0.6,  # 置信度权重60%
        beta=0.4  # 稳定性权重40%
    )

    risk_model = ASRRiskModel(config=config)

    # 测试用例1：使用真实音频文件
    print("\n1. 测试真实音频文件 (test.wav):")

    try:
        import librosa

        if not librosa.__version__:
            raise ImportError

        # 加载音频
        audio, sr = librosa.load("test.wav", sr=16000, mono=True)
        print(f"  加载音频: {len(audio) / sr:.1f}秒")

        # 计算风险
        risk_metrics = risk_model.compute_risk(audio, sample_rate=sr)

        print(f"  结果: {risk_metrics}")
        print(f"  风险等级: {risk_model.get_risk_level(risk_metrics.asr_risk_score)}")

        # 决策
        accept, reason = risk_model.should_accept(risk_metrics.asr_risk_score)
        print(f"  决策: {'接受' if accept else '拒绝'} - {reason}")

        # 显示详细指标
        if risk_metrics.raw_confidence:
            print(f"\n  详细置信度指标:")
            print(f"    平均概率: {risk_metrics.raw_confidence.mean_linear_prob:.3f}")
            print(f"    低置信token比例: {risk_metrics.raw_confidence.low_conf_ratio:.1%}")

        if risk_metrics.raw_stability:
            print(f"\n  详细稳定性指标:")
            print(f"    文本相似度: {risk_metrics.raw_stability.similarity_score:.3f}")
            print(f"    平均WER: {risk_metrics.raw_stability.wer_score:.3f}")

    except ImportError:
        print("  跳过（需要librosa库）")
    except FileNotFoundError:
        print("  跳过（test.wav不存在）")

    # 测试用例2：模拟数据测试
    print("\n2. 模拟数据测试:")

    test_cases = [
        ("高置信高稳定", 0.9, 0.8, 0.22),  # α*(1-0.9) + β*(1-0.8) = 0.6*0.1 + 0.4*0.2 = 0.06+0.08=0.14
        ("高置信低稳定", 0.9, 0.3, 0.46),  # 0.6*0.1 + 0.4*0.7 = 0.06+0.28=0.34
        ("低置信高稳定", 0.3, 0.8, 0.46),  # 0.6*0.7 + 0.4*0.2 = 0.42+0.08=0.50
        ("低置信低稳定", 0.3, 0.3, 0.70),  # 0.6*0.7 + 0.4*0.7 = 0.42+0.28=0.70
    ]

    for name, conf, stab, expected in test_cases:
        # 手动计算风险
        risk = config.alpha * (1.0 - conf) + config.beta * (1.0 - stab)
        risk = max(0.0, min(1.0, risk))

        level = risk_model.get_risk_level(risk)
        accept, reason = risk_model.should_accept(risk)

        print(f"  {name}: 置信度={conf:.1f}, 稳定性={stab:.1f}")
        print(f"    计算风险: {risk:.3f} (预期: {expected:.2f}, 实际: {risk:.2f})")
        print(f"    风险等级: {level}")
        print(f"    决策: {'接受' if accept else '拒绝'} - {reason}")

    # 清理
    risk_model.cleanup()

    return risk_metrics if 'risk_metrics' in locals() else None


def test_different_weights():
    """测试不同权重配置"""
    print("\n=== 测试不同权重配置 ===")

    # 固定测试数据
    confidence = 0.6
    stability = 0.7

    weight_configs = [
        ("偏重置信度", 0.8, 0.2),
        ("偏重稳定性", 0.2, 0.8),
        ("均衡", 0.5, 0.5),
        ("项目默认", 0.6, 0.4),
    ]

    for name, alpha, beta in weight_configs:
        config = ASRRiskConfig(alpha=alpha, beta=beta)
        risk = config.alpha * (1.0 - confidence) + config.beta * (1.0 - stability)

        print(f"  {name} (α={alpha:.1f}, β={beta:.1f}):")
        print(f"    置信度={confidence:.1f}, 稳定性={stability:.1f}")
        print(f"    风险分数 = {alpha:.1f}*(1-{confidence:.1f}) + {beta:.1f}*(1-{stability:.1f}) = {risk:.3f}")


def test_batch_processing():
    """测试批量处理"""
    print("\n=== 测试批量处理 ===")

    try:
        from asr_engine import create_test_audio

        # 创建风险模型
        risk_model = ASRRiskModel()

        # 生成多种测试音频
        audio_types = ["silence", "noise", "tone", "speech_like"]
        results = []

        for audio_type in audio_types:
            print(f"\n处理音频类型: {audio_type}")

            # 生成音频
            audio = create_test_audio(duration=2.0, audio_type=audio_type)

            # 计算风险
            risk_metrics = risk_model.compute_risk(audio, sample_rate=16000)

            results.append({
                "type": audio_type,
                "risk": risk_metrics.asr_risk_score,
                "confidence": risk_metrics.confidence_score,
                "stability": risk_metrics.stability_score
            })

            print(f"  风险分数: {risk_metrics.asr_risk_score:.3f}")
            print(f"  置信度: {risk_metrics.confidence_score:.3f}")
            print(f"  稳定性: {risk_metrics.stability_score:.3f}")

        # 分析结果
        print("\n批量处理分析:")
        risk_scores = [r["risk"] for r in results]
        conf_scores = [r["confidence"] for r in results]
        stab_scores = [r["stability"] for r in results]

        risk_analysis = RiskAnalyzer.analyze_risk_pattern(risk_scores)
        correlation = RiskAnalyzer.find_risk_correlation(conf_scores, stab_scores)

        print(f"  风险统计: 均值={risk_analysis['mean']:.3f}, 标准差={risk_analysis['std']:.3f}")
        print(f"  相关性: {correlation}")

        risk_model.cleanup()

    except Exception as e:
        print(f"批量处理测试失败: {e}")


def create_demo_scenarios():
    """创建演示场景"""
    print("\n=== 演示场景 ===")

    scenarios = [
        {
            "name": "清晰语音指令",
            "description": "清晰的'打开车窗'指令",
            "expected_confidence": 0.85,
            "expected_stability": 0.80,
            "expected_risk": "低"
        },
        {
            "name": "噪声环境",
            "description": "有背景噪声的语音",
            "expected_confidence": 0.60,
            "expected_stability": 0.50,
            "expected_risk": "中"
        },
        {
            "name": "模糊语音",
            "description": "音量小或发音不清",
            "expected_confidence": 0.30,
            "expected_stability": 0.40,
            "expected_risk": "高"
        },
        {
            "name": "非语音声音",
            "description": "音乐、噪声等非语音",
            "expected_confidence": 0.10,
            "expected_stability": 0.20,
            "expected_risk": "高"
        }
    ]

    config = ASRRiskConfig(alpha=0.6, beta=0.4)

    for scenario in scenarios:
        conf = scenario["expected_confidence"]
        stab = scenario["expected_stability"]
        risk = config.alpha * (1.0 - conf) + config.beta * (1.0 - stab)

        risk_model = ASRRiskModel(config=config)
        level = risk_model.get_risk_level(risk)
        accept, reason = risk_model.should_accept(risk)

        print(f"\n{scenario['name']}:")
        print(f"  描述: {scenario['description']}")
        print(f"  预期置信度: {conf:.2f}, 稳定性: {stab:.2f}")
        print(f"  计算风险: {risk:.3f} ({level})")
        print(f"  决策: {'✓ 接受' if accept else '✗ 拒绝'} - {reason}")

        risk_model.cleanup()


if __name__ == "__main__":
    print("ASR风险融合模块")
    print("=" * 60)
    print("项目公式: asr_risk = α*(1-confidence_score) + β*(1-stability_score)")
    print("=" * 60)

    # 运行测试
    test_asr_risk_model()

    test_different_weights()

    test_batch_processing()

    create_demo_scenarios()

    print("\n" + "=" * 60)
    print("✓ ASR风险融合模块测试完成")
    print("\n项目第二部分完成进度:")
    print("  1. asr_engine.py ✓ - ASR推理封装")
    print("  2. confidence_analyzer.py ✓ - 置信度分析")
    print("  3. stability_checker.py ✓ - 稳定性测试")
    print("  4. asr_risk_model.py ✓ - 风险融合")
    print("\n🎯 第二部分：ASR行为分析与不确定性量化 ✓ 完成！")