# modules/module3_semantic/detector.py
import logging
from core.base_module import BaseDetector
from core.protocol import SystemContext
from modules.module3_semantic.core.protocol import SemanticInput, VehicleContext, Language, WeatherCondition
from modules.module3_semantic.models.reasoning import SemanticSafetyEngine

class SemanticDetector(BaseDetector):
    def __init__(self):
        # 基类 __init__ 会自动调用 setup()
        super().__init__(module_id="C")
        self.logger = logging.getLogger("VGuard.ModuleC")

    def setup(self):
        """实现基类抽象方法：加载语义引擎"""
        try:
            self.engine = SemanticSafetyEngine()
            self.logger.info("SemanticSafetyEngine (RAG+DeepLearning) 加载成功")
        except Exception as e:
            self.logger.error(f"语义引擎加载失败: {e}")
            raise  # 抛出异常后基类会将 self.enabled 设为 False

    def detect(self, ctx: SystemContext) -> tuple[float, str, str, dict]:
        """
        实现基类抽象方法
        :param ctx: 统一系统上下文，包含 asr_text、speed、weather、has_pedestrians 等
        :return: (risk_score, decision, reason, evidence)
                 decision 必须是 "PASS"/"REVIEW"/"BLOCK"
                 evidence 会被放入最终 DetectionResult 的 metadata 中
        """
        # 引擎未就绪时返回保守值
        if not hasattr(self, 'engine') or self.engine is None:
            return 0.5, "REVIEW", "语义引擎未就绪", {}

        try:
            # 提取 ASR 文本
            text = ctx.asr_text
            speed_val = float(ctx.speed)
            weather_str = ctx.weather.lower() if ctx.weather else "sunny"

            # 映射天气枚举
            weather_enum = WeatherCondition.SUNNY
            if "rain" in weather_str:
                weather_enum = WeatherCondition.RAINY
            elif "fog" in weather_str:
                weather_enum = WeatherCondition.FOGGY
            elif "snow" in weather_str:
                weather_enum = WeatherCondition.SNOWY

            # 推断档位（SystemContext 暂无 gear 字段，暂用速度推断）
            gear = "D" if speed_val > 0 else "P"

            # 构造内部 VehicleContext（缺失字段使用默认值）
            veh_ctx = VehicleContext(
                speed=speed_val,
                speed_limit=120.0,                     # 可后续从 ctx 扩展
                gear=gear,
                weather=weather_enum,
                traffic_density="low",                  # 可后续从 ctx 扩展
                has_pedestrians=bool(ctx.has_pedestrians)
            )

            # 构造语义输入
            sem_input = SemanticInput(
                text=text,
                language=Language.ZH,
                context=veh_ctx
            )

            # 核心推理
            report = self.engine.evaluate(sem_input)

            # 映射决策
            decision_map = {"SAFE": "PASS", "WARNING": "REVIEW", "DANGER": "BLOCK"}
            decision = decision_map.get(report.level.value, "PASS")

            # 构造 evidence（将放入 metadata）
            evidence = {
                "intent_category": report.intent_category.value if hasattr(report, 'intent_category') else None,
                "matched_vector_id": report.matched_vector_id if hasattr(report, 'matched_vector_id') else None
            }

            return float(report.risk_score), decision, report.reason, evidence

        except Exception as e:
            self.logger.error(f"Semantic 模块崩溃: {e}", exc_info=True)
            # 语义层是最后一道防线，崩溃时必须给高分并建议 BLOCK
            return 0.9, "BLOCK", f"语义分析异常: {str(e)}", {}