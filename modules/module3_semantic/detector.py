# modules/module3_semantic/detector.py
import time
import os
import logging
from core.base_module import BaseDetector, DetectionResult

# 💡 关键：引入模块 C 内部的模型
from modules.module3_semantic.core.protocol import SemanticInput, VehicleContext, Language, WeatherCondition
from modules.module3_semantic.models.reasoning import SemanticSafetyEngine


class SemanticDetector(BaseDetector):
    def __init__(self):
        super().__init__(module_id="C")
        self.logger = logging.getLogger("VGuard.ModuleC")
        try:
            # 实例化最强大脑
            self.engine = SemanticSafetyEngine()
            self.logger.info("SemanticSafetyEngine (RAG+DeepLearning) 加载成功")
        except Exception as e:
            self.logger.error(f"语义引擎加载失败: {e}")
            self.engine = None

    def detect(self, text: str, context: dict = None) -> DetectionResult:
        """
        text: ASR 识别出的文本
        context: 包含 speed, gear, weather 等字段的字典
        """
        start_time = time.perf_counter()
        context = context or {}

        if self.engine is None:
            return DetectionResult("C", 0.5, "REVIEW", "语义引擎未就绪")

        try:
            # 🚀 适配层：手动清洗数据，确保符合 C 模块内部的 Pydantic 校验 (cite: 801, 803)
            speed_val = float(context.get("speed", 0.0))
            weather_str = context.get("weather", "sunny").lower()

            # 映射天气枚举
            weather_enum = WeatherCondition.SUNNY
            if "rain" in weather_str:
                weather_enum = WeatherCondition.RAINY
            elif "fog" in weather_str:
                weather_enum = WeatherCondition.FOGGY
            elif "snow" in weather_str:
                weather_enum = WeatherCondition.SNOWY

            # 1. 构造内部 VehicleContext 对象
            # 注意：补齐所有必填字段，防止 Pydantic 报错
            veh_ctx = VehicleContext(
                speed=speed_val,
                speed_limit=float(context.get("speed_limit", 120.0)),
                gear=context.get("gear", "D" if speed_val > 0 else "P"),
                weather=weather_enum,
                traffic_density=context.get("traffic_density", "low"),
                has_pedestrians=bool(context.get("has_pedestrians", False))
            )

            # 2. 构造语义输入
            sem_input = SemanticInput(
                text=text,
                language=Language.ZH,
                context=veh_ctx
            )

            # 3. 核心推理 (cite: 786)
            report = self.engine.evaluate(sem_input)

            # 4. 映射决策逻辑
            decision_map = {"SAFE": "PASS", "WARNING": "REVIEW", "DANGER": "BLOCK"}
            decision = decision_map.get(report.level.value, "PASS")

            latency = (time.perf_counter() - start_time) * 1000

            # 🚀 返回全系统统一协议
            return DetectionResult(
                module_id=self.module_id,
                risk_score=float(report.risk_score),
                decision=decision,
                reason=report.reason,
                latency_ms=round(latency, 2),
                metadata={"category": report.intent_category.value}
            )

        except Exception as e:
            self.logger.error(f"Semantic 模块崩溃: {e}")
            # 语义层是最后一道防线，崩溃时必须给高分并建议 BLOCK
            return DetectionResult(self.module_id, 0.9, "BLOCK", f"语义分析异常: {str(e)}")