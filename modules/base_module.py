# core/base_module.py
import time
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Dict

#从 protocol 导入数据定义
from core.protocol import SystemContext, RiskReport, DecisionType


class BaseDetector(ABC):
    def __init__(self, module_id: str):
        self.module_id = module_id
        self.logger = logging.getLogger(f"VGuard.{module_id}")
        self.enabled = True

        try:
            self.setup()
            self.logger.info("模块初始化成功")
        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            self.enabled = False

    @abstractmethod
    def setup(self):
        """加载模型权重"""
        pass

    @abstractmethod
    def detect(self, ctx: SystemContext) -> Tuple[float, str, str, Dict]:
        """
        子类只需实现这个核心逻辑
        Returns:
            (risk_score, suggestion, reason, evidence)
        """
        pass

    def run(self, ctx: SystemContext) -> RiskReport:
        """
        主控入口：负责计时、错误捕获、打包
        """
        start_t = time.perf_counter()

        # 默认安全值
        score = 0.0
        suggestion = DecisionType.PASS.value
        reason = "Normal"
        evidence = {}

        if self.enabled:
            try:
                # 调用子类的逻辑
                score, suggestion, reason, evidence = self.detect(ctx)
            except Exception as e:
                self.logger.error(f"检测运行时错误: {e}", exc_info=True)
                reason = f"Runtime Error: {str(e)}"

        # 计算耗时
        latency = (time.perf_counter() - start_t) * 1000

        # 打包成统一格式返回
        return RiskReport(
            module_id=self.module_id,
            risk_score=score,
            suggestion=suggestion,
            reason=reason,
            evidence=evidence,
            latency_ms=round(latency, 2)
        )