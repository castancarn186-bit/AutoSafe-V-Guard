import logging
from core.protocol import SystemContext
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from dataclasses import dataclass, field
import time

@dataclass
class DetectionResult:
    """全系统统一的风险报告协议（一等奖级标准）"""
    module_id: str           # 'A', 'B', 'C'
    risk_score: float        # 0.0 ~ 1.0
    decision: str            # 'PASS', 'REVIEW', 'BLOCK'
    reason: str              # 拦截理由
    latency_ms: float = 0.0  # 耗时
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return self.__dict__

class BaseDetector(ABC):
    """检测器基类协议"""
    def __init__(self, module_id: str, config: dict = None):
        self.module_id = module_id
        self.config = config or {}
        self.logger = logging.getLogger(f"VGuard.Module{module_id}")
        self.enabled = True
        try:
            self.setup()
            self.logger.info("模块初始化成功")
        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            self.enabled = False

    def setup(self):
        """子模块加载权重或模型"""
        pass

    @abstractmethod
    def detect(self, ctx: SystemContext) -> Tuple[float, str, str, Dict[str, Any]]:
        """
        核心检测接口：返回 (risk_score, decision, reason, evidence)
        """
        pass

    def run(self, ctx: SystemContext) -> DetectionResult:
        """主控入口：负责计时、错误捕获、打包"""
        start_t = time.perf_counter()
        score = 0.0
        decision = "PASS"
        reason = "Normal"
        evidence = {}

        if self.enabled:
            try:
                score, decision, reason, evidence = self.detect(ctx)
            except Exception as e:
                self.logger.error(f"检测运行时错误: {e}", exc_info=True)
                reason = f"Runtime Error: {str(e)}"

        latency = (time.perf_counter() - start_t) * 1000

        return DetectionResult(
            module_id=self.module_id,
            risk_score=score,
            decision=decision,
            reason=reason,
            metadata=evidence,
            latency_ms=round(latency, 2)
        )