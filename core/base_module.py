import logging
from core.protocol import SystemContext, RiskReport
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class DetectionResult:
    """统一风险报告协议"""
    module_id: str           # 'A', 'B', 'C'
    risk_score: float        # 0.0 ~ 1.0
    decision: str            # 'PASS', 'REVIEW', 'BLOCK'
    reason: str              # 拦截理由
    latency_ms: float = 0.0  # 耗时
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return self.__dict__

class BaseDetector(ABC):
    """检测器基类协议"""
    def __init__(self, module_id: str):
        self.module_id = module_id
        self.logger = logging.getLogger(f"VGuard.Module{module_id}")

    def setup(self):
        """可选的初始化逻辑（如加载权重），子模块可重写"""
        pass

    @abstractmethod
    def detect(self, ctx: SystemContext) -> RiskReport:
        """核心检测接口：必须由子模块实现"""
        pass