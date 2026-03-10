import logging
from core.protocol import SystemContext
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
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
    def __init__(self, module_id: str):
        self.module_id = module_id
        self.logger = logging.getLogger(f"VGuard.Module{module_id}")

    def setup(self):
        """子模块加载权重或模型"""
        pass

    @abstractmethod
    def detect(self, ctx: Any) -> DetectionResult:
        """
        核心检测接口：现在统一声明返回 DetectionResult。
        注：ctx 设为 Any 是为了兼容不同模块传入的参数差异。
        """
        pass