# core/protocol.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import time

@dataclass
class DetectionResult:
    """全系统统一风险报告协议 (一等奖级标准)"""
    module_id: str           # 'A', 'B', 'C'
    risk_score: float        # 0.0 ~ 1.0
    decision: str            # 'PASS', 'REVIEW', 'BLOCK'
    reason: str              # 拦截或放行理由
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class SystemContext:
    """统一环境上下文数据"""
    audio_frame: Any         # 支持 numpy 矩阵或 bytes
    asr_text: str = ""
    speed: float = 0.0
    weather: str = "sunny"
    has_pedestrians: bool = False # 补全语义模型必填项
    is_night: bool = False
    timestamp: float = field(default_factory=time.time)