# core/protocol.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum
import time


# --- 1. 定义枚举常量 (避免队友手滑写错字符串) ---
class DecisionType(Enum):
    PASS = "PASS"  # 放行
    WARN = "WARN"  # 警告 (二次确认)
    BLOCK = "BLOCK"  # 拦截


class WeatherType(Enum):
    SUNNY = "sunny"
    RAINY = "rainy"
    FOGGY = "foggy"


# --- 2. 定义输入包 (System Context) ---
# 这是发给 A/B/C 模块的“考卷”，里面包含了所有能用的信息
@dataclass
class SystemContext:
    audio_frame: bytes = b''  # 原始音频 (给 A 用)
    asr_text: str = ""  # 识别文本 (给 B/C 用)
    speed: float = 0.0  # 车速 (给 C 用)
    weather: str = "sunny"  # 天气 (给 C 用)
    timestamp: float = field(default_factory=time.time)


# --- 3. 定义输出包 (Risk Report) ---
# 这是 A/B/C 模块交回来的“答卷”
@dataclass
class RiskReport:
    module_id: str  # 'A', 'B', 'C'
    risk_score: float  # 0.0 (安全) -> 1.0 (危险)
    suggestion: str  # 使用 DecisionType.value (PASS/BLOCK)
    reason: str  # "检测到车速过快" (用于 UI 显示)

    # 你的 evidence 我保留了，用于存放频谱图、原始文本等证据
    evidence: Dict[str, Any] = field(default_factory=dict)

    # 我增加了这个，用于在 UI 上显示“计算耗时：15ms”，体现高性能
    latency_ms: float = 0.0

    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return {
            "module_id": self.module_id,
            "risk_score": self.risk_score,
            "suggestion": self.suggestion,
            "reason": self.reason,
            "evidence": self.evidence,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp
        }