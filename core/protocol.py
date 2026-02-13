# core/protocol.py
# core/protocol.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum
import time

class DecisionType(Enum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"

class WeatherType(Enum):
    SUNNY = "sunny"
    RAINY = "rainy"
    FOGGY = "foggy"

# --- 新增：攻击类型定义，用于演示 ---
class AttackType(Enum):
    NONE = "none"
    REPLAY = "replay"      # 声波重放攻击
    ADVERSARIAL = "adv"    # 对抗样本攻击
    SEMANTIC = "logic"     # 语义逻辑冲突

@dataclass
class SystemContext:
    audio_frame: bytes = b''
    asr_text: str = ""
    speed: float = 0.0
    weather: str = "sunny"
    timestamp: float = field(default_factory=time.time)

@dataclass
class RiskReport:
    module_id: str
    risk_score: float
    suggestion: str
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)
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