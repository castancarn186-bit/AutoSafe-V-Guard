# core/state.py

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModuleReport:
    """国家级竞赛标准：模块化风险报告结构"""
    module_id: str  # "A", "B", 或 "C"
    risk_score: float  # 0.0 ~ 1.0
    status: str  # "PASS", "BLOCK", "WARN", "SAFE"
    reason: str  # 拦截或放行的具体理由（可解释性核心）


@dataclass
class SharedState:
    """系统全局共享状态机"""
    # 基础状态
    is_running: bool = True

    # 环境状态
    vehicle_speed: int = 0
    weather: str = "晴朗"

    # 语音状态
    asr_text: str = "等待输入..."

    # 核心决策数据
    total_risk: float = 0.0
    decision: str = "PASS"  # PASS, BLOCK, WARN

    # 模块化报告列表 (UI 里的 X-RAY 视图数据源)
    latest_reports: List[ModuleReport] = field(default_factory=list)


# 全局单例
shared_state = SharedState()