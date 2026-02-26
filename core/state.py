# core/state.py

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModuleReport:
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


class SharedState:
    def __init__(self):
        self.vehicle_speed = 0.0
        self.weather = "晴朗"
        self.asr_text = "等待输入..."
        self.total_risk = 0.0
        self.decision = "PASS"
        self.is_running = True
        self.latest_reports = []  # 存储 A, B, C 模块的报告

    def update_module_report(self, new_report):
        """核心修复：增加这个方法来处理模块报告"""
        # 检查是否已经存在该模块的报告，如果有则替换，没有则添加
        found = False
        for i, r in enumerate(self.latest_reports):
            if r.module_id == new_report.module_id:
                self.latest_reports[i] = new_report
                found = True
                break
        if not found:
            self.latest_reports.append(new_report)

        # 自动更新全局总风险分数（取所有模块的最大值）
        if self.latest_reports:
            self.total_risk = max([r.risk_score for r in self.latest_reports])
# 全局单例
shared_state = SharedState()