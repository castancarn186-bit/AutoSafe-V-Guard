# core/state.py
from dataclasses import dataclass, field
from typing import List, Any
from core.protocol import DetectionResult


class SharedState:
    def __init__(self):
        # 基础环境
        self.is_running = True
        self.vehicle_speed = 0.0
        self.weather = "晴朗"

        # 语音与决策
        self.asr_text = "等待输入..."
        self.total_risk = 0.0
        self.decision = "PASS"
        self.execution_result = "待机"  # 修复模拟器崩溃点

        # UI 渲染依赖
        self.realtime_volume = 0.0
        self.latest_reports: List[DetectionResult] = []

    def update_module_report(self, new_report: DetectionResult):
        """更新单个模块的 X-Ray 视图"""
        found = False
        for i, r in enumerate(self.latest_reports):
            if r.module_id == new_report.module_id:
                self.latest_reports[i] = new_report
                found = True
                break
        if not found:
            self.latest_reports.append(new_report)


# 全局单例
shared_state = SharedState()