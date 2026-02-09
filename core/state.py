import asyncio
from core.protocol import RiskReport

class SystemState:
    """全系统共享状态机"""
    def __init__(self):
        self.latest_reports = []    # 存储最近一次 A/B/C 的报告
        self.total_risk = 0.0       # 综合风险值
        self.decision = "WAITING"   # 最终决策
        self.is_running = True      # 系统运行开关

# 实例化一个全局唯一的变量
shared_state = SystemState()