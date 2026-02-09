#统一协议定义	定义 RiskReport 数据类，确保全链路数据格式一致。
from dataclasses import dataclass, field
from typing import Dict, Any
import time

@dataclass
class RiskReport:
    """
    统一安全报告协议
    所有子模块(A/B/C)必须返回此格式，否则引擎将拒绝处理
    """
    module_id: str             # 模块ID: 'A', 'B', 'C'
    risk_score: float          # 风险值: 0.0 (安全) -> 1.0 (极度危险)
    suggestion: str            # 建议操作: 'PASS' (放行), 'WARN' (警告), 'BLOCK' (拦截)
    evidence: Dict[str, Any] = field(default_factory=dict) # 证据链: 用于可视化展示
    timestamp: float = field(default_factory=time.time)    # 时间戳

    def to_dict(self):
        """将数据转为字典，方便 UI 界面和 API 调用"""
        return self.__dict__