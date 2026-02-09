#风险融合决策引擎	收集 A/B/C 的报告，执行加权算法，输出最终决策。
import asyncio
import random
from typing import List
from core.protocol import RiskReport


class VGuardEngine:
    def __init__(self):
        # 定义每个模块的权重（加起来等于 1.0）
        self.weights = {
            "A": 0.4,  # 声学层关键
            "B": 0.3,
            "C": 0.3
        }

    async def run_fusion(self, reports: List[RiskReport]):
        """
        风险融合算法：加权求和
        """
        total_risk = 0.0
        for r in reports:
            weight = self.weights.get(r.module_id, 0.1)
            total_risk += r.risk_score * weight

        # 决策逻辑
        if total_risk > 0.7:
            decision = "BLOCK (拦截)"
        elif total_risk > 0.4:
            decision = "WARN (二次确认)"
        else:
            decision = "PASS (放行)"

        return total_risk, decision

    def generate_mock_reports(self) -> List[RiskReport]:
        """
        Mock 机制,模拟 1/2/3 三部分输出
        test
        """
        return [
            RiskReport(module_id="A", risk_score=random.uniform(0, 1), suggestion="MOCK"),
            RiskReport(module_id="B", risk_score=random.uniform(0, 1), suggestion="MOCK"),
            RiskReport(module_id="C", risk_score=random.uniform(0, 1), suggestion="MOCK")
        ]
'''
import asyncio
from typing import List
from core.protocol import RiskReport

class VGuardEngine:
    def __init__(self, modules: List):
        self.modules = modules
        # 风险权重分配 (这是答辩时可以展开讲的“算法优化点”)
        self.weights = {
            "A": 0.4,  # 声学物理层风险最高 (直接攻击)
            "B": 0.3,  # ASR 行为
            "C": 0.3   # 语义逻辑
        }

    async def run_pipeline(self, audio_data: bytes, vehicle_state: dict):
        """
        核心调度流水线
        """
        context = {"audio": audio_data, "state": vehicle_state}

        # 1. 并发调用 A/B/C 模块 (并行计算提升响应速度)
        # 就像三个专家同时对病人进行诊断
        tasks = [mod.detect(context) for mod in self.modules]
        reports: List[RiskReport] = await asyncio.gather(*tasks)

        # 2. 风险融合算法 (Weighted Sum Model)
        # 公式: $TotalRisk = \sum (Score_i \times Weight_i)$
        total_risk = sum(
            r.risk_score * self.weights.get(r.module_id, 0.1)
            for r in reports
        )

        # 3. 最终决策判决
        if total_risk > 0.7:
            decision = "BLOCK"
        elif total_risk > 0.4:
            decision = "WARN"
        else:
            decision = "PASS"

        return {
            "final_decision": decision,
            "total_risk": round(total_risk, 2),
            "breakdown": [r.to_dict() for r in reports]
        }
'''