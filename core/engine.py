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