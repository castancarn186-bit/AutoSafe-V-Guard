#风险融合决策引擎	收集 A/B/C 的报告，执行加权算法，输出最终决策。
# core/engine.py
import random
import time
from typing import List, Tuple
from core.protocol import RiskReport, DecisionType, SystemContext


class VGuardEngine:
    def __init__(self):
        # 定义权重：A(声学) 30%, B(ASR) 30%, C(语义) 40%
        self.weights = {"A": 0.3, "B": 0.3, "C": 0.4}

    def generate_mock_reports(self) -> List[RiskReport]:
        """
        生成模拟数据，用于在没有真实传感器时测试 UI
        注意：这里必须匹配 protocol.py 的 RiskReport 定义
        """
        # 模拟 A 模块
        report_a = RiskReport(
            module_id="A",
            risk_score=random.uniform(0.0, 0.2),  # 模拟低风险
            suggestion=DecisionType.PASS.value,
            reason="声纹特征正常",  # <--- 修复点：补上这个参数
            evidence={"spectrogram": "normal"},
            latency_ms=12.5
        )

        # 模拟 B 模块
        report_b = RiskReport(
            module_id="B",
            risk_score=random.uniform(0.0, 0.3),
            suggestion=DecisionType.PASS.value,
            reason="置信度高",  # <--- 修复点：补上这个参数
            evidence={"confidence": 0.95},
            latency_ms=22.1
        )

        # 模拟 C 模块
        report_c = RiskReport(
            module_id="C",
            risk_score=random.uniform(0.0, 0.1),
            suggestion=DecisionType.PASS.value,
            reason="逻辑合规",  # <--- 修复点：补上这个参数
            evidence={"policy": "allow"},
            latency_ms=8.4
        )

        return [report_a, report_b, report_c]

    async def run_fusion(self, reports: List[RiskReport]) -> Tuple[float, str]:
        """
        风险融合算法：加权平均 + 一票否决
        """
        if not reports:
            return 0.0, "System Idle"

        total_score = 0.0
        details = []

        # 1. 加权计算
        for r in reports:
            w = self.weights.get(r.module_id, 0.0)
            total_score += r.risk_score * w

            # 2. 一票否决逻辑 (如果任意模块风险 > 0.8，直接 BLOCK)
            if r.risk_score > 0.8:
                return r.risk_score, DecisionType.BLOCK.value

        # 3. 最终判定
        if total_score > 0.7:
            decision = DecisionType.BLOCK.value
        elif total_score > 0.4:
            decision = DecisionType.WARN.value
        else:
            decision = DecisionType.PASS.value

        return round(total_score, 2), decision
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